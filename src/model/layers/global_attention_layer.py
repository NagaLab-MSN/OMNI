from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_scatter import scatter_mean, scatter_softmax, scatter_add

class GlobalAttentionLayer(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, ntypes: List[str]):
        super().__init__()
        self.ntypes = ntypes
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.type_attn_query = nn.Parameter(torch.FloatTensor(size=(1, 2 * in_feats)))
        self.node_attn_fc = nn.Linear(2 * in_feats, 1, bias=False)
        self.final_proj = nn.Linear(in_feats, out_feats)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.type_attn_query)
        nn.init.xavier_uniform_(self.node_attn_fc.weight)
        nn.init.xavier_uniform_(self.final_proj.weight)

    def forward(self, target_nodes_dict: Dict[str, torch.Tensor], full_features: Dict[str, torch.Tensor], cached_rwr: Dict[str, Dict[int, List[Tuple[str, int]]]]):
        global_embeds = {}
        device = self.type_attn_query.device

        for ntype, node_ids_cpu in target_nodes_dict.items():
            if ntype not in cached_rwr or node_ids_cpu.numel() == 0:
                continue

            batch_node_indices = []
            neighbor_ntypes = []
    
            ordered_nids_by_type = defaultdict(list)
            original_indices_by_type = defaultdict(list)
            current_neighbor_idx = 0

            ntype_map = {nt: i for i, nt in enumerate(self.ntypes)}

            for i, node_id in enumerate(node_ids_cpu.tolist()):
                neighs = cached_rwr.get(ntype, {}).get(node_id, [])
                for nt, nid in neighs:
                    if nt in full_features:
                        batch_node_indices.append(i)
                        neighbor_ntypes.append(ntype_map[nt])
                        ordered_nids_by_type[nt].append(nid)
                        original_indices_by_type[nt].append(current_neighbor_idx)
                        current_neighbor_idx += 1
    
            if not batch_node_indices:
                global_embeds[ntype] = torch.zeros(len(node_ids_cpu), self.out_feats, device=device)
                continue

            batch_node_indices = torch.tensor(batch_node_indices, dtype=torch.long, device=device)

            num_total_neighbors = current_neighbor_idx
            if num_total_neighbors == 0:
                global_embeds[ntype] = torch.zeros(len(node_ids_cpu), self.out_feats, device=device)
                continue

            features_by_type_list = [
                full_features[nt][torch.tensor(nids, dtype=torch.long)]
                for nt, nids in ordered_nids_by_type.items()
            ]
        
            if not features_by_type_list:
                global_embeds[ntype] = torch.zeros(len(node_ids_cpu), self.out_feats, device=device)
                continue
        
            unordered_feats_cpu = torch.cat(features_by_type_list, dim=0)

            combined_indices = torch.cat([
                torch.tensor(indices, dtype=torch.long)
                for indices in original_indices_by_type.values()
            ])
        
            flat_neighbor_feats_cpu = torch.zeros_like(unordered_feats_cpu)
            flat_neighbor_feats_cpu.scatter_(0, combined_indices.unsqueeze(1).expand_as(unordered_feats_cpu), unordered_feats_cpu)

            flat_neighbor_feats = flat_neighbor_feats_cpu.to(device, non_blocking=True)
        
            target_node_feats = full_features[ntype][node_ids_cpu].to(device)
        
            unique_groups, group_indices, _ = torch.unique(
                torch.stack([batch_node_indices, torch.tensor(neighbor_ntypes, device=device)]),
                dim=1, return_inverse=True, return_counts=True)
            
            type_agg_feats = scatter_mean(flat_neighbor_feats, group_indices, dim=0)
            group_target_node_indices = unique_groups[0, :]
            expanded_target_feats = target_node_feats[group_target_node_indices]
            type_inputs = torch.cat([expanded_target_feats, type_agg_feats], dim=1)
            type_logits = self.leaky_relu((type_inputs @ self.type_attn_query.T).squeeze(1))
            type_alphas = scatter_softmax(type_logits, group_target_node_indices, dim=0)
            alphas_per_neighbor = type_alphas[group_indices]
            expanded_target_for_nodes = target_node_feats[batch_node_indices]
            node_inputs = torch.cat([expanded_target_for_nodes, flat_neighbor_feats], dim=1)
            beta_logits = self.leaky_relu(self.node_attn_fc(node_inputs))
            beta_unnorm = torch.exp(beta_logits.squeeze(1)) * alphas_per_neighbor
            betas = scatter_softmax(beta_unnorm, batch_node_indices, dim=0)
            aggregated_feats = scatter_add(betas.unsqueeze(1) * flat_neighbor_feats, batch_node_indices, dim=0)
        
            if aggregated_feats.shape[0] < target_node_feats.shape[0]:
                final_target_feats = target_node_feats[:aggregated_feats.shape[0]]
            else:
                final_target_feats = target_node_feats

            final_global = aggregated_feats + final_target_feats
            global_embeds[ntype] = self.final_proj(final_global)
        
        return global_embeds
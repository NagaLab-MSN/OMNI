from collections import defaultdict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import dgl

from .gat_layer import GATLayer
from .global_attention_layer import GlobalAttentionLayer

class HeteroRelGAT(nn.Module):
    def __init__(self, rel_graph: dgl.DGLHeteroGraph, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.ntypes = rel_graph.ntypes
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.proj = nn.ModuleDict({nt: nn.Linear(in_dim, hidden_dim) for nt in self.ntypes})
        self.rel_gat = nn.ModuleDict({
            '_'.join(rel): GATLayer(hidden_dim, hidden_dim, num_heads, dropout) for rel in rel_graph.canonical_etypes
        })
        self.norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim * num_heads) for nt in self.ntypes})
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.global_layer = GlobalAttentionLayer(hidden_dim, hidden_dim * num_heads, self.ntypes)
        self.fusion = nn.ModuleDict({nt: nn.Linear((hidden_dim * num_heads) * 2, out_dim) for nt in self.ntypes})

    def forward(self, blocks: List, x_src: Dict[str, torch.Tensor], full_features_for_global:
        Dict[str, torch.Tensor], cached_rwr: Dict[str, Dict[int, List[Tuple[str, int]]]]):
        block = blocks[-1]
        h = {nt: self.proj[nt](x_src[nt]) for nt in x_src.keys()}
        h_dst_nodes = {ntype: h[ntype][:block.number_of_dst_nodes(ntype)] for ntype in block.dsttypes}
        outputs_by_dst = defaultdict(list)

        for rel in block.canonical_etypes:
            srctype, _, dsttype = rel
            if block[rel].num_edges() > 0:
                gat_key = '_'.join(rel)
                h_src_for_rel = h[srctype]
                h_dst_for_rel = h_dst_nodes[dsttype]
                out = self.rel_gat[gat_key](block[rel], (h_src_for_rel, h_dst_for_rel))
                outputs_by_dst[dsttype].append(out)

        final_local = {}
        for ntype, outs in outputs_by_dst.items():
            if not outs: continue
            agg = torch.stack(outs).mean(dim=0).reshape(outs[0].shape[0], -1)
            dst_rep = h_dst_nodes[ntype].repeat(1, self.num_heads)
            fused = self.act(self.norm[ntype](agg + dst_rep))
            final_local[ntype] = self.dropout(fused)

        target_nodes_dict = {nt: block.dstnodes[nt].data[dgl.NID].cpu() for nt in block.dsttypes}
        global_feats = self.global_layer(target_nodes_dict, full_features_for_global, cached_rwr)
    
        outputs = {}
        
        for ntype, h_dst in h_dst_nodes.items():
            
            if ntype in final_local:
                local_feat = final_local[ntype]
            else:
                local_feat = h_dst.repeat(1, self.num_heads)
        
            global_feat = global_feats.get(ntype)
            if global_feat is None or global_feat.shape[0] != local_feat.shape[0]:
                global_feat = torch.zeros_like(local_feat)
            
            combined_features = torch.cat([local_feat, global_feat.to(local_feat.device)], dim=1)
            outputs[ntype] = self.fusion[ntype](combined_features)
        
        return outputs
import argparse
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import dgl
import pandas as pd

from src.model.full_gnn_system import FullGNNSystem
from src.data.graph_datamodule import GraphDataModule
from src.model.layers.hetero_rel_gat import HeteroRelGAT

# ==============================================================================
#  RUNTIME PATCH
# ==============================================================================
class PatchedHeteroRelGAT(HeteroRelGAT):
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
                
                src_nodes_for_rel = block[rel].srcnodes()
                dst_nodes_for_rel = block[rel].dstnodes()
                
                h_src_for_rel = h[srctype][src_nodes_for_rel]
                h_dst_for_rel = h_dst_nodes[dsttype][dst_nodes_for_rel]
                
                out = self.rel_gat[gat_key](block[rel], (h_src_for_rel, h_dst_for_rel))
                full_dst_shape_out = torch.zeros(
                    (h_dst_nodes[dsttype].shape[0],) + out.shape[1:],
                    device=out.device)
                full_dst_shape_out[dst_nodes_for_rel] = out
                outputs_by_dst[dsttype].append(full_dst_shape_out)

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
# ==============================================================================

def predict_interaction(
    model: FullGNNSystem,
    dm: GraphDataModule,
    chemical_id: str,
    gene_id: str,
    device: torch.device
):
    """
    Predicts interaction types and their probabilities for a given chemical and gene ID.
    """
    model.eval()

    print("Updating the model's full feature cache...")
    model.update_full_features_cache()
    print("Cache update complete.")

    id_to_idx = dm.id_to_idx
    if chemical_id not in id_to_idx['chemical']:
        print(f"Error: Chemical ID '{chemical_id}' not found in the dataset.")
        return None
    if gene_id not in id_to_idx['gene']:
        print(f"Error: Gene ID '{gene_id}' not found in the dataset.")
        return None

    chem_idx = id_to_idx['chemical'][chemical_id]
    gene_idx = id_to_idx['gene'][gene_id]

    seed_nodes = {
        'chemical': torch.tensor([chem_idx]),
        'gene': torch.tensor([gene_idx])
    }
    
    sampler = dgl.dataloading.NeighborSampler([int(f) for f in model.hparams.fanouts.split(',')])
    _, _, blocks = sampler.sample(dm.g_train, seed_nodes)

    with torch.no_grad():
        blocks = [b.to(device) for b in blocks]

        x_src = {
            nt: model.model.embeds[nt](blocks[0].srcnodes[nt].data[dgl.NID].to(device))
            for nt in blocks[0].srctypes
        }

        node_embeds = model.model.encoder(
            blocks,
            x_src,
            model.full_features_cache,
            dm.rwr_train
        )
        
        h_u = node_embeds['chemical']
        h_v = node_embeds['gene']

        if h_u.shape[0] == 0 or h_v.shape[0] == 0:
            print("Error: Could not compute node embeddings for the given pair.")
            return None

        predictions = []
        chem_gene_etypes = [
            ('chemical', action, 'gene') for action in dm.chem_gene_actions
        ]

        for etype in chem_gene_etypes:
            relation_name = etype[1]
            key = '_'.join(etype)
            predictor = model.model.decoder.predictors[key]
            score = predictor(torch.cat([h_u, h_v], dim=1))
            prob = torch.sigmoid(score).item()
            predictions.append({
                "Relation_Type": relation_name,
                "Probability": prob
            })
    
    results_df = pd.DataFrame(predictions)
    results_df = results_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)
    
    return results_df

def main():
    parser = argparse.ArgumentParser("Manual Prediction Script for Chemical-Gene Interactions")
    parser.add_argument('--chemical_id', type=str, required=True, help="The ID of the chemical to test (e.g., 'D000041').")
    parser.add_argument('--gene_id', type=str, required=True, help="The ID of the gene to test (e.g., '1017').")
    parser.add_argument('--model_checkpoint', type=str, default='final_model_multi_rel.ckpt', help="Path to the trained model checkpoint (.ckpt) file.")
    parser.add_argument('--base_data_path', type=str, required=True, help="Path to the directory containing the source CSV files.")
    parser.add_argument('--cache_graph', type=str, default='cached_graph_multi_rel.pt')
    parser.add_argument('--cache_splits', type=str, default='cached_splits_multi_rel.pt')
    parser.add_argument('--cache_train_graph', type=str, default='cached_train_graph_multi_rel.pt')
    parser.add_argument('--cache_rwr_train', type=str, default='cached_rwr_train_multi_rel.pt')
    parser.add_argument('--cache_actions', type=str, default='chem_gene_actions_list.json')
    parser.add_argument('--feature_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--fanouts', type=str, default="15,10")
    # Suppress unused arguments
    for arg in ['force_reload', 'epochs', 'batch_size', 'gpus', 'num_nodes', 'num_workers', 'random_seed', 'neg_k', 'chunksize', 'rwr_len', 'rwr_restart_prob', 'rwr_top_k', 'model_output_file']:
        parser.add_argument(f'--{arg}', help=argparse.SUPPRESS)

    args = parser.parse_args()

    print("--- Initializing DataModule to load graph artifacts ---")
    # Force num_workers to 0 for prediction as it's not needed
    args.num_workers = 0 
    dm = GraphDataModule(args)
    dm.setup()
    
    print(f"\n--- Loading model from checkpoint: {args.model_checkpoint} ---")
    
    g_meta = dm.g_train
    all_etypes = dm.g_full.canonical_etypes

    model = FullGNNSystem.load_from_checkpoint(
        args.model_checkpoint,
        g_meta=g_meta,
        all_etypes=all_etypes,
        hparams=args
    )

    print("--- Patching the loaded model's encoder at runtime ---")
    patched_encoder = PatchedHeteroRelGAT(
        g_meta, model.hparams.feature_dim, model.hparams.hidden_dim,
        model.hparams.out_dim, model.hparams.num_heads, model.hparams.dropout
    )
    patched_encoder.load_state_dict(model.model.encoder.state_dict())
    model.model.encoder = patched_encoder
    print("--- Patch successful ---")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    print(f"\n--- Predicting interactions for Chemical '{args.chemical_id}' and Gene '{args.gene_id}' ---")
    
    results = predict_interaction(model, dm, args.chemical_id, args.gene_id, device)

    if results is not None:
        print("\n" + "="*50)
        print("           PREDICTION RESULTS")
        print("="*50)
        print(results.to_string())
        print("="*50)

if __name__ == "__main__":
    main()
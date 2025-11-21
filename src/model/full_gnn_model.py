from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import dgl

from .layers.hetero_rel_gat import HeteroRelGAT

class EdgeDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, relation_etypes: List[Tuple[str,str,str]]):
        super().__init__()
        self.predictors = nn.ModuleDict({
            '_'.join(et): nn.Sequential(
                nn.Linear(in_dim*2, hidden_dim), 
                nn.ReLU(), 
                nn.Dropout(dropout), 
                nn.Linear(hidden_dim, 1)
            ) for et in relation_etypes
        })

    def forward(self, pair_graph: dgl.DGLHeteroGraph, node_embed: Dict[str, torch.Tensor]):
        logits = {}
        for rel in pair_graph.canonical_etypes:
            key = '_'.join(rel)
            if key in self.predictors:
                u, v = pair_graph.edges(form='uv', etype=rel)
                if u.numel() == 0: continue
                h_u, h_v = node_embed[rel[0]][u], node_embed[rel[2]][v]
                logits[rel] = self.predictors[key](torch.cat([h_u, h_v], dim=1)).squeeze(1)
        return logits

class FullGNNModel(nn.Module):
    def __init__(self, g_meta: dgl.DGLHeteroGraph, feat_dim: int, hidden_dim: int, out_dim: int, 
                 num_heads: int, dropout: float, relation_etypes: List[Tuple[str,str,str]]):
        super().__init__()
        self.embeds = nn.ModuleDict({nt: nn.Embedding(g_meta.num_nodes(nt), feat_dim) for nt in g_meta.ntypes})
        self.encoder = HeteroRelGAT(g_meta, feat_dim, hidden_dim, out_dim, num_heads, dropout)
        self.decoder = EdgeDecoder(out_dim, hidden_dim // 2, dropout, relation_etypes)

    def forward(self,
                blocks: List,
                pair_graph: dgl.DGLHeteroGraph,
                neg_pair_graph: Optional[dgl.DGLHeteroGraph],
                cached_rwr: Dict[str, Dict[int, List[Tuple[str, int]]]],
                full_features_cache: Dict[str, torch.Tensor]):
        x_src = {nt: self.embeds[nt](blocks[-1].srcnodes[nt].data[dgl.NID]) for nt in blocks[-1].srctypes}
        node_out = self.encoder(blocks, x_src, full_features_cache, cached_rwr)
        pos_logits = self.decoder(pair_graph, node_out)
        neg_logits = self.decoder(neg_pair_graph, node_out) if neg_pair_graph else None
        return pos_logits, neg_logits
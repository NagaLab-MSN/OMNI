import torch
import torch.nn as nn
import dgl
import dgl.function as fn

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_heads, dropout):
        super().__init__()
        self.num_heads, self.out_feats = num_heads, out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.empty(1, num_heads, out_feats))
        self.attn_r = nn.Parameter(torch.empty(1, num_heads, out_feats))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)

    def forward(self, g, feat):
        with g.local_scope():
            h_src, h_dst = (feat, feat) if not isinstance(feat, tuple) else feat
            feat_src_fc = self.dropout(self.fc(h_src)).view(-1, self.num_heads, self.out_feats)
            feat_dst_fc = self.dropout(self.fc(h_dst)).view(-1, self.num_heads, self.out_feats)
            el = (feat_src_fc * self.attn_l).sum(dim=-1, keepdim=True)
            er = (feat_dst_fc * self.attn_r).sum(dim=-1, keepdim=True)
            g.srcdata.update({'ft': feat_src_fc, 'el': el})
            g.dstdata.update({'er': er})
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            alpha = dgl.ops.edge_softmax(g, e)
            g.edata['alpha'] = alpha
            g.update_all(fn.u_mul_e('ft', 'alpha', 'm'), fn.sum('m', 'h_out'))
            return g.dstdata['h_out']
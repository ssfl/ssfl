import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from common import device


__author__ = "Marc: thanhdatn@student.unimelb.edu.au"
edge_types = ['tb', 'lr', 'bt', 'child', 'parent', 'master']
e2idmap = [(e, i) for i, e in enumerate(edge_types)]

class MPNN(MessagePassing):
    def __init__(self, dim_in, dim_out, aggr='max'):
        super().__init__(aggr=aggr)
        self.lin_tgt = nn.Linear(dim_in, dim_out)
        self.lin_src = nn.Linear(dim_in, dim_out)
        self.emb_self_edge = nn.Embedding(2, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x_src, x_dst, es, weights=None):
        return self.relu(self.propagate(es, x=(x_src, x_dst),
                              size=(x_src.size(0),x_dst.size(0)),
                              weights=weights))

    def message(self, x_i, x_j, edge_index, weights=None):
        return self.lin_tgt(x_i) + self.lin_src(x_j) + self.emb_self_edge((edge_index[1] == edge_index[0]).long())


class MPNNModel(nn.Module):
    def __init__(self, dim_cl, dim_cc,
                 dim_h, edim, netypes, t_srcs, t_tgts,
                 dim_al, dim_ac, n_layers=5, device=device):
        super().__init__()
        self.enc_cl = nn.Linear(dim_cl, dim_h)
        self.enc_cc = nn.Linear(dim_cc, dim_h)

        self.enc_al = nn.Linear(dim_al, dim_h)
        self.enc_ac = nn.Linear(dim_ac, dim_h)

        nn.init.xavier_normal_(self.enc_al.weight)
        nn.init.normal_(self.enc_al.bias)
        nn.init.xavier_normal_(self.enc_al.weight)
        nn.init.normal_(self.enc_ac.bias)

        self.emb_pt = nn.Parameter(torch.FloatTensor(dim_h))
        nn.init.normal_(self.emb_pt)
        self.emb_ft = nn.Parameter(torch.FloatTensor(dim_h))
        nn.init.normal_(self.emb_ft)
        self.t_srcs = t_srcs
        self.t_tgts = t_tgts

        self.netypes = netypes
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        self.mpnns = nn.ModuleList(
            [nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(dim_h, dim_h), nn.ReLU())
                    for _ in range(self.netypes)])
                for _ in range(self.n_layers)])


    def forward(self, xs, ess, weights=None):
        '''xs: cl, cc, al, ac, pt, ft'''
        xs = [self.enc_cl(xs[0]) + self.enc_cc(xs[1]),
              self.enc_ac(xs[2]) + self.enc_ac(xs[3]),
              self.emb_pt(xs[4]), self.emb_ft(xs[5])]
        for i in range(self.n_layers):
            out = [0, 0, 0, 0]
            for j, (es, t_src, t_tgt) in enumerate(
                    zip(ess, self.t_srcs, self.t_tgts)):
                out[t_tgt] += self.mpnns[i][j](xs[t_src], xs[t_tgt], es,
                                               weights[j])
            xs = self.relu(out)
        return xs


from torch.utils.data import Dataset
from codeflaws.dataloader_cfl import CodeflawsCFLNxStatementDataset, \
    CodeflawsCFLStatementGraphMetadata
from utils.data_utils import NxDataloader
from utils.utils import ConfigClass
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from Typing import List
import os
from torch_geometric import Data
from torch_geometric.utils import add_self_loops
import networkx as nx
import torch.utils.data


class CodeflawsCFLPyGStatementDataset(Dataset):
    def __init__(self, dataloader: NxDataloader,
                 meta_data: CodeflawsCFLStatementGraphMetadata,
                 ast_enc=None,
                 save_dir=ConfigClass.preprocess_dir_codeflaws,
                 name='pyg_cfl_stmt'):
        self.dataloader = dataloader
        self.meta_data = meta_data if meta_data else\
            CodeflawsCFLStatementGraphMetadata(dataloader.get_dataset())
        self.save_dir = save_dir
        self.vocab_dict = dict(tuple(line.split()) for line in open(
            'preprocess/codeflaws_vocab.txt', 'r'))
        self.name = name
        self.graph_save_path = f"{save_dir}/{name}.pkl"
        self.info_path = f"{save_dir}/{name}_info.pkl"
        self.ast_enc = ast_enc
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

        def has_cache(self):
            return os.path.exists(self.graph_save_path) &\
                os.path.exists(self.info_path)

        def __len__(self):
            return len(self.gs)

        def __getitem__(self, i):
            return self.gs[i], self.gs_stmt_nodes[i]

        def convert_from_nx_to_pyg(self, nx_g, stmt_nodes):
            nx_g = augment_with_reverse_edge_cat(nx_g, self.meta_data.t_e_asts,
                                                 [])
            ori_ns = list(nx_g.nodes())[:]
            nx_g = nx.convert_node_labels_to_integers(nx_g)
            new_ns = list(nx_g.nodes())[:]
            map_ns = {n: i for n, i in zip(ori_ns, new_ns)}
            ess = [[[], []] for i in range(len(self.meta_data.t_all))]
            for u, v, e in nx_g.edges(data=True):
                e['etype'] = self.meta_data.t_all.index(e['etype'])
                ess[e['etype']][0].append(u)
                ess[e['etype']][1].append(v)
            ess = [add_self_loops(torch.tensor(es).long())[0] for es in ess]
            data = Data(ess=ess)
            n_asts = [n for n in nx_g if nx_g.nodes[n]['graph'] == 'ast']
            l_a = torch.tensor(
                [self.meta_data.ntype2id[nx_g.nodes[n]['ntype']]
                 for n in n_asts]).long()
            if self.ast_enc is not None:
                data.c_a = torch.tensor([
                    self.ast_enc(nx_g.nodes[n]['token']) for n in n_asts]
                ).float()
            # for cfg, it will be text
            data.lbl = torch.tensor([nx_g.nodes[n]['status'] for n in n_asts])
            ts = torch.tensor([0] * (len(nx_g.nodes()) - len(n_asts)))
            data.xs = [l_a, ts]
            return data, \
                torch.tensor(list(map_ns[n] for n in stmt_nodes)).int()

        def process(self):
            self.gs, self.gs_stmt_nodes = []
            for nx_g, stmt_nodes in self.dataloader:
                g, g_stmt_nodes = self.convert_from_nx_to_pyg(nx_g, stmt_nodes)
                self.gs.append(g)
                self.gs_stmt_nodes.append(g_stmt_nodes)
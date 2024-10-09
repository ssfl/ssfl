from typing import Tuple, List, Union
import random
from collections import Sequence
from itertools import accumulate
from typing import Callable
import networkx as nx
from utils.nprocs import nprocs
from utils.multiprocessing import multi_process_data, Queue
from graph_algos.nx_shortcuts import nodes_where, where_node, edges_where
import pickle as pkl
import os
import tqdm


class NxDataset(Sequence):

    def __init__(self):
        if self.has_cache():
            self.load()
        else:
            self.process()
            self.save()

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def has_cache(self):
        raise NotImplementedError


class AstNxDataset(NxDataset):

    def __init__(
        self,
        all_entries,
        process_func,
        save_dir,
        name,
        special_attrs: List[Tuple[str, Callable[[nx.Graph], None]]],
        post_process_func=None,
    ):
        self.save_dir = save_dir
        self.info_path = f"{save_dir}/nx_{name}_info.pkl"
        self.name = name
        self.process_func = process_func
        self.all_entries = all_entries
        self.special_attrs = special_attrs
        self.post_process_func = post_process_func
        for k, _ in self.special_attrs:
            setattr(self, k, [])
        super().__init__()

    def has_cache(self):
        return os.path.exists(self.info_path)

    def __len__(self):
        return len(self.active_idxs)

    def __getitem__(self, i):
        try:
            nx_g = pkl.load(
                open(f"{self.save_dir}/nx_{self.name}_{self.active_idxs[i]}.pkl", "rb")
            )
        except UnicodeDecodeError:
            nx_g = self.process_func(self.all_entries[self.active_idxs[i]])
            if self.post_process_func:
                nx_g = self.post_process_func(nx_g)

            pkl.dump(
                nx_g,
                open(f"{self.save_dir}/nx_{self.name}_{self.active_idxs[i]}.pkl", "wb"),
            )
        return [nx_g] + [self.__dict__[k][i] for k, _ in self.special_attrs]

    def process(self):
        self.ast_types = set()
        self.ast_etypes = set()
        self.keys = []
        self.err_idxs = []
        self.active_idxs = []

        bar = tqdm.tqdm(list(self.all_entries))
        bar.set_description(f"Loading Nx Data {self.name}")
        for i, key in enumerate(bar):
            try:
                if not os.path.exists(f"{self.save_dir}/nx_{self.name}_{i}.pkl"):
                    nx_g = self.process_func(key)
                    pkl.dump(
                        nx_g, open(f"{self.save_dir}/nx_{self.name}_{i}.pkl", "wb")
                    )
                else:
                    nx_g = pkl.load(
                        open(f"{self.save_dir}/nx_{self.name}_{i}.pkl", "rb")
                    )
            except Exception as ex:
                print(type(ex).__name__, ex.args)
                self.err_idxs.append(i)
                count = len(self.err_idxs)
                bar.set_postfix(syntax_error_files=count)
                continue
            self.active_idxs.append(i)
            self.keys.append(key)
            self.ast_types = self.ast_types.union(
                [nx_g.nodes[n]["ntype"] for n in nodes_where(nx_g, graph="ast")]
            )
            self.ast_etypes = self.ast_etypes.union(
                x[-1]["label"]
                for x in edges_where(
                    nx_g, where_node(graph="ast"), where_node(graph="ast")
                )
            )
            for k, f in self.special_attrs:
                self.__dict__[k].append(f(nx_g))

    def load(self):
        info_dict = pkl.load(open(self.info_path, "rb"))
        for k, v in info_dict.items():
            setattr(self, k, v)

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        # gs is saved somewhere else
        pkl.dump(
            {
                "ast_types": self.ast_types,
                "ast_etypes": self.ast_etypes,
                "keys": self.keys,
                "err_idxs": self.err_idxs,
                "active_idxs": self.active_idxs,
                **{k: self.__dict__[k] for k, _ in self.special_attrs},
            },
            open(self.info_path, "wb"),
        )


class NxDataloader(Sequence):

    def __init__(self, nx_dataset: NxDataset, idxs):
        self.idxs = idxs
        self.nx_dataset = nx_dataset

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        return self.nx_dataset[self.idxs[i]]

    def get_dataset(self) -> NxDataset:
        return self.nx_dataset


def split_nx_dataset(
    nx_dataset: NxDataset, ratio: Union[int, List[int]] = 0.8, shuffle=True
) -> Tuple[NxDataset, NxDataset]:
    N = len(nx_dataset)
    idxs = list(range(N))
    if shuffle:
        random.shuffle(idxs)
    if isinstance(ratio, list):
        assert len(ratio), "Ratio must not be empty List"
        assert sum(ratio) == 1, "Sum of ratio must be 1"
        cumsum = list(accumulate(ratio))
        out = []
        for i, c in enumerate(cumsum):
            if i:
                out.append(
                    NxDataloader(nx_dataset, idxs[int(cumsum[i - 1] * N) : int(c * N)])
                )
            else:
                out.append(NxDataloader(nx_dataset, idxs[: int(c * N)]))
        return out
    return NxDataloader(nx_dataset, idxs[: int(ratio * N)]), NxDataloader(
        nx_dataset, idxs[int(N * ratio) :]
    )


def del_all_status(nx_g):
    n_asts, n_cfgs = nodes_where(nx_g, graph="ast"), nodes_where(nx_g, graph="cfg")
    for n in n_asts:
        del nx_g.nodes[n]["status"]
    for n in n_cfgs:
        del nx_g.nodes[n]["status"]
    return nx_g


class AstGraphMetadata(object):

    def __init__(self, nx_g_dataset):
        self.t_asts = sorted(nx_g_dataset.ast_types)
        self.ntype2id = {n: i for i, n in enumerate(sorted(self.t_asts))}
        self.id2ntype = self.t_asts
        self.t_e_asts = sorted(nx_g_dataset.ast_etypes)
        if "cfg_etypes" in nx_g_dataset.__dict__:
            self.t_e_cfgs = sorted(nx_g_dataset.cfg_etypes)
        self.meta_graph = self.construct_edge_metagraph()

    def construct_edge_metagraph(self):
        self.t_e_a_a = list(sorted(self.t_e_asts)) + list(
            map(lambda x: f"{x}_reverse", sorted(list(self.t_e_asts)))
        )
        if "t_e_cfgs" in self.__dict__:
            self.t_e_c_c = (
                self.t_e_cfgs
                + ["c_self_loop"]
                + [et + "_reverse" for et in self.t_e_cfgs]
            )
            self.t_e_c_a = ["corresponding_ast"]
            self.t_e_a_c = ["corresponding_cfg"]
            self.t_e_c_t = ["c_pass_t", "c_fail_t"]
            self.t_e_t_c = ["t_pass_c", "t_fail_c"]

        self.t_e_a_t = ["a_pass_t", "a_fail_t"]
        self.t_e_t_a = ["t_pass_a", "t_fail_a"]
        self.t_all = self.t_e_a_a + self.t_e_a_t + self.t_e_t_a
        self.srcs = (
            ["ast"] * len(self.t_e_a_a)
            + ["ast"] * len(self.t_e_a_t)
            + ["test"] * len(self.t_e_t_a)
        )
        self.dsts = (
            ["ast"] * len(self.t_e_a_a)
            + ["test"] * len(self.t_e_a_t)
            + ["ast"] * len(self.t_e_t_a)
        )
        if "t_e_cfgs" in self.__dict__:
            self.t_all += (
                self.t_e_c_c + self.t_e_c_a + self.t_e_a_c + self.t_e_c_t + self.t_e_t_c
            )
            self.srcs += ["cfg", "cfg", "ast", "cfg", "test"]
            self.dsts += ["cfg", "ast", "cfg", "test", "cfg"]

        return list(zip(self.srcs, self.t_all, self.dsts))

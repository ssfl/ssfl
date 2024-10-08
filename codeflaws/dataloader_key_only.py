from utils.utils import ConfigClass
from codeflaws.data_utils import get_cfg_ast_cov, all_codeflaws_keys
from utils.nx_graph_builder import augment_with_reverse_edge_cat
from graph_algos.nx_shortcuts import (
    nodes_where,
    where_node_not,
    edges_where,
    where_node,
)
from utils.data_utils import AstNxDataset, del_all_status
import os
import random
import pickle as pkl
import json
import torch
import tqdm
from collections import defaultdict
from pycparser.plyparser import ParseError

errorneous_keys = json.load(open("error_instance.json", "r"))
non_err_keys = [k for k in all_codeflaws_keys if k not in errorneous_keys]


class CodeflawsNxDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(
            all_entries=all_codeflaws_keys,
            process_func=lambda k: get_cfg_ast_cov(k)[-1],
            save_dir=save_dir,
            name="key_only",
            special_attrs={
                "ast_lb_d": lambda nx_g: nodes_where(nx_g, graph="ast", status=1),
                "ast_lb_i": lambda nx_g: nodes_where(nx_g, graph="ast", status=2),
                "cfg_lb": lambda nx_g: nodes_where(nx_g, graph="cfg", status=1),
            },
            post_process_func=del_all_status,
        )
        self.cfg_etypes = ["parent_child", "next", "ref", "func_call"]

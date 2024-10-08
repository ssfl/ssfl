from dgl.data import DGLDataset
from dgl import save_graphs, load_graphs
import dgl
from utils.utils import ConfigClass
from codeflaws.data_utils import (
    all_codeflaws_keys,
    get_nx_ast_node_annt_gumtree,
    get_nx_ast_stmt_annt_gumtree,
)
from utils.gumtree_utils import GumtreeASTUtils
from utils.data_utils import AstNxDataset, AstGraphMetadata
from graph_algos.nx_shortcuts import (
    nodes_where,
    edges_where,
    where_node,
    where_node_not,
)
from collections import defaultdict
import os
import random
import pickle as pkl
import json
import torch
import tqdm

from json import JSONDecodeError


class CodeflawsGumtreeNxStatementDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(
            all_entries=all_codeflaws_keys,
            process_func=get_nx_ast_stmt_annt_gumtree,
            save_dir=save_dir,
            name="gumtree_stmt",
            special_attrs=[
                (
                    "stmt_nodes",
                    lambda nx_g: nodes_where(
                        nx_g,
                        lambda x: GumtreeASTUtils.check_is_stmt_cpp(
                            nx_g.nodes[x]["ntype"], graph="ast"
                        ),
                    ),
                )
            ],
        )


class CodeflawsGumtreeNxNodeDataset(AstNxDataset):
    def __init__(self, save_dir=ConfigClass.preprocess_dir_codeflaws):
        super().__init__(
            all_entries=all_codeflaws_keys,
            process_func=get_nx_ast_node_annt_gumtree,
            save_dir=save_dir,
            name="gumtree_node",
            special_attrs=[],
        )

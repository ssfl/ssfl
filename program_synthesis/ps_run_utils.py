from utils.ps_utils import FindProgram, NodeVariable, RelationVariable, ExcludeProgram
from networkx import isomorphism
import networkx as nx
from collections import defaultdict, Counter
from typing import List, Tuple, Dict
import numpy as np
import itertools


def batch_find_program_executor(nx_g, find_programs: List[FindProgram]) -> List[
    List[
        Tuple[
            Dict[NodeVariable, str],
            Dict[RelationVariable, Tuple[NodeVariable, NodeVariable, int]],
        ]
    ]
]:
    if not find_programs:
        return []
    # strategy to speed up program executor:
    # find all program that have same set of path (excluding label)
    # iterate through all binding
    # and then test. In this way, we do not have to perform isomorphism multiple times
    assert all(
        isinstance(f, FindProgram) for f in find_programs
    ), "All programs must be FindProgram"
    # First, group programs by their path
    path_to_programs = defaultdict(list)
    for i, f in enumerate(find_programs):
        path_to_programs[tuple(f.relation_constraint)].append((i, f))

    out_words = [[] for _ in range(len(find_programs))]
    for path in path_to_programs:
        nx_graph_query = nx.MultiDiGraph()
        word_vars = path_to_programs[path][0][1].word_variables
        for w in word_vars:
            nx_graph_query.add_node(w)
        for w1, w2, r in path:
            nx_graph_query.add_edge(w1, w2, key=0)

        # print(nx_g.nodes(), nx_g.edges())
        gm = isomorphism.MultiDiGraphMatcher(nx_g, nx_graph_query)
        # print(nx_graph_query.nodes(), nx_graph_query.edges(), gm.subgraph_is_isomorphic(), gm.subgraph_is_monomorphic())
        for subgraph in gm.subgraph_monomorphisms_iter():
            subgraph = {v: k for k, v in subgraph.items()}
            # get the corresponding binding for word_variables and relation_variables
            word_binding = {w: subgraph[w] for w in word_vars}
            relation_binding = {
                r: (subgraph[w1], subgraph[w2], 0) for w1, w2, r in path
            }
            # word_val = {w: nx_g.nodes[word_binding[w]] for i, w in enumerate(word_vars)}
            # relation_val = {r: (nx_g.nodes[word_binding[w1]], nx_g.nodes[word_binding[w2]], 0) for w1, w2, r in path}

            for i, f in path_to_programs[path]:
                val = f.evaluate_binding(word_binding, relation_binding, nx_g)
                if val:
                    out_words[i].append((word_binding, relation_binding))
    return out_words


def batch_program_executor(nx_g, ps, fps) -> List[int]:
    out_bindings = batch_find_program_executor(nx_g, fps)
    eval_mappings = {}
    w0 = NodeVariable("w0")
    for j, p_bindings in enumerate(out_bindings):
        return_var = fps[j].return_variables[0]
        eval_mappings[fps[j]] = []
        for w_binding, r_binding in p_bindings:
            wlast = w_binding[return_var]
            eval_mappings[fps[j]].append(wlast)
    ps = [p.replace_find_programs_with_values(eval_mappings) for p in ps]
    nodes = list(itertools.chain(*[p.evaluate(nx_g) for p in ps]))
    return nodes


def get_counter_programs(ps_linking):
    counter_programs = set()
    for p in ps_linking:
        if isinstance(p, ExcludeProgram):
            counter_programs.update(p.excl_programs)
    return counter_programs


def link_entity(
    data,
    nx_g,
    ps,
    fps,
    ps_counter=[],
    use_rem_sem=False,
):
    targets = batch_program_executor(nx_g, ps, fps)

    nodes = []
    if use_rem_sem:
        nodes_counter = batch_program_executor(nx_g, ps_counter, fps)
        nodes_sem = [n for n, d in nx_g.nodes(keys=True) if nx_g.nodes[n]["rank"] <= 5]
        nodes_sem = set(nodes_sem) - set(nodes_counter)
        nodes = itertools.chain(nodes, nodes_sem)
    return nodes

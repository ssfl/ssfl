import networkx as nx
from typing import List, Tuple, Dict, Set, Optional
from utils.draw_utils import cfg_to_agraph, ast_to_agraph, cfg_ast_to_agraph

from utils.relation_building_utils import (
    calculate_relation_set,
    dummy_calculate_relation_set,
    calculate_relation,
)
import argparse
import numpy as np
import itertools
import functools
from collections import defaultdict, namedtuple
from networkx.algorithms import constraint, isomorphism
from .ps_utils import (
    FalseValue,
    LiteralReplacement,
    Program,
    GrammarReplacement,
    FindProgram,
    RelationLabelConstant,
    RelationLabelProperty,
    TrueValue,
    UnionProgram,
    NodeLabelProperty,
    NodeVariable,
    RelationVariable,
    RelationConstraint,
    LabelEqualConstraint,
    RelationLabelEqualConstraint,
    construct_fl_specs,
    LabelConstant,
    AndConstraint,
    LiteralSet,
    Constraint,
    Hole,
    replace_hole,
    find_holes,
    SymbolicList,
    FilterStrategy,
    fill_hole,
    Expression,
    FloatConstant,
)
from utils.visualization_script import visualize_program_with_support
from .version_space import (
    VersionSpace as VS,
    construct_counter_program,
    construct_constraints_to_valid_version_spaces,
    get_intersect_constraint_vs,
    join_counter_vss,
    join_intersect_vss
)
import json
import pickle as pkl
import os
import tqdm
import copy
import multiprocessing
from multiprocessing import Pool
from .ps_run_utils import batch_find_program_executor
from .decisiontree_ps import (
    construct_or_get_initial_programs,
    report_metrics,
    add_constraint_to_find_program,
    get_args,
    logger,
    setup_grammar,
    setup_cache_dir,
    setup_dataset,
    check_add_perfect_program,
)
from utils.metrics import get_p_r_f1
from utils.misc import (
    mapping2tuple,
    tuple2mapping,
    pexists,
    pjoin,
    mappings2linking_tuples,
)
import cv2
import time

SpecType = List[Tuple[int, List[int]]]

TASK = "linking"


def setup_specs(args, dataset, task):
    assert task in ["locate"]
    if pexists(f"{args.cache_dir}/specs_{task}.pkl"):
        with open(f"{args.cache_dir}/specs_{task}.pkl", "rb") as f:
            specs = pkl.load(f)
    else:
        specs = construct_fl_specs(dataset)

        with open(f"{args.cache_dir}/specs_{task}.pkl", "wb") as f:
            pkl.dump(specs, f)
    return specs


def get_all_path(nx_g, w1, w2, hops=2):
    path_set_counter = defaultdict(int)
    for path in nx.all_simple_paths(nx_g, w1, w2, cutoff=hops):
        all_path_types = []
        for i in range(len(path) - 1):
            # get the relation between path[i] and path[i+1]
            new_rels = []
            for e in nx_g.edges((path[i], path[i + 1]), data=True):
                new_rels = (
                    nx_g.nodes[path[i]]["label"],
                    e[2]["lbl"],
                    nx_g.nodes[path[i + 1]]["label"],
                )
            if len(all_path_types) == 0:
                all_path_types = [new_rels]
            else:
                all_path_types = [x + new_rels for x in all_path_types]
        for path_type in all_path_types:
            path_type = tuple(path_type)
            path_set_counter[path_type] += 1
    for path_type, count in path_set_counter.items():
        yield path_type, count


def construct_initial_program_set(all_positive_paths):
    print("Constructing initial program set")
    programs = []
    for path, count in all_positive_paths:
        # split path in to ((w0, r0, w1), (w1, r1, w2), ...)
        path = [tuple(path[i : i + 3]) for i in range(0, len(path), 3)]
        word_labels = [LabelConstant(x[0]) for x in path] + [LabelConstant(path[-1][2])]
        rel_labels = [RelationLabelConstant(x[1]) for x in path]
        word_vars = list([NodeVariable(f"w{i}") for i in range(len(word_labels))])
        rel_vars = list([RelationVariable(f"r{i}") for i in range(len(rel_labels))])
        relation_constraints = [
            RelationConstraint(word_vars[i], word_vars[i + 1], rel_vars[i])
            for i in range(len(word_vars) - 1)
        ]
        # Now add constraints for word
        label_constraint = [
            LabelEqualConstraint(NodeLabelProperty(word_vars[i]), word_labels[i])
            for i in range(len(word_vars))
        ]
        label_constraint = functools.reduce(
            lambda x, y: AndConstraint(x, y), label_constraint
        )
        relation_label_constraint = [
            RelationLabelEqualConstraint(
                RelationLabelProperty(rel_vars[i]), rel_labels[i]
            )
            for i in range(len(rel_vars))
        ]
        relation_label_constraint = functools.reduce(
            lambda x, y: AndConstraint(x, y), relation_label_constraint
        )
        # Label constraitn for relations
        return_vars = [word_vars[-1]]
        programs.append(
            FindProgram(
                word_vars,
                rel_vars,
                relation_constraints,
                AndConstraint(label_constraint, relation_label_constraint),
                return_vars,
            )
        )
    return programs


def construct_or_get_initial_programs(pos_paths, cache_fp, logger=logger):
    if pexists(cache_fp):
        with open(cache_fp, "rb") as f:
            programs = pkl.load(f)
    else:
        start_time = time.time()
        programs = construct_initial_program_set(pos_paths)
        logger.log("Construct initial program set", float(time.time() - start_time))
        with open(cache_fp, "wb") as f:
            pkl.dump(programs, f)
    return programs


def get_starting_nodes(nx_g, n, hops: int):
    # The algorithm to find possible nodes to start from
    # Two strategies:
    # 1. Find all nodes that are within hops distance from n
    # 2. Find an AST nodes that are within hops distance from n that are also covered by the same of bigger set of test
    cover_fail_set = set(
        [
            n2
            for n2 in nx_g.neighbors(n)
            if nx_g.nodes[n2]["graph"] == "test"
            and nx_g.edges[(n2, n)][0]["label"] == "t_fail_a"
        ]
    )
    for i in range(hops):
        for path in nx.all_simple_paths(nx_g, n, n, cutoff=i):
            if nx_g.nodes(path[-1])["graph"] == "ast":
                cover_fail_set_source = set(
                    [
                        n2
                        for n2 in nx_g.neighbors(path[-1])
                        if nx_g.nodes[n2]["graph"] == "test"
                        and nx_g.edges[(n2, path[-1])][0]["label"] == "t_fail_a"
                    ]
                )
                if cover_fail_set.issubset(cover_fail_set_source):
                    yield path[-1]


def get_all_positive_relation_paths_fl(
    specs: SpecType, relation_set, hops=2, dataset: List[nx_g.MultiDiGraph]
):
    path_set_counter = defaultdict(int)
    bar = tqdm.tqdm(specs, total=len(specs))
    bar.set_description("Mining positive relations")
    for i, faulty_nodes in bar:
        nx_g = dataset[i]
        for n in faulty_nodes:
            for n2 in get_starting_nodes(nx_g, n, hops=hops):
                for path, count in get_all_path(nx_g, n2, n, hops=hops):
                    path_set_counter[path] += count
    for path_type, count in path_set_counter.items():
        yield path_type, count


def get_path_specs_fl(
    dataset,
    specs: SpecType,
    relation_set,
    hops=2,
    dataset: List[nx_g.MultiDiGraph],
    cache_dir=None,
):
    data_sample_set_relation = (
        {} if data_sample_set_relation_cache is None else data_sample_set_relation_cache
    )
    assert data_sample_set_relation_cache is not None
    pos_relations = []
    neg_rels = []
    print("Start mining positive relations")
    if pexists(f"{cache_dir}/all_positive_paths.pkl"):
        pos_relations = pkl.load(open(f"{cache_dir}/all_positive_paths.pkl", "rb"))
    else:
        pos_relations = list(
            get_all_positive_relation_paths_fl(
                specs,
                relation_set,
                hops=hops,
                dataset=dataset
            )
        )
        pkl.dump(pos_relations, open(f"{cache_dir}/all_positive_paths.pkl", "wb"))
    return pos_relations


def collect_program_execution_fl(
    programs, specs: SpecType, dataset: List[nx.MultiDiGraph]
):
    tt, ft, tf = defaultdict(set), defaultdict(set), defaultdict(set)
    # TT:  True x True
    # FT: Predicted False, but was supposed to be True
    # TF: Predicted True, but was supposed to be False
    bar = tqdm.tqdm(specs)
    bar.set_description("Getting Program Output")
    all_out_mappings, all_nodes = defaultdict(set), defaultdict(set)
    for i, faulty_nodes in bar:
        faulty_nodes = set(faulty_nodes)
        bar.set_description(f"Getting Program Output {i}")
        nx_g = dataset[i]
        programs = programs
        out_mappingss = batch_find_program_executor(nx_g, programs)
        word_mappingss = list([list([om[0] for om in oms]) for oms in out_mappingss])
        assert len(out_mappingss) == len(programs), len(out_mappingss)
        assert len(word_mappingss) == len(programs), len(word_mappingss)
        for p, oms in zip(programs, out_mappingss):
            wret = p.return_variables[0]
            for mapping in oms:
                all_out_mappings[p].add((i, mapping2tuple(mapping)))
                all_nodes[p].add((i, mapping[0][wret]))
        for p in programs:
            ft[p] |= set([(i, n) for n in faulty_nodes if n not in all_nodes[p]])
            tt[p] |= set([(i, n) for n in faulty_nodes if n in all_nodes[p]])
            tf[p] |= set([(i, n) for n in all_nodes[p] if n not in faulty_nodes])

    print("Total tt: ", sum([len(tt[p]) for p in tt]))
    return tt, ft, tf, all_out_mappings


def build_version_space(
    programs, specs, data_sample_set_relation_cache, logger, cache_dir: str
):
    assert cache_dir is not None, "Cache dir must be specified"
    if pexists("{cache_dir}/stage2_{TASK}.pkl"):
        with open(f"{cache_dir}/stage2_{TASK}.pkl", "rb") as f:
            tt, tf, ft, io_to_program, all_out_mappings = pkl.load(f)
            print(len(tt), len(tf), len(ft))
    else:
        bar = tqdm.tqdm(specs)
        start_time = time.time()
        bar.set_description("Stage 2 - Getting Program Output")
        tt, tf, ft, all_out_mappings = collect_program_execution_fl(
            programs, specs, data_sample_set_relation_cache
        )
        end_time = time.time()
        print("Time to collect program execution: ", end_time - start_time)
        logger.log("Collect program execution", float(end_time - start_time))
        print(len(programs), len(tt))
        io_to_program = defaultdict(list)
        report_metrics(programs, tt, tf, ft, io_to_program)
        with open(f"{cache_dir}/stage2_{TASK}.pkl", "wb") as f:
            pkl.dump([tt, tf, ft, io_to_program, all_out_mappings], f)
    return tt, tf, ft, io_to_program, all_out_mappings


def build_io_to_program(tt, tf, ft, all_out_mappings, programs, dataset, specs: SpecType):
    tt, tf, ft = [defaultdict(set) for _ in range(3)]
    io_to_program = defaultdict(list)
    all_nodes = defaultdict(set)
    for p in programs:
        wret = p.return_variables[0]
        for i, (w_bind, r_bind) in sorted(list(all_out_mappings[p])):
            wbind, r_bind = tuple2mapping((w_bind, r_bind))
            faulty_nodes = set(specs[i][1])
            if w_bind[wret] in faulty_nodes:
                tt[p].add((i, w_bind[wret]))
            else:
                tf[p].add((i, w_bind[wret]))
        # Calculate ft
        for i, faulty_nodes in specs:
            ft[p] |= set([(i, n) for n in faulty_nodes if (i, n) not in tt[p]])
        io_to_program[tuple(tt[p]), tuple(tf[p]), tuple(ft[p])].append(p)
    for p in programs:
        wret = p.return_variables[0]
        for i, (w_bind, r_bind) in sorted(list(all_out_mappings[p])):
            w_bind, r_bind = tuple2mapping((w_bind, r_bind))
            all_nodes[p].add((i, w_bind[wret]))
        assert all_nodes[p] == (tt[p] | tf[p])

    return io_to_program


def precision_counter_version_space_based_entity_linking(
    pos_paths, dataset, specs, data_sample_set_relation_cache, cache_dir
):
    assert cache_dir is not None, "Cache dir must be specified"
    # STAGE 1: Build base relation spaces
    programs = construct_or_get_initial_programs(
        pos_paths, f"{cache_dir}/stage1_{TASK}.pkl", logger
    )
    print("Number of programs in stage 1: ", len(programs))
    # STAGE 2: Build version space
    tt, tf, ft, io_to_program, all_out_mappings = build_version_space(
        programs, specs, data_sample_set_relation_cache, logger, cache_dir
    )
    io_to_program = build_io_to_program(
        tt, tf, ft, all_out_mappings, programs, dataset, specs
    )

    # STAGE 3: Build version space
    vss = []
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        if tt_p:
            vss.append(VS(set(tt_p), set(tf_p), set(ft_p), ps, all_out_mappings[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10
    pps, io2pps, pcps, cov_tt, cov_tt_perfect = [], {}, [], set(), set()
    cov_tt_counter = set()
    seen_p = set()
    start_time = time.time()
    for it in range(max_its):
        if pexists(f"{cache_dir}/stage3_{it}_{TASK}.pkl"):
            vss, c2vs = pkl.load(open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "rb"))
        else:
            c2vs = construct_constraints_to_valid_version_spaces(vss)
            # Save this for this iter
            with open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "wb") as f:
                pkl.dump([vss, c2vs], f)
        if pexists(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl"):
            with open(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl", "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss, new_io_to_vs = [], {}
            has_child = [False] * len(vss)
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description("Stage 3 - Creating New Version Spaces")
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0
                for vs_idx in vs_idxs:
                    cnt += 1
                    new_program = add_constraint_to_find_program(
                        vss[vs_idx].programs[0], c
                    )
                    if new_program in seen_p:
                        continue
                    seen_p.add(new_program)
                    big_bar.set_postfix(
                        {
                            "cnt": cnt,
                            "cov_tt": len(cov_tt),
                            "cov_tt_perfect": len(cov_tt_perfect),
                            "cov_tt_counter": len(cov_tt_counter),
                        }
                    )
                    vs = vss[vs_idx]
                    vs_matches = get_intersect_constraint_vs(
                        c, vs, data_sample_set_relation_cache, cache
                    )
                    if not vs_matches:
                        continue
                    ios = mappings2linking_tuples(vss[vs_idx].programs[0], vs_matches)
                    # Now check the tt, tf, ft
                    new_tt, new_tf = (ios & vss[vs_idx].tt), (ios & vss[vs_idx].tf)
                    new_ft = vss[vs_idx].ft
                    io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                    if not new_tt and new_tf - cov_tt_counter:
                        print(f"Found new counter program")
                        pcps.append(new_program)
                        io2pps[io_key] = VS(
                            new_tt, new_tf, new_ft, [new_program], vs_matches
                        )
                        cov_tt_counter |= new_tf
                        continue

                    if not new_tt:
                        continue
                    old_p, _, _ = get_p_r_f1(
                        vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft
                    )
                    new_p, _, _ = get_p_r_f1(new_tt, new_tf, new_ft)
                    if io_key in new_io_to_vs:
                        continue
                    if check_add_perfect_program(
                        new_tt,
                        new_tf,
                        new_ft,
                        cov_tt_perfect,
                        io_key,
                        new_program,
                        vs_matches,
                        io2pps,
                        pps,
                        cache_dir,
                        it,
                        logger,
                        TASK,
                        start_time,
                    ):
                        continue
                    if new_p > old_p:
                        if not (new_tt - cov_tt):
                            continue
                        cov_tt |= new_tt
                        print(f"Found new increased precision: {old_p} -> {new_p}")
                        acc += 1
                    has_child[vs_idx] = True
                    if io_key not in new_io_to_vs:
                        new_vs = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        new_vss.append(new_vs)
                        new_io_to_vs[io_key] = new_vs
                    else:
                        new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", c)

            print("Number of perfect programs:", len(pps))
            with open(pjoin(cache_dir, f"stage3_{it}_pps_{TASK}.pkl"), "wb") as f:
                pkl.dump(pps, f)

            # Adding dependent programs and counter dependent program
            extra_pps, extra_cov_tt = join_counter_vss(
                pps, pcps, cov_tt_perfect, new_vss, cov_tt_counter
            )
            pps += extra_pps
            cov_tt_perfect |= extra_cov_tt
            extra_pps, extra_cov_tt = join_inteserct_vss(
                    new_vss, cov_tt_perfect)
            pps += extra_pps
            cov_tt_perfect |= extra_cov_tt
            print(
                "Number of perfect program after exclusion refinement:",
                len(pps),
                len(cov_tt_perfect),
            )
            with open(pjoin(cache_dir, f"stage3_{it}_pps_{TASK}.pkl"), "wb") as f:
                pkl.dump(pps, f)
            nvss = len(new_vss)

            new_vss = [vs for vs in new_vss if vs.tt - cov_tt_perfect]
            nvss_after = len(new_vss)
            if nvss_after > 10000:  # Too heavy, cannot run
                break
            print(f"Number of new version spaces after pruning: {nvss} -> {nvss_after}")

            if pexists(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl")):
                with open(
                    pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "rb"
                ) as f:
                    new_vss = pkl.load(f)
            else:
                with open(
                    pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "wb"
                ) as f:
                    pkl.dump(new_vss, f)

        vss = new_vss

    return programs


def precision_version_space_based_entity_linking(
    pos_paths, dataset, specs, data_sample_set_relation_cache, cache_dir
):
    assert cache_dir is not None, "Cache dir must be specified"
    # STAGE 1: Build base relation spaces
    programs = construct_or_get_initial_programs(
        pos_paths, f"{cache_dir}/stage1_linking.pkl", logger
    )
    print("Number of programs in stage 1: ", len(programs))
    # STAGE 2: Build version space
    tt, tf, ft, io_to_program, all_out_mappings = build_version_space(
        programs, specs, data_sample_set_relation_cache, logger, cache_dir
    )
    io_to_program = build_io_to_program(tt, tf, ft, all_out_mappings, programs, dataset)

    # STAGE 3: Build version space
    vss = []
    for (tt_p, tf_p, ft_p), ps in io_to_program.items():
        if tt_p:
            vss.append(VS(set(tt_p), set(tf_p), set(ft_p), ps, all_out_mappings[ps[0]]))

    print("Number of version spaces: ", len(vss))
    max_its = 10

    pps, io2pps, pcps, cov_tt, cov_tt_perfect = [], {}, [], set(), set()
    cov_tt_counter = set()
    start_time = time.time()
    for it in range(max_its):
        if pexists(f"{cache_dir}/stage3_{it}_{TASK}.pkl"):
            vss, c2vs = pkl.load(open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "rb"))
        else:
            c2vs = construct_constraints_to_valid_version_spaces(vss)
            # Save this for this iter
            with open(f"{cache_dir}/stage3_{it}_{TASK}.pkl", "wb") as f:
                pkl.dump([vss, c2vs], f)
        if pexists(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl"):
            with open(f"{cache_dir}/stage3_{it}_new_vs_{TASK}.pkl", "rb") as f:
                new_vss = pkl.load(f)
        else:
            new_vss, new_io_to_vs = [], {}
            has_child = [False] * len(vss)
            big_bar = tqdm.tqdm(c2vs.items())
            big_bar.set_description("Stage 3 - Creating New Version Spaces")
            for c, vs_idxs in big_bar:
                # Cache to save computation cycles
                cache, cnt, acc = {}, 0, 0
                for vs_idx in vs_idxs:
                    cnt += 1
                    big_bar.set_postfix(
                        {
                            "cnt": cnt,
                            "cov_tt": len(cov_tt),
                            "cov_tt_perfect": len(cov_tt_perfect),
                            "cov_tt_counter": len(cov_tt_counter),
                        }
                    )
                    vs = vss[vs_idx]
                    vs_matches = get_intersect_constraint_vs(
                        c, vs, data_sample_set_relation_cache, cache
                    )
                    if not vs_matches:
                        continue
                    ios = mappings2linking_tuples(vss[vs_idx].programs[0], vs_matches)
                    # Now check the tt, tf, ft
                    new_tt, new_tf = (ios & vss[vs_idx].tt), (ios & vss[vs_idx].tf)
                    new_ft = vss[vs_idx].ft
                    io_key = tuple((tuple(new_tt), tuple(new_tf), tuple(new_ft)))
                    new_program = add_constraint_to_find_program(
                        vss[vs_idx].programs[0], c
                    )
                    if not new_tt and new_tf - cov_tt_counter:
                        print(f"Found new counter program")
                        pcps.append(new_program)
                        io2pps[io_key] = VS(
                            new_tt, new_tf, new_ft, [new_program], vs_matches
                        )
                        cov_tt_counter |= new_tf
                        continue

                    if not new_tt:
                        continue
                    old_p, _, _ = get_p_r_f1(
                        vss[vs_idx].tt, vss[vs_idx].tf, vss[vs_idx].ft
                    )
                    new_p, _, _ = get_p_r_f1(new_tt, new_tf, new_ft)
                    if io_key in new_io_to_vs:
                        continue
                    if check_add_perfect_program(
                        new_tt,
                        new_tf,
                        new_ft,
                        cov_tt_perfect,
                        io_key,
                        new_program,
                        vs_matches,
                        io2pps,
                        pps,
                        cache_dir,
                        it,
                        logger,
                        TASK,
                        start_time,
                    ):
                        continue
                    if new_p > old_p:
                        if not (new_tt - cov_tt):
                            continue
                        cov_tt |= new_tt
                        print(f"Found new increased precision: {old_p} -> {new_p}")
                        acc += 1
                    has_child[vs_idx] = True
                    if io_key not in new_io_to_vs:
                        new_vs = VS(new_tt, new_tf, new_ft, [new_program], vs_matches)
                        new_vss.append(new_vs)
                        new_io_to_vs[io_key] = new_vs
                    else:
                        new_io_to_vs[io_key].programs.append(new_program)
                if not acc:
                    print("Rejecting: ", c)

            print("Number of perfect programs:", len(pps))
            with open(pjoin(cache_dir, f"stage3_{it}_pps_{TASK}.pkl"), "wb") as f:
                pkl.dump(pps, f)

            nvss = len(new_vss)
            new_vss = [vs for vs in new_vss if vs.tt - cov_tt_perfect]
            nvss_after = len(new_vss)
            if nvss_after > 10000:  # Too heavy, cannot run
                break
            print(f"Number of new version spaces after pruning: {nvss} -> {nvss_after}")

            if pexists(pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl")):
                with open(
                    pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "rb"
                ) as f:
                    new_vss = pkl.load(f)
            else:
                with open(
                    pjoin(cache_dir, f"stage3_{it}_new_vs_{TASK}.pkl"), "wb"
                ) as f:
                    pkl.dump(new_vss, f)

        vss = new_vss

    return programs


def dump_config(args):
    with open(f"{args.cache_dir}/config_linking.json", "w") as f:
        json.dump(args.__dict__, f)


if __name__ == "__main__":
    relation_set = dummy_calculate_relation_set(None, None, None)
    args = get_args()
    args.cache_dir = setup_cache_dir(args, "entity_linking")
    dump_config(args)
    logger.set_fp(f"{args.cache_dir}/log.json")
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz", exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz_no_rel", exist_ok=True)
    os.makedirs(f"{args.cache_dir}/viz_entity_mapping", exist_ok=True)
    args = setup_grammar(args)
    start_time = time.time()
    dataset = setup_dataset(args)
    specs, entity_dataset = setup_specs(args, dataset, "linking")
    end_time = time.time()
    print(f"Time taken to load dataset and construct specs: {end_time - start_time}")
    logger.log("construct spec time: ", float(end_time - start_time))

    start_time = time.time()
    if pexists(f"{args.cache_dir}/ds_cache_linking_kv.pkl"):
        with open(f"{args.cache_dir}/ds_cache_linking_kv.pkl", "rb") as f:
            data_sample_set_relation_cache = pkl.load(f)
    else:
        data_sample_set_relation_cache = []
        bar = tqdm.tqdm(total=len(dataset))
        bar.set_description("Constructing data sample set relation cache")
        for i, data in enumerate(entity_dataset):
            if args.use_layoutlm_output and "legacy" in args.rel_type:
                nx_g = args.build_nx_g(dataset[i], data)
            else:
                nx_g = args.build_nx_g(data)
            data_sample_set_relation_cache.append(nx_g)
            img = viz_data(data, nx_g)
            img_no_rel = viz_data_no_rel(data)
            img_ent_map = viz_data_entity_mapping(data)
            cv2.imwrite(f"{args.cache_dir}/viz/{i}.png", img)
            cv2.imwrite(f"{args.cache_dir}/viz_no_rel/{i}.png", img_no_rel)
            cv2.imwrite(f"{args.cache_dir}/viz_entity_mapping/{i}.png", img_ent_map)
            bar.update(1)

        end_time = time.time()
        print(
            f"Time taken to construct data sample set relation cache: {end_time - start_time}"
        )
        logger.log(
            "construct data sample set relation cache time: ",
            float(end_time - start_time),
        )
        with open(f"{args.cache_dir}/ds_cache_linking_kv.pkl", "wb") as f:
            pkl.dump(data_sample_set_relation_cache, f)

    if args.use_sem:
        assert args.model in ["layoutlmv3"]
        if args.model == "layoutlmv3":
            if pexists(f"{args.cache_dir}/embs_layoutlmv3.pkl"):
                with open(f"{args.cache_dir}/embs_layoutlmv3.pkl", "rb") as f:
                    all_embs = pkl.load(f)
            else:
                from models.layout_lmv3_utils import get_word_embedding

                start_time = time.time()
                all_embs = []
                for data in entity_dataset:
                    all_embs.append(get_word_embedding(data))
                end_time = time.time()
                print(f"Time taken to get word embedding: {end_time - start_time}")
                logger.log("get word embedding time: ", float(end_time - start_time))
            for i, nx_g in enumerate(data_sample_set_relation_cache):
                for w in sorted(nx_g.nodes()):
                    nx_g.nodes[w]["emb"] = all_embs[i][w]
    # Now we have the data sample set relation cache
    print("Stage 1 - Constructing Program Space")
    if pexists(f"{args.cache_dir}/pos_paths_linking_kv.pkl"):
        with open(f"{args.cache_dir}/pos_paths_linking_kv.pkl", "rb") as f:
            pos_paths = pkl.load(f)
    else:
        start_time = time.time()
        pos_paths = get_path_specs_linking(
            entity_dataset,
            specs,
            relation_set=args.relation_set,
            data_sample_set_relation_cache=data_sample_set_relation_cache,
            cache_dir=args.cache_dir,
            hops=args.hops,
        )
        end_time = time.time()
        print(f"Time taken to construct positive paths: {end_time - start_time}")
        logger.log("construct positive paths time: ", float(end_time - start_time))
        with open(f"{args.cache_dir}/pos_paths_linking_kv.pkl", "wb") as f:
            pkl.dump(pos_paths, f)

    if args.strategy == "precision":
        programs = precision_version_space_based_entity_linking(
            pos_paths,
            entity_dataset,
            specs,
            data_sample_set_relation_cache,
            args.cache_dir,
        )
    elif args.strategy == "precision_counter":
        programs = precision_counter_version_space_based_entity_linking(
            pos_paths,
            entity_dataset,
            specs,
            data_sample_set_relation_cache,
            args.cache_dir,
        )

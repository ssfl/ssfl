import os
import json
import networkx as nx
from graph_algos.nx_shortcuts import neighbors_in, neighbors_out

NESTABLE_STMTS_JAVA = [
    "Block",
    "SwitchStatement",
    "IfStatement",
    "ForStatement",
    "WhileStatement",
]

STMTS_CPP_SRCML = [
    "for",
    "while",
    "switch",
    "decl_stmt",
    "if_stmt",
    "case",
    "break",
    "do",
    "continue",
    "goto",
    "empty_stmt",
    "expr_stmt",
    "default",
    "label",
    "continue",
    "return",
    "placeholder_stmt",
]


class GumtreeWrapper:
    ast_cpp_command = "java -jar ./jars/ast_extractor_cpp.jar "
    ast_java_command = "java -jar ./jars/ast_extractor_java.jar "

    def get_tree_diff(path1: str, path2: str):
        # Note: Please change to JAVA 11 before running this
        if path1.endswith("java"):
            cmd = GumtreeWrapper.ast_java_command + path1 + " " + path2
        elif path1.endswith("c") or path1.endswith("cpp") or path1.endswith("cxx"):
            cmd = GumtreeWrapper.ast_cpp_command + path1 + " " + path2
        json_str = "\n".join(os.popen(cmd).read().split("\n")[1:])  # ignore 1st l
        map_dict = json.loads(json_str)
        map_dict["mapping"] = {int(k): int(v) for k, v in map_dict["mapping"].items()}
        return map_dict


class GumtreeASTUtils:
    """AST Shortcuts based on Gumtree Format: JavaParser for Java,
    SrcML for Cpp"""

    def check_is_stmt_java(ntype):
        return "Statement" in ntype

    def check_is_stmt_cpp(ntype):
        return ntype in STMTS_CPP_SRCML

    def get_prev_sibs(u, q: nx.MultiDiGraph):
        parents = neighbors_in(u, q)
        if len(parents) > 0:
            p = parents[0]
            return neighbors_out(p, q, lambda _, x, k, e: x < u)
        return []

    def get_next_sibs(u, q: nx.MultiDiGraph):
        parents = neighbors_in(u, q)
        if len(parents) > 0:
            p = parents[0]
            return neighbors_out(p, q, lambda _, x, k, e: x > u)
        return []

    def build_nx_graph(ndicts):
        nx_ast = nx.MultiDiGraph()
        for ndict in sorted(ndicts, key=lambda x: x["id"]):
            nx_ast.add_node(
                ndict["id"],
                ntype=ndict["type"],
                token=ndict["label"],
                graph="ast",
                start_line=ndict["range"]["begin"]["line"],
                end_line=ndict["range"]["end"]["line"],
                status=0,
            )

            # Hypo 1: NX always keep the order of edges between one nodes and all
            # others, so we can recover sibling from this
            if ndict["parent_id"] != -1:
                nx_ast.add_edge(ndict["parent_id"], ndict["id"], label="parent_child")
        return nx_ast

    def has_stmt_child(u, q, check_func):
        for n in nx.descendants(q, u):
            if check_func(q.nodes[n]["ntype"]):
                return True
        return False


class GumtreeBasedAnnotation:
    """Gumtree-based annotation between differencing ASTs"""

    def add_placeholder_stmts_java(nx_ast):
        """add PlaceholderStatement for each block"""
        queue = [0]
        while len(queue) > 0:
            node = queue.pop(0)
            if nx_ast.nodes[node]["ntype"] in NESTABLE_STMTS_JAVA:
                child_stmts = neighbors_out(node, nx_ast)
                new_node = max(nx_ast.nodes()) + 1
                if len(child_stmts) > 0:
                    end_line = max([nx_ast.nodes[c]["end_line"] for c in child_stmts])
                else:
                    end_line = nx_ast.nodes[node]["start_line"]
                nx_ast.add_node(
                    new_node,
                    ntype="PlaceHolderStatement",
                    token="",
                    graph="ast",
                    start_line=end_line,
                    end_line=end_line,
                    status=0,
                )
                nx_ast.add_edge(node, new_node, label="parent_child")
                child_stmts = child_stmts + [len(nx_ast.nodes()) - 1]
                queue.extend(child_stmts)
            else:
                queue.extend(neighbors_out(node, nx_ast))
        return nx_ast

    def add_placeholder_stmts_cpp(nx_ast):
        """add PlaceholderStatement for each block"""
        queue = [0]
        while len(queue) > 0:
            node = queue.pop(0)
            child_stmts = neighbors_out(
                node,
                nx_ast,
                lambda u, v, k, e: GumtreeASTUtils.check_is_stmt_cpp(
                    nx_ast.nodes[v]["ntype"]
                ),
            )
            if len(child_stmts) > 0 or nx_ast.nodes[node]["ntype"] in [
                "block_content",
                "case",
            ]:
                new_node = max(nx_ast.nodes()) + 1
                if len(child_stmts) > 0:
                    end_line = max([nx_ast.nodes[c]["end_line"] for c in child_stmts])
                else:
                    end_line = nx_ast.nodes[node]["start_line"]
                nx_ast.add_node(
                    new_node,
                    ntype="placeholder_stmt",
                    token="",
                    graph="ast",
                    start_line=end_line,
                    end_line=end_line,
                    status=0,
                )
                nx_ast.add_edge(node, new_node, label="parent_child")
                child_stmts = child_stmts + [len(nx_ast.nodes()) - 1]
            queue.extend(neighbors_out(node, nx_ast))
        return nx_ast

    def check_statement_elem_removed(
        n, nx_ast, ldels, check_func=GumtreeASTUtils.check_is_stmt_java
    ):
        queue = [n]
        started = False
        while len(queue) > 0:
            n = queue.pop(0)
            if check_func(nx_ast.nodes[n]["ntype"]) and started:
                continue
            if n in ldels:
                return True
            queue.extend(neighbors_out(n, nx_ast))
            started = True
        return False

    def check_statement_elem_inserted(
        n, nx_ast, lisrts, check_func=GumtreeASTUtils.check_is_stmt_java
    ):
        queue = [n]
        started = False
        while len(queue) > 0:
            n = queue.pop(0)
            if check_func(nx_ast.nodes[n]["ntype"]) and started:
                continue
            if n in lisrts:
                return True
            queue.extend(neighbors_out(n, nx_ast))
            started = True
        return False

    def get_non_inserted_ancestor(rev_map_dict, dst_n, nx_ast_dst):
        parents = neighbors_in(dst_n, nx_ast_dst)
        while len(parents) > 0:
            parent = parents[0]
            if parent not in rev_map_dict:
                parents = neighbors_in(parent, nx_ast_dst)
            else:
                return parent

    def find_modified_statement(
        nx_ast_src, ldels, check_func=GumtreeASTUtils.check_is_stmt_java
    ):
        ns = [n for n in nx_ast_src.nodes() if check_func(nx_ast_src.nodes[n]["ntype"])]
        ns = [
            n
            for n in ns
            if GumtreeBasedAnnotation.check_statement_elem_removed(
                n, nx_ast_src, ldels, check_func
            )
        ]
        return ns

    def find_inserted_statement(
        nx_ast_src,
        nx_ast_dst,
        rev_map_dict,
        lisrts,
        check_func=GumtreeASTUtils.check_is_stmt_java,
    ):
        ns = [n for n in nx_ast_dst.nodes() if check_func(nx_ast_dst.nodes[n]["ntype"])]
        inserted_stmts = []
        # First, check if the statement itself is inserted
        ns_i = [n for n in ns if n in lisrts]
        for s_i in ns_i:
            if neighbors_in(s_i, nx_ast_dst)[0] in lisrts:
                continue
            dst_p = neighbors_in(s_i, nx_ast_dst)[0]
            dst_prev_sibs = GumtreeASTUtils.get_prev_sibs(s_i, nx_ast_dst)
            dst_prev_sibs = [n for n in dst_prev_sibs if n not in lisrts]
            src_p = rev_map_dict[dst_p]
            # print("node: ", src_p)
            # for n in neighbors_out(src_p, nx_ast_src):
            #    print(nx_ast_src.nodes[n])
            if len(dst_prev_sibs) > 0:
                src_prev_sib = rev_map_dict[max(dst_prev_sibs)]
                try:
                    # print(GumtreeASTUtils.get_next_sibs(src_prev_sib, nx_ast_src))
                    src_next_sib = min(
                        GumtreeASTUtils.get_next_sibs(src_prev_sib, nx_ast_src)
                    )
                except:
                    print(nx_ast_dst.nodes[dst_p]["ntype"])
            else:
                # get the first child in the block
                src_next_sib = min(neighbors_out(rev_map_dict[dst_p], nx_ast_src))
            inserted_stmts.append(src_next_sib)

        ns_ni = [n for n in ns if n not in lisrts]
        for s_n in ns_ni:
            if GumtreeBasedAnnotation.check_statement_elem_inserted(
                s_n, nx_ast_dst, lisrts, check_func
            ):
                inserted_stmts.append(rev_map_dict[s_n])
        return inserted_stmts

    def build_nx_graph_stmt_annt(map_dict, lang="java"):
        """Statement-level annotation"""
        # Insert a place holder node for each
        if lang == "cpp":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_cpp
            check_func = GumtreeASTUtils.check_is_stmt_cpp
        elif lang == "java":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_java
            check_func = GumtreeASTUtils.check_is_stmt_java

        nx_ast_src = GumtreeASTUtils.build_nx_graph(map_dict["srcNodes"])
        nx_ast_dst = GumtreeASTUtils.build_nx_graph(map_dict["dstNodes"])

        nx_ast_src = add_placeholder_func(nx_ast_src)
        # Node post processing, recheck if token equal
        for n_s in nx_ast_src.nodes():
            if n_s in map_dict["mapping"]:
                n_d = map_dict["mapping"][n_s]
                if nx_ast_src.nodes[n_s]["token"] != nx_ast_dst.nodes[n_d]["token"]:
                    del map_dict["mapping"][n_s]
                    map_dict["deleted"].append(n_s)
                    map_dict["inserted"].append(n_d)

        rev_map_dict = {v: k for k, v in map_dict["mapping"].items()}
        linvalids = [
            n
            for n in nx_ast_dst.nodes()
            if n not in map_dict["inserted"] and n not in rev_map_dict
        ]
        map_dict["inserted"].extend(linvalids)

        # moved nodes: check parent
        for nsid, ndid in map_dict["mapping"].items():
            psids = neighbors_in(nsid, nx_ast_src)
            pdids = neighbors_in(ndid, nx_ast_dst)
            if len(psids) > 0 and len(pdids) > 0:
                psid = psids[0]
                pdid = pdids[0]
                if psid in map_dict["mapping"] and pdid in rev_map_dict:
                    if map_dict["mapping"][psid] != pdid:  # Moved
                        map_dict["deleted"].append(nsid)
                        map_dict["inserted"].append(ndid)

        # modified nodes
        for st_n in GumtreeBasedAnnotation.find_modified_statement(
            nx_ast_src, map_dict["deleted"], check_func=check_func
        ):
            nx_ast_src.nodes[st_n]["status"] = 1

        # inserted nodes: check sibling
        for st_n in GumtreeBasedAnnotation.find_inserted_statement(
            nx_ast_src,
            nx_ast_dst,
            rev_map_dict,
            map_dict["inserted"],
            check_func=check_func,
        ):
            nx_ast_src.nodes[st_n]["status"] = 1

        # remove nodes
        remove_subgraphs = []
        for st_n in nx_ast_src.nodes():
            if check_func(nx_ast_src.nodes[st_n]["ntype"]):
                # nx_ast_src.nodes[st_n]['status'] = 4
                continue

            if GumtreeASTUtils.has_stmt_child(st_n, nx_ast_src, check_func):
                # print(st_n)
                continue
            rs = nx.descendants(nx_ast_src, st_n) | {st_n}
            if len(list(rs)):
                remove_subgraphs.append(rs)

        for rs in remove_subgraphs:
            nx_ast_src.remove_nodes_from(rs)

        return nx_ast_src, nx_ast_dst

    def build_nx_graph_node_annt(map_dict, lang="java"):
        nx_ast_src = GumtreeASTUtils.build_nx_graph(map_dict["srcNodes"])
        nx_ast_dst = GumtreeASTUtils.build_nx_graph(map_dict["dstNodes"])

        if lang == "java":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_java
        elif lang == "cpp":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_cpp

        nx_ast_src = add_placeholder_func(nx_ast_src)

        # Node post processing, recheck if token equal
        for n_s in nx_ast_src.nodes():
            if n_s in map_dict["mapping"]:
                n_d = map_dict["mapping"][n_s]
                if nx_ast_src.nodes[n_s]["token"] != nx_ast_dst.nodes[n_d]["token"]:
                    del map_dict["mapping"][n_s]
                    map_dict["deleted"].append(n_s)
                    map_dict["inserted"].append(n_d)

        # get nx graph of two asts
        # deleted nodes
        for nid in map_dict["deleted"]:
            nx_ast_src.nodes[nid]["status"] = 1

        # inserted nodes: check sibling
        rev_map_dict = {v: k for k, v in map_dict["mapping"].items()}
        for nid in map_dict["inserted"]:
            nx_ast_dst.nodes[nid]["status"] = 2
            dst_p = GumtreeBasedAnnotation.get_non_inserted_ancestor(
                rev_map_dict, nid, nx_ast_dst
            )
            if dst_p is None:
                continue
            nx_ast_src.nodes[rev_map_dict[dst_p]]["status"] = 2
            # sibling
            dst_prev_sibs = GumtreeASTUtils.get_prev_sibs(nid, nx_ast_dst)
            dst_prev_sibs = [n for n in dst_prev_sibs if n in rev_map_dict]
            if len(dst_prev_sibs) > 0:
                # Consider letting it to be 3
                nx_ast_src.nodes[rev_map_dict[max(dst_prev_sibs)]]["status"] = 1
                nx_ast_dst.nodes[max(dst_prev_sibs)]["status"] = 1
            else:
                src_p = rev_map_dict[dst_p]
                childs = neighbors_out(src_p, nx_ast_src)
                if len(childs) > 0:
                    prev_sib = min(childs)
                    nx_ast_src.nodes[prev_sib]["status"] = 1

        # moved nodes: check parent
        for nsid, ndid in map_dict["mapping"].items():
            psids = neighbors_in(nsid, nx_ast_src)
            pdids = neighbors_in(ndid, nx_ast_dst)
            if len(psids) > 0 and len(pdids) > 0:
                psid = psids[0]
                pdid = pdids[0]
                if psid in map_dict["mapping"] and pdid in rev_map_dict:
                    if map_dict["mapping"][psid] != pdid:
                        # This node has been removed in the original tree
                        nx_ast_src.nodes[nsid]["status"] = 1
                        # and inserted in the new tree
                        nx_ast_dst.nodes[ndid]["status"] = 2

                        # reverse parent finding
                        to = rev_map_dict[pdid]
                        # Original destination node status is inserted
                        nx_ast_src.nodes[to]["status"] = 2
                        # Get the sibling before it
                        dst_prev_sibs = GumtreeASTUtils.get_prev_sibs(ndid, nx_ast_dst)
                        dst_prev_sibs = [n for n in dst_prev_sibs if n in rev_map_dict]
                        if len(dst_prev_sibs) > 0:
                            # consider letting it be 3
                            nx_ast_src.nodes[rev_map_dict[max(dst_prev_sibs)]][
                                "status"
                            ] = 1
                            nx_ast_dst.nodes[max(dst_prev_sibs)]["status"] = 1
                        else:
                            src_p = rev_map_dict[pdid]
                            childs = neighbors_out(src_p, nx_ast_src)
                            if len(childs) > 0:
                                prev_sib = min(childs)
                                nx_ast_src.nodes[prev_sib]["status"] = 1

        return nx_ast_src, nx_ast_dst

    def get_coverage_graph_ast(nx_ast_g, cov_maps, verdicts):
        """Get coverage graph ast
        Parameters
        ----------
        cov_maps: each test case has coverage map: line -> freq,
        verdicts: each test case has 1 verdict: True/False
        Returns
        ----------
        nx_ast_g: nx_multi_di_graph
              Short description
        """
        nx_ast_g = nx_ast_g.copy()
        for i, (cov_map, verdict) in enumerate(zip(cov_maps, verdicts)):
            test_node = max(nx_ast_g.nodes()) + 1
            nx_ast_g.add_node(test_node, name=f"test_{i}", ntype="test", graph="test")
            link_type = "fail" if not verdict else "pass"
            for line in cov_map:
                if cov_map[line] <= 0:
                    continue
                for a_node in nx_ast_g.nodes:
                    if nx_ast_g.nodes[a_node]["graph"] != "ast":
                        continue
                    if nx_ast_g.nodes[a_node]["start_line"] == line:
                        queue = [a_node]
                        while len(queue) > 0:
                            node = queue.pop()
                            if (
                                len(
                                    neighbors_out(
                                        node,
                                        nx_ast_g,
                                        lambda u, v, k, e: v == test_node,
                                    )
                                )
                                > 0
                            ):
                                # Visited
                                continue
                            nx_ast_g.add_edge(
                                node, test_node, label=f"a_{link_type}_test"
                            )
                            queue.extend(
                                neighbors_out(
                                    node,
                                    nx_ast_g,
                                    lambda u, v, k, e: nx_ast_g.nodes[v]["graph"]
                                    == "ast",
                                )
                            )

        return nx_ast_g

    def build_nx_ast_cov_annt(
        src_b, src_f, cov_maps, verdicts, annot_func=build_nx_graph_node_annt
    ):
        map_dict = GumtreeWrapper.get_tree_diff(src_b, src_f)
        nx_ast_src, nx_ast_dst = annot_func(map_dict, lang="cpp")
        nx_ast_cov = GumtreeBasedAnnotation.get_coverage_graph_ast(
            nx_ast_src, cov_maps, verdicts
        )
        return nx_ast_cov

    def build_nx_ast_cov(src_b, cov_maps, verdicts, lang="cpp"):
        map_dict = GumtreeWrapper.get_tree_diff(src_b, src_b)
        nx_ast_src = GumtreeASTUtils.build_nx_graph(map_dict["srcNodes"])
        if lang == "java":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_java
        elif lang == "cpp":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_cpp
        nx_ast_src = add_placeholder_func(nx_ast_src)
        for node in nx_ast_src.nodes():
            nx_ast_src.nodes[node]["status"] = 0
        nx_ast_cov = GumtreeBasedAnnotation.get_coverage_graph_ast(
            nx_ast_src, cov_maps, verdicts
        )
        return nx_ast_src

    def build_mapping_stmt(map_dict, lang="java"):
        """Statement-level annotation"""
        # Insert a place holder node for each
        if lang == "cpp":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_cpp
            check_func = GumtreeASTUtils.check_is_stmt_cpp
        elif lang == "java":
            add_placeholder_func = GumtreeBasedAnnotation.add_placeholder_stmts_java
            check_func = GumtreeASTUtils.check_is_stmt_java

        nx_ast_src = GumtreeASTUtils.build_nx_graph(map_dict["srcNodes"])
        nx_ast_dst = GumtreeASTUtils.build_nx_graph(map_dict["dstNodes"])

        # nx_ast_src = add_placeholder_func(nx_ast_src)
        # Node post processing, recheck if token equal
        for n_s in nx_ast_src.nodes():
            if n_s in map_dict["mapping"]:
                n_d = map_dict["mapping"][n_s]
                if nx_ast_src.nodes[n_s]["token"] != nx_ast_dst.nodes[n_d]["token"]:
                    del map_dict["mapping"][n_s]
                    map_dict["deleted"].append(n_s)
                    map_dict["inserted"].append(n_d)

        rev_map_dict = {v: k for k, v in map_dict["mapping"].items()}
        linvalids = [
            n
            for n in nx_ast_dst.nodes()
            if n not in map_dict["inserted"] and n not in rev_map_dict
        ]
        map_dict["inserted"].extend(linvalids)

        # moved nodes: check parent
        for nsid, ndid in map_dict["mapping"].items():
            psids = neighbors_in(nsid, nx_ast_src)
            pdids = neighbors_in(ndid, nx_ast_dst)
            if len(psids) > 0 and len(pdids) > 0:
                psid = psids[0]
                pdid = pdids[0]
                if psid in map_dict["mapping"] and pdid in rev_map_dict:
                    if map_dict["mapping"][psid] != pdid:  # Moved
                        map_dict["deleted"].append(nsid)
                        map_dict["inserted"].append(ndid)

        for st_n in GumtreeBasedAnnotation.find_modified_statement(
            nx_ast_src, map_dict["deleted"], check_func=check_func
        ):
            nx_ast_src.nodes[st_n]["status"] = 1

        # inserted nodes: check sibling
        for st_n in GumtreeBasedAnnotation.find_inserted_statement(
            nx_ast_src,
            nx_ast_dst,
            rev_map_dict,
            map_dict["inserted"],
            check_func=check_func,
        ):
            nx_ast_src.nodes[st_n]["status"] = 1

        return nx_ast_src, nx_ast_dst

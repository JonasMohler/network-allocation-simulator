import os
from src.util.const import *

# File names
SP_FILE_NAME = f"shortest_paths.{FILE_TYPE[SHORTEST_PATH]}"
DIAMETER_FILE_NAME = f"diameter.{FILE_TYPE[DIAMETER]}"
DEGREE_FILE_NAME = f"degree.{FILE_TYPE[DEGREE]}"
REACH_FILE_NAME = "reach.json"
COVER_FILE_NAME = f"cover.{FILE_TYPE[COVER]}"
TM_FILE_NAME = f"allocation_matrix.{FILE_TYPE[ALLOCATION_MATRIX]}"
TM_COVER_FILE_NAME = "tm_cover.json"
PC_FILE_NAME = f"path_counts.{FILE_TYPE[PATH_COUNTS]}"
PL_FILE_NAME = f"path_lengths.{FILE_TYPE[PATH_LENGTHS]}"
TOPOLOGY_FILE_NAME = f"topology.{FILE_TYPE[TOPOLOGY]}"
C_IMPROVEMENT_NAME = f"c_imp.{FILE_TYPE[COVER_IMPROVEMENT]}"

# Allocations
ALLOC_FILE_NAME = f"allocations.{FILE_TYPE[ALLOCATION]}"
ALLOC_SUM_FILE_NAME = "alloc_sum.json"
TM_ALLOC_FILE_NAME = "alloc_traffic_mat.json"
TM_ALLOC_SUM_FILE_NAME = "tm_alloc_sum.json"


def sampled_alloc_file_name(ratio):
    return f"{ratio}_{ALLOC_FILE_NAME}"


def sampled_k_alloc_file_name(ratio, num_sp):
    return f"{ratio}_{num_sp}_{ALLOC_FILE_NAME}"


def k_alloc_file_name(num_sp):
    return f"{num_sp}_{ALLOC_FILE_NAME}"


def sampled_sp_file_name(ratio):
    return f"{ratio}_{SP_FILE_NAME}"


def sampled_ksp_file_name(ratio, k):
    return f"{ratio}_{k}_{SP_FILE_NAME}"


def ksp_file_name(k):
    return f"{k}_{SP_FILE_NAME}"


def sampled_cover_file_name(ratio, thresh):
    return f"{thresh}_{ratio}_{COVER_FILE_NAME}"


def cover_file_name(thresh):
    return f"{thresh}_{COVER_FILE_NAME}"


def cov_imp_file_name(thresh, num_sp):
    return f"{thresh}_{num_sp}_{C_IMPROVEMENT_NAME}"


def sampled_multi_cover_file_name(ratio, thresh, num_sp):
    return f"{thresh}_{ratio}_{num_sp}_{COVER_FILE_NAME}"


def multi_cover_file_name(thresh, num_sp):
    return f"{thresh}_{num_sp}_{COVER_FILE_NAME}"


def sampled_degree_file_name(ratio):
    return f"{ratio}_{DEGREE_FILE_NAME}"


def sampled_pc_file_name(ratio):
    return f"{ratio}_{PC_FILE_NAME}"


def sampled_k_pc_file_name(num_sp, ratio):
    return f"{ratio}_{num_sp}_{PC_FILE_NAME}"


def k_pc_file_name(num_sp):
    return f"{num_sp}_{PC_FILE_NAME}"


def get_topo_path(graph):
    return os.path.join(DATA_PATH, graph)


def get_strategies_path(graph):
    return os.path.join(get_topo_path(graph), 'strategies/')


def get_graph_path(graph):
    return os.path.join(get_topo_path(graph), 'graph/')


def get_graph_figure_path(graph):
    return os.path.join(os.path.join(FIGURE_PATH, 'graph/'), graph)


def get_general_figure_path():
    return os.path.join(FIGURE_PATH, 'general/')


def get_full_path(graph, data_type, strategy=None, ratio=None, thresh=None, fig_name=None, gen_fig=False, num_sp=None):

    strategies_path = get_strategies_path(graph)
    graph_path = get_graph_path(graph)

    full_path = None

    if data_type == ALLOCATION:

        if strategy is not None:
            if num_sp is not None:
                if ratio is not None:
                    name = sampled_k_alloc_file_name(ratio, num_sp)
                else:
                    name = k_alloc_file_name(num_sp)
            else:
                if ratio is not None:
                    name = sampled_alloc_file_name(ratio)
                else:
                    name = ALLOC_FILE_NAME
        else:
            raise ValueError(f"No strategy provided")

        full_path = os.path.join(f"{strategies_path}{strategy}/", name)

    if data_type == COVER:

        if strategy is not None:
            if num_sp is not None:
                if ratio is not None:
                    name = sampled_multi_cover_file_name(ratio, thresh, num_sp)
                else:
                    name = multi_cover_file_name(thresh, num_sp)
            else:
                if ratio is not None:
                    name = sampled_cover_file_name(ratio, thresh)
                else:
                    name = cover_file_name(thresh)

            full_path = os.path.join(f"{strategies_path}{strategy}", name)

    if data_type == DEGREE:

        if ratio is not None:
            name = sampled_degree_file_name(ratio)
        else:
            name = DEGREE_FILE_NAME

        full_path = os.path.join(graph_path, name)

    if data_type == SHORTEST_PATH:

        if num_sp is None:
            if ratio is not None:
                name = sampled_sp_file_name(ratio)
            else:
                name = SP_FILE_NAME
        else:
            if ratio is not None:
                name = sampled_ksp_file_name(num_sp, ratio)
            else:
                name = ksp_file_name(num_sp)

        full_path = os.path.join(graph_path, name)

    if data_type == ALLOCATION_MATRIX:
        name = TM_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == PATH_LENGTHS:
        name = PL_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == PATH_COUNTS:
        if num_sp is None:
            if ratio is not None:
                name = sampled_pc_file_name(ratio)
            else:
                name = PC_FILE_NAME
        else:
            if ratio is not None:
                name = sampled_k_pc_file_name(num_sp, ratio)
            else:
                name = k_pc_file_name(num_sp)

        full_path = os.path.join(graph_path, name)

    if data_type == TOPOLOGY:
        name = TOPOLOGY_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == DIAMETER:
        name = DIAMETER_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == FIGURE:
        if gen_fig:
            full_path = os.path.join(get_general_figure_path(), fig_name)
        else:
            full_path = os.path.join(get_graph_figure_path(graph, fig_name))

    if data_type == COVER_IMPROVEMENT:
        name = cov_imp_file_name(thresh, num_sp)
        full_path = os.path.join(os.path.join(strategies_path, strategy), name)

    return full_path


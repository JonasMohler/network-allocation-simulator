import os
from src.util.const import *

# File names
SP_FILE_NAME = "shortest_paths.json"
DIAMETER_FILE_NAME = "diameter.txt"
DEGREE_FILE_NAME = "degree.json"
REACH_FILE_NAME = "reach.json"
COVER_FILE_NAME = "cover.json"
TM_FILE_NAME = "allocation_matrix.pkl"
TM_COVER_FILE_NAME = "tm_cover.json"
PC_FILE_NAME = "path_counts.pkl"
PL_FILE_NAME = "path_lengths.csv"
TOPOLOGY_FILE_NAME = "topology.xml"

# Allocations
ALLOC_FILE_NAME = "allocations.json"
ALLOC_SUM_FILE_NAME = "alloc_sum.json"
TM_ALLOC_FILE_NAME = "alloc_traffic_mat.json"
TM_ALLOC_SUM_FILE_NAME = "tm_alloc_sum.json"


def sampled_alloc_file_name(ratio):
    return f"{ratio}_{ALLOC_FILE_NAME}"


def sampled_sp_file_name(ratio):
    return f"{ratio}_{SP_FILE_NAME}"


def sampled_cover_file_name(ratio, thresh):
    return f"{ratio}_{thresh}_{COVER_FILE_NAME}"


def cover_file_name(thresh):
    return f"{thresh}_{COVER_FILE_NAME}"


def sampled_degree_file_name(ratio):
    return f"{ratio}_{DEGREE_FILE_NAME}"


def sampled_pc_file_name(ratio):
    return f"{ratio}_{PC_FILE_NAME}"


def get_topo_path(graph):
    return os.path.join(DATA_PATH, graph)


def get_strategies_path(graph):
    return os.path.join(get_topo_path(graph), 'strategies/')


def get_graph_path(graph):
    return os.path.join(get_topo_path(graph), 'graph/')


def get_full_path(graph, data_type, strategy=None, ratio=None, thresh=None):
    topology_path = get_topo_path(graph)

    strategies_path = get_strategies_path(graph)
    graph_path = get_graph_path(graph)

    full_path = None

    if data_type == ALLOCATION:

        if strategy is not None:

            if ratio is not None:
                name = sampled_alloc_file_name(ratio)
            else:
                name = ALLOC_FILE_NAME
        else:
            raise ValueError(f"No strategy provided")

        full_path = os.path.join(f"{strategies_path}{strategy}/", name)

    if data_type == COVER:

        if strategy is not None:

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

        if ratio is not None:
            name = sampled_sp_file_name(ratio)
        else:
            name = SP_FILE_NAME

        full_path = os.path.join(graph_path, name)

    if data_type == ALLOCATION_MATRIX:
        name = TM_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == PATH_LENGTHS:
        name = PL_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == PATH_COUNTS:
        if ratio is not None:
            name = sampled_pc_file_name(ratio)
        else:
            name = PC_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == TOPOLOGY:
        name = TOPOLOGY_FILE_NAME
        full_path = os.path.join(graph_path, name)

    if data_type == DIAMETER:
        name = DIAMETER_FILE_NAME
        full_path = os.path.join(graph_path, name)

    return full_path


"""
Loaders for the intermediate data used by GMA computation algorithms.
"""
import json
import os
import pickle
import numpy as np

import fnss

from src.util.naming import *
from src.util.const import *


def _load(graph, data_type, strategy=None, ratio=None):

    full_path = get_full_path(graph, data_type, strategy, ratio)

    # Load file

    if FILE_TYPE[data_type] == "json":
        with open(full_path, "r") as infile:
            data = json.load(infile)

    elif FILE_TYPE[data_type] == "txt":
        with open(full_path, "r") as infile:
            data = infile.read()

    elif FILE_TYPE[data_type] == "pkl":
        with open(full_path, "rb") as infile:
            data = pickle.load(infile)

    elif FILE_TYPE[data_type] == "csv":
        data = np.genfromtxt(full_path, delimiter=',')

    elif FILE_TYPE[data_type] == "xml":
        data = fnss.read_topology(full_path)

    return data


def _store(data, graph, data_type, strategy=None, ratio=None):

    # Build parent directory and make if they don't yet exist and is allowed
    topology_path = get_topo_path(graph)

    if not os.path.exists(topology_path):
        if data_type == TOPOLOGY:
            os.mkdir(topology_path)
        else:
            raise ValueError(F"Graph not found: {topology_path}")

    strategy_path = get_strategies_path(graph)
    graph_path = get_graph_path(graph)

    if not os.path.exists(strategy_path):
        os.mkdir(strategy_path)

    if strategy is not None:
        concrete_strategy_path = os.path.join(strategy_path, strategy)
        if not os.path.exists(concrete_strategy_path):
            os.mkdir(concrete_strategy_path)

    if not os.path.exists(graph_path):
        os.mkdir(graph_path)

    # Get file name
    full_path = get_full_path(graph, data_type, strategy, ratio)

    # Store

    if FILE_TYPE[data_type] == "json":
        with open(full_path, "w+") as outfile:
            json.dump(data, outfile)

    elif FILE_TYPE[data_type] == "txt":
        with open(full_path, "w+") as outfile:
            outfile.write(str(data))

    elif FILE_TYPE[data_type] == "pkl":
        with open(full_path, "wb+") as outfile:
            pickle.dump(data, outfile)

    elif FILE_TYPE[data_type] == "csv":
        with open(full_path, "w+") as outfile:
            np.savetxt(outfile, data, delimiter=',')
    elif FILE_TYPE[data_type] == "xml":
        fnss.write_topology(data, full_path)


def get_graph(graph):
    data = _load(graph, TOPOLOGY)
    return data


def set_graph(data, graph):
    _store(data, graph, TOPOLOGY)


def get_allocations(graph, strategy, ratio=None):
    data = _load(graph, ALLOCATION, strategy=strategy, ratio=ratio)
    return data


def set_allocations(data, graph, strategy, ratio=None):
    _store(data, graph, ALLOCATION, strategy=strategy, ratio=ratio)


def get_diameter(graph):
    data = _load(graph, DIAMETER)
    return data


def set_diameter(data, graph):
    _store(data, graph, DIAMETER)


def get_degrees(graph):
    data = _load(graph, DEGREE)
    return data


def set_degrees(data, graph):
    _store(data, graph, DEGREE)


def get_cover(graph, strategy, ratio=None):
    data = _load(graph, COVER, strategy=strategy, ratio=ratio)
    return data


def set_cover(data, graph, strategy, ratio=None):
    _store(data, graph, COVER, strategy=strategy, ratio=ratio)
    pass


def get_tm(graph):
    data = _load(graph, ALLOCATION_MATRIX)
    return data


def set_tm(data, graph):
    _store(data, graph, ALLOCATION_MATRIX)


def get_pl(graph):
    data = _load(graph, PATH_LENGTHS)
    return data


def set_pl(data, graph):
    _store(data, graph, PATH_LENGTHS)


def get_pc(graph, ratio=None):
    data = _load(graph, PATH_COUNTS, ratio=ratio)
    return data


def set_pc(data, graph, ratio=None):
    _store(data, graph, PATH_COUNTS, ratio=ratio)


def get_shortest_paths(graph, ratio=None):
    data = _load(graph, SHORTEST_PATH, ratio=ratio)
    return data


def set_shortest_paths(data, graph, ratio=None):
    _store(data, graph, SHORTEST_PATH, ratio=ratio)


'''
def get_graph_name_from_dir(dir_name):
    path = os.path.normpath(dir_name)
    return os.path.basename(path)


def n_nodes_from_name(dir_name):
    """Get the number of nodes from the directory name of the graph."""
    dir_name = get_graph_name_from_dir(dir_name)
    n_nodes = dir_name.split("(")[1]
    n_nodes = n_nodes.split(")")[0]
    n_nodes = int(n_nodes.split(",")[0])
    return n_nodes
'''
'''

def save_data(dir_name, file_name, data):
    path = os.path.join(dir_name, file_name)
    if file_name.endswith("json"):
        with open(path, "w+") as outfile:

            json.dump(data, outfile)

    elif file_name.endswith("txt"):
        with open(path, "w+") as outfile:
            outfile.write(str(data))
    elif file_name.endswith("pkl"):
        with open(path, "wb+") as outfile:
            pickle.dump(data, outfile)
    elif file_name.endswith("csv"):
        with open(path,"w+") as outfile:
            np.savetxt(outfile, data, delimiter=',')
    else:
        raise ValueError(f"Data does not have a valid extension: {file_name}")
    return data


def _get_data(dir_name, file_name):
    #print(f'get data for {dir_name} and {file_name}')
    path = os.path.join(dir_name, file_name)
    if file_name.endswith("json"):
        #print('in json')
        with open(path, "r") as infile:
            #print('could open')
            data = json.load(infile)
            #print('could load')
    elif file_name.endswith("txt"):
        with open(path, "r") as infile:
            data = infile.read()
    elif file_name.endswith("pkl"):
        with open(path, "rb") as infile:
            data = pickle.load(infile)
    elif file_name.endswith("csv"):
        data = np.genfromtxt(path,delimiter=',')
    else:
        raise ValueError(f"Data does not have a valid extension: {file_name}")
    return data
'''

'''
def load_all_metrics(dir_name):
    dirpath = os.path.join(dir_name)
    n_nodes = n_nodes_from_name(dir_name)
    all_metrics = {"n_nodes": n_nodes}
    for cur_metrics_file in os.listdir(dirpath):
        try:
            data = _get_data(dir_name, cur_metrics_file)
            metric_name = cur_metrics_file.split(".", 1)[0]
            all_metrics[metric_name] = data
        except ValueError:
            continue
    return all_metrics

'''
'''
def save_graph(dir_name, graph):
    graph_name = get_graph_name_from_dir(dir_name)
    graph_name = f"{graph_name}_topology.xml"
    gpath = os.path.join(dir_name, graph_name)
    fnss.write_topology(graph, gpath)


def load_graph(dir_name):
    graph_name = get_graph_name_from_dir(dir_name)
    graph_name = f"{graph_name}_topology.xml"
    #print(dir_name)
    gpath = os.path.join(dir_name, graph_name)
    graph = fnss.read_topology(gpath)
    return graph


def load_degrees(dir_name):
    data = _get_data(dir_name, DEGREE_FILE_NAME)
    degrees = data["degrees"]
    nodes = data["nodes"]
    return nodes, degrees


def load_allocations(dir_name, use_tm):
    if not use_tm:
        data = _get_data(dir_name, kalloc_file_name(1))
    else:
        print(f"Loading {TM_ALLOC_FILE_NAME}")
        data = _get_data(dir_name, kalloc_tm_file_name(1))
    return data


def load_k_alloc_sum_ratio(dir_name, strat, num_sp, use_tm, ratio):
    """Load the sum of the allocations over k shortest paths."""
    if not use_tm:
        data = _get_data(dir_name, kalloc_sum_file_name_ratio(strat, num_sp, ratio))
    else:
        data = _get_data(dir_name, kalloc_sum_tm_file_name_ratio(strat, num_sp, ratio))
    return data

def load_k_alloc_sum_as_matrix_ratio(dir_name, strat, num_sp, use_tm, nodes, ratio):

    json = load_k_alloc_sum_ratio(dir_name, strat, num_sp, True, ratio)

    n_nodes = len(json)
    mat = np.zeros(shape=(n_nodes, n_nodes))

    for i in [0, n_nodes-1]:
        for j in [0, n_nodes-1]:
            if not i==j and str(i) in json and str(j) in json[str(i)]:

                mat[i][j] = json[str(nodes[i])][str(nodes[j])]
                #mat[i][j] = json[i][j]
            else:
                mat[i][j] = -1

    return mat

def load_diameter(dir_name):
    data = _get_data(dir_name, DIAMETER_FILE_NAME)
    return int(data)

'''


'''
def load_cover(dir_name):
    data = _get_data(dir_name, COVER_FILE_NAME)
    return data


def load_shortest_paths(dir_name):
    #print(f'loading: {dir_name}, ksp: {ksp_file_name(1)}')
    data = _get_data(dir_name, ksp_file_name(1))
    #print('loaded sps')
    return data


def load_k_shortest_paths(dir_name, num_sp):
    data = _get_data(dir_name, ksp_file_name(num_sp))
    return data

def load_sampled_shortest_paths(dir_name, num_sp, ratio):
    data = _get_data(dir_name, kspr_file_name(num_sp, ratio))
    return data


def load_traffic_matrices(dir_name):
    data = _get_data(dir_name, TM_FILE_NAME)
    return data

def load_path_counters(dir_name, ratio):
    data = _get_data(dir_name, f"{ratio}_{PC_FILE_NAME}")
    return data

def load_path_length(dir_name):
    data = _get_data(dir_name, PL_FILE_NAME)
    return data
def load_n_dict(dir_name):
    data = _get_data(dir_name,ND_FILE_NAME)
    return data

def load_traffic_matrices_allc(dir_name):
    data = _get_data(dir_name, TM_ALLOC_FILE_NAME)
    return data

def load_reach(dir_name):
    data = _get_data(dir_name, REACH_FILE_NAME)
    return data
'''
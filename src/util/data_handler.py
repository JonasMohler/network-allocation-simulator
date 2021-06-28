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


def _load(graph, data_type, strategy=None, ratio=None, thresh=None):

    full_path = get_full_path(graph, data_type, strategy, ratio, thresh)

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


def _store(data, graph, data_type, strategy=None, ratio=None, thresh=None):

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
    full_path = get_full_path(graph, data_type, strategy, ratio, thresh)

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


def get_cover(graph, strategy, thresh, ratio=None):
    data = _load(graph, COVER, strategy=strategy, ratio=ratio, thresh=thresh)
    return data


def set_cover(data, graph, strategy, thresh, ratio=None):
    _store(data, graph, COVER, strategy=strategy, ratio=ratio, thresh=thresh)


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


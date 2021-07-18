from abc import abstractmethod
from collections import defaultdict
from typing import Tuple, Union
import sys

import numpy as np
import networkx as nx
from src.types import Path, PathsResult, PathsDict, AllocationMatrices
from src.path_algorithms.path_algorithm import PathAlgorithm, get_entry_conv_div


class ScaledQoSAlgorithm(PathAlgorithm):

    def __init__(self, graph: nx.Graph, allocation_matrices: AllocationMatrices, node_count=None, path_count=None):
        super(ScaledQoSAlgorithm, self).__init__()
        self.graph = graph
        self.allocation_matrices = allocation_matrices
        self.node_count = node_count
        self.path_counts = path_count

    @abstractmethod
    def calculate_norm(self, node, src, prev_node, next_node):
        raise NotImplementedError

    def compute_for_path(self, path: Path) -> float:

        if len(path) < 2:
            return 0

        cap = get_entry_cap(self.graph, self.allocation_matrices, path[0], None, path[1])

        norm = self.calculate_norm(path[0], path[0], path[0], path[1])

        t_vals = [(cap/norm) if norm != -1 else 0]

        for idx in range(1, len(path)):
            # Get the nodes for the current hop

            cur_node = path[idx]
            prev_node = path[idx - 1]
            next_node = path[idx + 1] if (idx + 1) < len(path) else None

            # Get interface-to-interface capacity
            print(f'cur node: {cur_node}')
            print(f"prev node: {prev_node}")
            print(f"next node: {next_node}")
            print(f"path: {path}")
            cap = get_entry_cap(self.graph, self.allocation_matrices, cur_node, prev_node, next_node)
            next_node = next_node if next_node is not None else cur_node
            norm = self.calculate_norm(path[idx], path[0], prev_node, next_node)

            #print(f'Cap: {cap}')
            #print(f"norm: {norm}")
            t_vals.append((cap/norm))
            #print(f"t_val: {cap/norm}")

        t_vals = np.asarray(t_vals)

        alloc = float(np.amin(t_vals))

        return alloc


class ScaledQoSAlgorithmPT(ScaledQoSAlgorithm):
    """A `PathAlgorithm` for GMA-style algorithms.

    Has as additional instance variables the graph, the allocation matrices."""

    def __init__(self, graph: nx.Graph, allocation_matrices: AllocationMatrices, path_count=None, node_count=None):
        super(ScaledQoSAlgorithmPT, self).__init__(graph, allocation_matrices, path_count=path_count, node_count=node_count)

    def calculate_norm(self, node, src, n_prev, n_next):
        #print(f"Node Count: {self.node_count}")
        return self.node_count


class ScaledQoSAlgorithmOT(ScaledQoSAlgorithm):
    """A `PathAlgorithm` for GMA-style algorithms.

    Has as additional instance variables the graph, the allocation matrices."""

    def __init__(self, graph: nx.Graph, allocation_matrices: AllocationMatrices, path_counts, node_count=None):
        super(ScaledQoSAlgorithmOT, self).__init__(graph, allocation_matrices, path_count=path_counts, node_count=node_count)
        self.path_counts = path_counts

    def calculate_norm(self, node, src, n_prev, n_next):
        ases = get_as_count_by_iface_pair(self.path_counts, node, n_prev, n_next)
        #print(f"Node Count: {self.node_count}")
        #print(f"Active As Count: {ases}")

        return ases


class ScaledQoSAlgorithmPB(ScaledQoSAlgorithm):
    """A `PathAlgorithm` for GMA-style algorithms.

    Has as additional instance variables the graph, the allocation matrices."""

    def __init__(self, graph: nx.Graph, allocation_matrices: AllocationMatrices, path_counts, node_count):
        super(ScaledQoSAlgorithmPB, self).__init__(graph, allocation_matrices, node_count=node_count, path_count=path_counts)

    def calculate_norm(self, node, src, n_prev, n_next):
        src_paths = get_per_src_traversing_path_count(self.path_counts, node, src, n_prev, n_next)
        #print(f"Node Count: {self.node_count}")
        #print(f"Per source count: {src_paths}")
        return src_paths*self.node_count


class ScaledQoSAlgorithmOB(ScaledQoSAlgorithm):
    """A `PathAlgorithm` for GMA-style algorithms.

    Has as additional instance variables the graph, the allocation matrices."""

    def __init__(self, graph: nx.Graph, allocation_matrices: AllocationMatrices, path_count=None, node_count=None):
        super(ScaledQoSAlgorithmOB, self).__init__(graph, allocation_matrices, path_count=path_count, node_count=node_count)

    def calculate_norm(self, node, src, n_prev, n_next):
        ases = get_as_count_by_iface_pair(self.path_counts, node, n_prev, n_next)
        per_src_count = get_per_src_traversing_path_count(self.path_counts, node, src, n_prev, n_next)
        #print(f"Node Count: {self.node_count}")
        #print(f"Active As Count: {ases}")
        #print(f"Per source count: {per_src_count}")
        return ases*per_src_count

'''
def get_n_paths_from_src(path_counts, node, src, nodes):
    count=0
    c = path_counts[node]
    for n in nodes:
        if (src, n) in c.items():
            count = count + c[(src, n)]
    return count


# TODO: Clarify here: Do we care about all paths going through AS *OR* all paths going through the respective interface pair?
def get_traversing_path_count(path_counts, node, nodes):
    count = 0
    current_node = path_counts[node]
    for n1 in nodes:
        for n2 in nodes:
            if (n1, n2) in current_node:
                count = count + sum(current_node[(n1, n2)].values())
    return count
'''


def get_per_src_traversing_path_count(path_counts, node, src, n_prev, n_next):

    current_node = path_counts[node]
    if (str(n_prev), str(n_next)) in current_node and str(src) in current_node[(str(n_prev), str(n_next))]:
        count = current_node[(str(n_prev), str(n_next))][str(src)]
        #print('Found')
    else:
        #print(f'No per src traversing path count for iface pair ({n_prev}, {n_next}) and src {src} in node {node}')
        return 1
    return count


def get_as_count_by_iface_pair(path_counts, node, n_prev, n_next):
    current_pair = path_counts[node][(str(n_prev), str(n_next))]
    count = len(list(current_pair.keys()))
    return count


def get_entry_cap(
    graph: nx.Graph,
    allocation_matrices: AllocationMatrices,
    cur_node: int,
    prev_node: Union[int, None],
    next_node: Union[int, None],
) -> float:

    if prev_node is None and next_node is None:
        raise ValueError("1-node paths are not permitted")
    cur_matrix = allocation_matrices[cur_node]
    internal_iface = cur_matrix.shape[0] - 1
    # Indices of neighbors
    neigh = sorted(list(graph[cur_node]))
    #print(neigh)
    # Check if we are at the beginning, middle, or end of a path
    if next_node is not None:
        try:
            #print(f"Neighbors: {neigh}; cur node: {cur_node}")
            #if next_node ==26:
            #    print(f"neigh: {neigh}")
            #    sys.exit()
            next_idx = neigh.index(next_node)
        except ValueError as e:
            print(f"Neighbors: {neigh}; error: {e}; cur node: {cur_node}")
            return 0
            #raise ValueError(e)
    else:
        next_idx = internal_iface

    if prev_node is not None:
        try:
            prev_idx = neigh.index(prev_node)
        except ValueError as e:
            print(f"Neighbors: {neigh}; error: {e}; cur node: {cur_node}")
            return 0
            #raise ValueError(e)
    else:
        prev_idx = internal_iface

    cap = cur_matrix[next_idx, prev_idx]
    return cap
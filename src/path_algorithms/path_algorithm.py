"""The base class for all algorithms that work on paths in graphs."""
from abc import abstractmethod
from collections import defaultdict
from typing import Tuple, Union

import networkx as nx
import numpy as np

from src.util.const import PRECISION
from src.types import Path, PathsResult, SinglePathsDict, AllocationMatrices

'''
from gma.gma_const import PRECISION
from gma.gma_types import Path, PathsResult, PathsDict, AllocationMatrices
'''


class PathAlgorithm:
    def __init__(self):
        pass

    @abstractmethod
    def compute_for_path(self, path: Path) -> float:
        """Compute the algorithm for the path provided."""
        raise NotImplementedError()

    '''
    def compute_for_all_paths(self, paths: SinglePathsDict) -> PathsResult:
        """Compute the algorithm for all the paths provided."""
        result: PathsResult = defaultdict(dict)
        
        for src, src_paths in paths.items():
            for dst, src_dst_path in src_paths.items():
                
                if src != dst:
                    results = self.compute_for_path(src_dst_path)
                    for path in src_dst_paths:
                        cur_result = self.compute_for_path(path)
                        #print(f'cur result: {cur_result}')
                        # Round the results to the required precision
                        cur_result = np.round(cur_result, decimals=PRECISION)
                        #print(f"rounded: {cur_result}")
                        src_dst_results.append(cur_result)
                    result[src][dst] = src_dst_results
        return result
    '''

    def compute_for_all_single_paths(self, paths: SinglePathsDict, sampled) -> PathsResult:
        """Compute the algorithm for all the paths provided."""

        i=0
        alln = len(paths.items())

        result: PathsResult = defaultdict(dict)
        for src, src_paths in paths.items():
            for dst, src_dst_path in src_paths.items():
                if src != dst:
                    #if sampled:
                    #    res = self.compute_for_path(src_dst_path)
                    #else:
                        res = self.compute_for_path(src_dst_path)
                    result[src][dst] = [np.round(res, decimals=PRECISION)]
            i=i+1
            if i == alln:
                print(f"{round(100 * i / alln, 4)}%")
            else:
                print(f"{round(100 * i / alln, 4)}%", end="\r")
        return result

    def compute_for_all_multipaths(self, paths):

        result: PathsResult = defaultdict(dict)
        #print('IN MULTIPATH')
        for src, dests in paths.items():
            for dst, d_paths in dests.items():
                rs = 0
                for p in d_paths:
                    if src!=dst:
                        res = self.compute_for_path(p)
                        rs = rs+np.round(res, decimals=PRECISION)
                result[src][dst] = rs
        #print('PRE RETURN MULTIPATH')
        return result


class GMAPathAlgorithm(PathAlgorithm):
    """A `PathAlgorithm` for GMA-style algorithms.

    Has as additional instance variables the graph and the allocation matrices."""

    def __init__(self, graph: nx.Graph, allocation_matrices: AllocationMatrices):
        super(GMAPathAlgorithm, self).__init__()
        self.graph = graph
        self.allocation_matrices = allocation_matrices

    def get_conv_div(
        self, cur_node: int, prev_node: Union[int, None], next_node: Union[int, None]
    ) -> Tuple[float, float, float]:
        """Get current allocation value, convergent, and divergent.

        Args:
            cur_node: the index of the node to compute conv and div of.
            prev_node: the incoming node for the div.
            next_node: the outgoing noded for the conv.
        """
        return get_entry_conv_div(
            self.graph, self.allocation_matrices, cur_node, prev_node, next_node
        )


def get_entry_conv_div(
    graph: nx.Graph,
    allocation_matrices: AllocationMatrices,
    cur_node: int,
    prev_node: Union[int, None],
    next_node: Union[int, None],
) -> Tuple[float, float, float]:
    """Get current allocation value, convergent, and divergent.

    Args:
        graph: the graph for which to perform the evaluation.
        allocation_matrices: the allocation matrices.
        cur_node: the index of the node to compute conv and div of.
        prev_node: the incoming node for the div.
        next_node: the outgoing noded for the conv.
    """
    if prev_node is None and next_node is None:
        raise ValueError("1-node paths are not permitted")
    cur_matrix = allocation_matrices[cur_node]
    internal_iface = cur_matrix.shape[0] - 1
    # Indices of neighbors
    neigh = sorted(list(graph[cur_node]))
    cur_div = None
    # Check if we are at the beginning, middle, or end of a path
    if next_node is not None:
        try:
            next_idx = neigh.index(next_node)
        except ValueError as e:
            print(f"Neighbors: {neigh}; error: {e}; cur node: {cur_node}")
            raise ValueError(e)
    else:
        next_idx = internal_iface
    cur_conv = float(np.sum(cur_matrix[next_idx, :]))
    if prev_node is None:
        # Starting hop
        prev_idx = internal_iface
    else:
        # Intermediate hops
        try:
            prev_idx = neigh.index(prev_node)
        except ValueError as e:
            print(f"Neighbors: {neigh}; error: {e}; cur node: {cur_node}, next node: {next_node}")
            raise ValueError(e)
        cur_div = float(np.sum(cur_matrix[:, prev_idx]))
    cur_ij = float(cur_matrix[next_idx, prev_idx])
    return cur_ij, cur_conv, cur_div



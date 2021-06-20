from src.multiprocessing.node.NodeMultiprocessing import NodeMultiprocessing

import time
import numpy as np
import networkx as nx

from src.types import SinglePathsDict, PathLengthsDict, PathCountsDict, AllocationMatrices
from src.util.utility import allocation_matrix, single_shortest_path, single_shortest_path_length, per_node_alloc_to_cover
from src.util.const import *


class ShortestPathsComputation(NodeMultiprocessing):
    description = "Computation of SHORTEST PATHS per node"
    
    def __init__(self, cur_dir, nodes, n_proc, force, graph):
        super(ShortestPathsComputation, self).__init__(cur_dir, nodes, n_proc, SHORTEST_PATH, force)
        self.graph = graph

    def per_node_op(self, node):
        try:
            sp = {node: single_shortest_path(self.graph, node)}
            return sp
        except Exception as e:
            print(f"Error occured in Node Shortest Path computation: {e}")


class PathLengthComputation(NodeMultiprocessing):
    description = "Computation of PATH LENGTHS per node"

    def __init__(self, cur_dir, nodes, n_proc, force, graph):
        super(PathLengthComputation, self).__init__(cur_dir, nodes, n_proc, PATH_LENGTHS, force)
        self.graph = graph

    def per_node_op(self, node):
        try:
            pl = {node: single_shortest_path_length(self.graph, node)}
            return pl
        except Exception as e:
            print(f"Error occured in Node Path Length computation: {e}")


class PathSampling(NodeMultiprocessing):
    description = "Sampling of Paths per node"

    def __init__(self, cur_dir, nodes, n_proc, force_recompute, shortest_paths, degrees, centrality, n_dests, ratio=None):
        super(PathSampling, self).__init__(cur_dir, nodes, n_proc, SHORTEST_PATH, force_recompute, ratio=ratio)
        self.sp = shortest_paths
        self.degrees = degrees
        self.centrality = centrality
        self.n_dests = n_dests

    def per_node_op(self, cur_node):
        try:

            st = time.time()
            try:
                node_sps = self.sp[cur_node]
            except KeyError:
                print(f"Key error in sps assignment:\ncur node: {cur_node}\nsp:\n{self.sp}")

            t1 = time.time() -st
            st1 = time.time()
            # Choose #destinations nodes from shortest paths
            destinations = np.random.choice(self.nodes, size=self.n_dests, p=self.centrality, replace=False)

            t2 = time.time() -st1
            st2 = time.time()

            # Keep only selected shortest paths
            selected_paths = {}
            for d in destinations:
                if d != cur_node:

                    selected_paths[d] = node_sps[d]

            t3 = time.time() -st2
            t4 = time.time() - st
            print(f"Overall time: {t4}")
            print(f"Assignment: {t1}, %{t1/t4}")
            print(f"Choice: {t2}, %{t2/t4}")
            print(f"Select: {t3}, %{t3/t4}")

            res = {cur_node: selected_paths}

            return res

        except Exception as e:
            print(f"Error occured in Node Path Sampling: {e}")


class PathCounting(NodeMultiprocessing):
    description = "Counting of paths traversing nodes"

    def __init__(self, cur_dir, nodes, n_proc, force, shortest_paths, ratio=None):
        super(PathCounting, self).__init__(cur_dir, nodes, n_proc, PATH_COUNTS, force, ratio=ratio)
        self.shortest_paths = shortest_paths

    def per_node_op(self, cur_node):

        try:

            traversing_paths = []
            sp = self.shortest_paths

            for n in sp:

                for n2 in sp[n]:

                    if cur_node in sp[n][n2]:

                        traversing_paths.append(sp[n][n2])

            counter = dict()
            for path in traversing_paths:
                if len(path)>1:
                    src = path[0]
                    id = path.index(cur_node)

                    if id == 0:
                        # node is first on path
                        i_in = cur_node
                        i_out = path[id+1]
                    elif id == len(path) - 1:
                        # node is last on path
                        i_in = path[id-1]
                        i_out = cur_node
                    else:
                        i_in = path[id-1]
                        i_out = path[id+1]

                    if (i_in, i_out) in counter:
                        if src in counter[(i_in, i_out)]:
                            counter[(i_in, i_out)][src] = counter[(i_in, i_out)][src]+1
                        else:
                            counter[(i_in, i_out)][src] = 1
                    else:
                        counter[(i_in, i_out)] = dict()
                        counter[(i_in, i_out)][src] = 1

            res = {cur_node: counter}

            return res

        except Exception as e:
            print(f"Error occured in Node Path counting: {e}")


class CoverComputation(NodeMultiprocessing):
    description = "Cover Computation per Node"

    def __init__(self, cur_dir, nodes, n_proc, force, alloc, shortest_paths,thresh, strategy, ratio=None):
        super(CoverComputation, self).__init__(cur_dir, nodes, n_proc, COVER, force, ratio=ratio, strategy=strategy, thresh=thresh)
        self.shortest_paths = shortest_paths
        self.alloc = alloc
        '''
        self.strategy = strategy
        self.thresh = thresh
        '''


    def per_node_op(self, cur_node):


        res = per_node_alloc_to_cover(self.alloc[cur_node], self.thresh, self.shortest_paths[cur_node])
        return {cur_node: res}


class AllocationMatrixComputation(NodeMultiprocessing):
    description = "Calculate per node Allocation Matrix"

    def __init__(self, cur_dir, nodes, n_proc, force, graph):
        super(AllocationMatrixComputation, self).__init__(cur_dir, nodes, n_proc, ALLOCATION_MATRIX, force)
        self.graph = graph

    def per_node_op(self, cur_node):

        try:
            tm = allocation_matrix(self.graph, cur_node)

            res = AllocationMatrices(cur_node, tm)

            return res

        except Exception as e:
            print(f"Error occured in per node allocation matrix computation; {e}")

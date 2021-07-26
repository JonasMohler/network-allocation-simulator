"""Parallel runners for experimentation and analysis."""
import os.path

import numpy as np
import pandas as pd

from src.util.utility import *
from src.multiprocessing.topology.TopolgyMultiprocessing import TopologyMultiprocessing
import src.multiprocessing.node.PerNodeOperations as pno
from src.util.const import *
import src.util.data_handler as dh
import fnss
import math
#from multiprocessing import Array, shared_memory
from multiprocessing.managers import SyncManager

from src.path_algorithms.gma_improved import GMAImproved
from src.path_algorithms.sqos_algorithm import ScaledQoSAlgorithmOB, ScaledQoSAlgorithmOT, ScaledQoSAlgorithmPB, ScaledQoSAlgorithmPT


class PathSampling(TopologyMultiprocessing):
    description = 'Sampling of Paths'
    
    def __init__(self, dirs, n_proc, force_recompute, ratio):
        super(PathSampling, self).__init__(dirs, n_proc, SHORTEST_PATH, force_recompute)
        self.ratio = ratio

    def find_or_compute_precursors(self, cur_dir):
        # Needs shortest paths
        pass

    def per_dir_op(self, cur_dir):
        try:
            #super(PathSampling, self).per_dir_op(cur_dir)
            if not os.path.exists(dh.get_full_path(cur_dir, SHORTEST_PATH, ratio=self.ratio)):
                sp = dh.get_shortest_paths(cur_dir)
                deg = dh.get_degrees(cur_dir)
                # Calculate the centrality of all nodes
                s = sum(deg['degrees'])

                centrality = [i / s for i in deg['degrees']]
                r = float(self.ratio)

                l = len(deg['nodes'])

                n_dests = int(round(l * r))

                # for each node, select a number of destinations at random, weighted by the centrality of nodes
                proc = pno.PathSampling(cur_dir, deg['nodes'], self.n_proc, self.force, sp, deg, centrality, n_dests, self.ratio)
                proc.run()

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Path Sampling: {e}")


class PathSampling2(TopologyMultiprocessing):
    description = 'Sampling of Paths'

    def __init__(self, dirs, n_proc, force_recompute, ratio, num_sp):
        super(PathSampling2, self).__init__(dirs, n_proc, SHORTEST_PATH, force_recompute)
        self.ratio = ratio
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        # Needs shortest paths
        pass

    def per_dir_op(self, cur_dir):
        try:
            # super(PathSampling, self).per_dir_op(cur_dir)
            if not os.path.exists(dh.get_full_path(cur_dir, SHORTEST_PATH, ratio=self.ratio, num_sp=self.num_sp)):
                if self.num_sp is not None:
                    sp = dh.get_k_shortest_paths(cur_dir, self.num_sp)
                else:
                    sp = dh.get_shortest_paths(cur_dir)


                if str(self.ratio).startswith('a'):
                    # Degree-weighted number, uniform dsts
                    # Higher degree nodes select more destinations than lower degree nodes
                    # High degree nodes more likely to be picked as destination
                    #sp = dh.get_shortest_paths(cur_dir)
                    deg = dh.get_degrees(cur_dir)
                    nodes = deg['nodes']
                    degrees = deg['degrees']

                    # Calculate the weight
                    r = float(self.ratio[1:])
                    s = sum(degrees)
                    num_n = len(nodes)
                    node_is = range(num_n)
                    sample_sizes = [int(math.ceil(x*num_n*r/s)) for x in degrees]
                    all_selected_paths = {}
                    for i in range(len(nodes)):


                        dest_is = np.random.choice(node_is, sample_sizes[i], replace=False)

                        selected_paths = {}
                        for d in dest_is:
                            if nodes[d] != nodes[i]:

                                if self.num_sp is not None:
                                    tmp = []
                                    for j in range(self.num_sp):
                                        if len(sp[nodes[i]][nodes[d]]) > j:
                                            tmp.append(sp[nodes[i]][nodes[d]][j])
                                    selected_paths[nodes[d]] = tmp
                                else:
                                    selected_paths[nodes[d]] = sp[nodes[i]][nodes[d]]
                        all_selected_paths[nodes[i]] = selected_paths
                        if i == num_n:
                            print(f"{round(100 * (i - 1) / num_n, 4)}%")
                        else:
                            print(f"{round(100 * (i - 1) / num_n, 4)}%", end="\r")

                    #dh.set_shortest_paths(all_selected_paths, cur_dir, ratio=self.ratio)

                elif str(self.ratio).startswith('u'):
                    # Uniform number, uniform dsts
                    # Every node selects #nodes * ratio destinations
                    # Every node as likely to be picked as destination
                    r = float(self.ratio[1:])
                    #sp = dh.get_shortest_paths(cur_dir)
                    deg = dh.get_degrees(cur_dir)
                    nodes = deg['nodes']

                    n_nodes = len(nodes)
                    node_is = range(n_nodes)
                    n_samples = int(round(n_nodes * r))

                    all_selected_paths = {}
                    for i in range(len(nodes)):

                        dest_is = np.random.choice(node_is, n_samples, replace=False)

                        selected_paths = {}
                        for d in dest_is:
                            if nodes[d] != nodes[i]:
                                if self.num_sp is not None:
                                    tmp = []
                                    for j in range(int(self.num_sp)):
                                        if len(sp[nodes[i]][nodes[d]]) > j:
                                            tmp.append(sp[nodes[i]][nodes[d]][j])
                                        #print(j)
                                    selected_paths[nodes[d]] = tmp
                                else:
                                    selected_paths[nodes[d]] = sp[nodes[i]][nodes[d]]

                        all_selected_paths[nodes[i]] = selected_paths
                        if i == n_nodes:
                            print(f"{round(100 * (i-1) / n_nodes, 4)}%")
                        else:
                            print(f"{round(100 * (i-1) / n_nodes, 4)}%", end="\r")

                    #dh.set_shortest_paths(all_selected_paths, cur_dir, ratio=self.ratio)

                else:
                    # Uniform number, degree-weighted dsts
                    # Every node selects #nodes * ratio destinations
                    # High degree nodes more likely to be picked as destination
                    #sp = dh.get_shortest_paths(cur_dir)
                    deg = dh.get_degrees(cur_dir)
                    nodes = deg['nodes']
                    degrees = deg['degrees']
                    # Calculate the centrality of all nodes
                    s = sum(degrees)

                    centrality = [i / s for i in deg['degrees']]
                    r = float(self.ratio)

                    n_nodes = len(nodes)

                    n_samples = int(round(n_nodes * r))

                    dests = numpy_choice(n_nodes, n_samples, nodes, centrality)

                    all_selected_paths = {}
                    for i in range(len(nodes)):

                        selected_paths = {}
                        for d in dests[i]:
                            if d != nodes[i]:
                                if self.num_sp is not None:
                                    tmp = []
                                    for j in range(int(self.num_sp)):
                                        if len(sp[nodes[i]][d]) > j:
                                            tmp.append(sp[nodes[i]][d][j])
                                    selected_paths[d] = tmp
                                else:
                                    selected_paths[d] = sp[nodes[i]][d]

                        all_selected_paths[nodes[i]] = selected_paths
                        if i == n_nodes:
                            print(f"{round(100*i/n_nodes, 4)}%")
                        else:
                            print(f"{round(100*i/n_nodes, 4)}%", end="\r")

                    #dh.set_shortest_paths(all_selected_paths, cur_dir, ratio=self.ratio)

                if self.num_sp is not None:
                    dh.set_k_shortest_paths(all_selected_paths, cur_dir, self.num_sp, ratio=self.ratio)
                else:
                    dh.set_shortest_paths(all_selected_paths, cur_dir, ratio=self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Path Sampling: {e}")


class PathCounting(TopologyMultiprocessing):
    description = 'Counting of Paths'
    
    def __init__(self, dirs, n_proc, force_recompute, ratio=None):
        super(PathCounting, self).__init__(dirs, n_proc, PATH_COUNTS, force_recompute, ratio=ratio)

    def find_or_compute_precursors(self, cur_dir):
        # Needs Sampled shortest paths
        pass

    def per_dir_op(self, cur_dir):

        try:

            super(PathCounting, self).per_dir_op(cur_dir)

            deg = dh.get_degrees(cur_dir)
            sp = dh.get_shortest_paths(cur_dir, self.ratio)
            man = SyncManager()
            man.start()
            d = man.dict(sp)
            #arr = Array(lock=False)
            nodes = deg['nodes']
            #for n in nodes:
            #    arr[int(n)]=sp[n]

            proc = pno.PathCounting2(cur_dir, nodes, self.n_proc, self.force, sp, ratio=self.ratio, shared_dict=d)

            proc.run()
            man.shutdown()

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Path Counting: {e}")


class PathCounting2(TopologyMultiprocessing):
    description = 'Counting of Paths'

    def __init__(self, dirs, n_proc, force_recompute, ratio=None, num_sp=None):
        super(PathCounting2, self).__init__(dirs, n_proc, PATH_COUNTS, force_recompute, ratio=ratio)
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        # Needs Sampled shortest paths
        pass

    def per_dir_op(self, cur_dir):

        if not os.path.exists(dh.get_full_path(cur_dir, PATH_COUNTS, self.strategy, self.ratio, num_sp=self.num_sp)):
            try:
                super(PathCounting2, self).per_dir_op(cur_dir)
                res = {}
                deg = dh.get_degrees(cur_dir)
                if self.num_sp is not None:
                    sp = dh.get_k_shortest_paths(cur_dir, self.num_sp, ratio=self.ratio)
                else:
                    sp = dh.get_shortest_paths(cur_dir, self.ratio)
                nodes = deg['nodes']

                c = 0
                alln = len(nodes)
                for cur_node in nodes:
                    counter = dict()
                    print(cur_node)
                    for n in sp:
                        for n2 in sp[n]:
                            if self.num_sp is not None:
                                for i in range(int(self.num_sp)):
                                    if len(sp[n][n2]) > i:
                                        if cur_node in sp[n][n2][i]:

                                            path = sp[n][n2][i]
                                            #print(path)
                                            # for path in traversing_paths:
                                            if len(path) > 1:
                                                src = n
                                                id = path.index(cur_node)
                                                if id == 0:
                                                    # node is first on path
                                                    i_in = cur_node
                                                    i_out = path[id + 1]
                                                elif id == len(path) - 1:
                                                    # node is last on path
                                                    i_in = path[id - 1]
                                                    i_out = cur_node
                                                else:
                                                    i_in = path[id - 1]
                                                    i_out = path[id + 1]

                                                if (i_in, i_out) in counter:

                                                    if src in counter[(i_in, i_out)]:

                                                        counter[(i_in, i_out)][src] = counter[(i_in, i_out)][src] + 1
                                                    else:

                                                        counter[(i_in, i_out)][src] = 1
                                                else:

                                                    counter[(i_in, i_out)] = dict()
                                                    #print(counter[(i_in, i_out)])
                                                    counter[(i_in, i_out)][src] = 1
                                                    #print(counter[(i_in, i_out)])

                            else:
                                if cur_node in sp[n][n2]:
                                    path = sp[n][n2]
                                    # for path in traversing_paths:
                                    if len(path) > 1:
                                        src = n
                                        id = path.index(cur_node)

                                        if id == 0:
                                            # node is first on path
                                            i_in = cur_node
                                            i_out = path[id + 1]
                                        elif id == len(path) - 1:
                                            # node is last on path
                                            i_in = path[id - 1]
                                            i_out = cur_node
                                        else:
                                            i_in = path[id - 1]
                                            i_out = path[id + 1]

                                        if (i_in, i_out) in counter:
                                            if src in counter[(i_in, i_out)]:
                                                counter[(i_in, i_out)][src] = counter[(i_in, i_out)][src] + 1
                                            else:
                                                counter[(i_in, i_out)][src] = 1
                                        else:
                                            counter[(i_in, i_out)] = dict()
                                            counter[(i_in, i_out)][src] = 1
                    #print(cur_node)
                    res[cur_node] = counter
                    c=c+1
                    if c == alln:
                        print(f"{round(100*c/alln, 4)}%")
                    else:
                        print(f"{round(100*c/alln, 4)}%", end="\r")


                if self.num_sp is not None:
                    dh.set_pc(res, cur_dir, self.ratio, self.num_sp)
                else:
                    dh.set_pc(res, cur_dir, self.ratio)

                print(f"{cur_dir}: Done")
            except Exception as e:
                print(f"Error occured in Path Counting: {e}")


class PathCounting3(TopologyMultiprocessing):
    description = 'Counting of Paths'

    def __init__(self, dirs, n_proc, force_recompute, ratio=None):
        super(PathCounting3, self).__init__(dirs, n_proc, PATH_COUNTS, force_recompute, ratio=ratio)

    def find_or_compute_precursors(self, cur_dir):
        # Needs Sampled shortest paths
        pass

    def per_dir_op(self, cur_dir):

        try:
            super(PathCounting3, self).per_dir_op(cur_dir)
            res = {}
            deg = dh.get_degrees(cur_dir)
            sp = dh.get_shortest_paths(cur_dir, self.ratio)
            nodes = deg['nodes']

            cols = ['in_node', 'src_interface', 'dst_interface', 'src_node', 'count']
            df = pd.DataFrame(columns=cols)

            c = 0
            alln = len(nodes)

            for n in sp:
                for n2 in sp[n]:

                    path = sp[n][n2]
                    if len(path) > 1:
                        for cur_node in path:

                            src = n
                            id = path.index(cur_node)

                            if id == 0:
                                # node is first on path
                                i_in = cur_node
                                i_out = path[id + 1]
                            elif id == len(path) - 1:
                                # node is last on path
                                i_in = path[id - 1]
                                i_out = cur_node
                            else:
                                i_in = path[id - 1]
                                i_out = path[id + 1]

                            idx = df.index[((df['in_node'] == cur_node) & (df['src_interface'] == i_in) & (df['dst_interface'] == i_out) & (df['src_node'] == src))]
                            #print(f"ids found: {idx}")
                            if len(idx) > 0:
                            #    print('updating')
                                df.at[idx[0], 'count'] += 1
                            else:
                            #    print('first insert')
                                s = pd.Series([cur_node, i_in, i_out, src, 1], index=cols)
                            #    print(f'SERIES: {s}')
                                df = df.append(s, ignore_index=True)
                            #    print(f"DF: {df}")

            dh.set_pc(df, cur_dir, self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Path Counting: {e}")


class ShortestPathsComputation(TopologyMultiprocessing):
    description = "Computation of SHORTEST PATHS"

    def __init__(self, dirs, n_proc, force_recompute):
        super(ShortestPathsComputation, self).__init__(dirs, n_proc, SHORTEST_PATH, force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        # Needs a topology
        # Needs Degrees (Nodes only actually)
        pass

    def per_dir_op(self, cur_dir):
        try:

            super(ShortestPathsComputation, self).per_dir_op(cur_dir)
            self.find_or_compute_precursors(cur_dir)
            deg = dh.get_degrees(cur_dir)

            nodes = deg['nodes']

            graph = dh.get_graph(cur_dir)

            proc = pno.ShortestPathsComputation(cur_dir, nodes, self.n_proc, self.force, graph)

            proc.run()

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Shortest Path Computation: {e}")


class AllShortestPathsComputation(TopologyMultiprocessing):
    description = "Computation of SHORTEST PATHS"

    def __init__(self, dirs, n_proc, force_recompute):
        super(AllShortestPathsComputation, self).__init__(dirs, n_proc, SHORTEST_PATH, force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        # Needs a topology
        # Needs Degrees (Nodes only actually)
        pass

    def per_dir_op(self, cur_dir):
        try:

            super(AllShortestPathsComputation, self).per_dir_op(cur_dir)
            self.find_or_compute_precursors(cur_dir)

            graph = dh.get_graph(cur_dir)

            sps = all_shortest_paths(graph)
            dh.set_shortest_paths(sps, cur_dir)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Shortest Path Computation: {e}")


class AllKShortestPathsComputation(TopologyMultiprocessing):
    description = "Computation of K SHORTEST PATHS"

    def __init__(self, dirs, n_proc, force_recompute, k):
        super(AllKShortestPathsComputation, self).__init__(dirs, n_proc, SHORTEST_PATH, force_recompute)
        self.k = int(k)

    def find_or_compute_precursors(self, cur_dir):
        # Needs a topology
        # Needs Degrees (Nodes only actually)
        pass

    def per_dir_op(self, cur_dir):
        try:

            super(AllKShortestPathsComputation, self).per_dir_op(cur_dir)
            self.find_or_compute_precursors(cur_dir)

            graph = dh.get_graph(cur_dir)

            sps = all_k_shortest_paths(graph, self.k)
            dh.set_k_shortest_paths(sps, cur_dir, str(self.k))

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Shortest Path Computation: {e}")


class DegreesComputation(TopologyMultiprocessing):
    """Compute the degrees of the graph."""

    description = "Computation of graph node DEGREES"

    def __init__(self, dirs, n_proc, force_recompute):
        super(DegreesComputation, self).__init__(dirs, n_proc, DEGREE, force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        # Needs a Topology
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(DegreesComputation, self).per_dir_op(cur_dir)

            graph = dh.get_graph(cur_dir)
            nodes, degrees = all_degrees(graph)
            data = {
                "nodes": nodes,
                "degrees": degrees,
            }

            dh.set_degrees(data, cur_dir)

            print(f"{cur_dir}: Done")

        except Exception as e:
            print(f"Error occured in Degrees Computation: {e}")


class DiameterComputation(TopologyMultiprocessing):
    """Compute the diameter of the network given shortest paths."""

    description = "Computation of graph DIAMETER"

    def __init__(self, dirs, n_proc, force_recompute):
        super(DiameterComputation, self).__init__(dirs, n_proc, DIAMETER, force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        # Needs a Topology
        pass

    def per_dir_op(self, cur_dir):
        try:
            path = dh.get_full_path(cur_dir, self.data_type)
            if not os.path.exists(path):
                super(DiameterComputation, self).per_dir_op(cur_dir)

                sps = dh.get_shortest_paths(cur_dir)
                diameters = shortest_to_diameter(sps)

                dh.set_diameter(diameters, cur_dir)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occured in Diameter Computation: {e}")


class CoverageComputation(TopologyMultiprocessing):
    """Compute coverage given an allocation."""

    description = "Computation of graph COVERAGE"

    def __init__(self, dirs, n_proc, strategy, force_recompute, c_thresh, ratio=None, num_sp=None):
        super(CoverageComputation, self).__init__(dirs, n_proc, COVER, force_recompute, strategy, ratio)
        self.cover_thresh = c_thresh
        self.description = self.description + " with "+self.strategy+f" ALLOCATION MATRICES and thresh: {c_thresh}"
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        # Needs Allocations
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(CoverageComputation, self).per_dir_op(cur_dir)

            alloc = dh.get_allocations(cur_dir, self.strategy, self.ratio, self.num_sp)
            if self.num_sp is not None:
                sp = dh.get_k_shortest_paths(cur_dir, self.num_sp, self.ratio)
                cover = alloc_to_cover(alloc, self.cover_thresh, sp, True)
            else:
                sp = dh.get_shortest_paths(cur_dir, self.ratio)
                cover = alloc_to_cover(alloc, self.cover_thresh, sp, False)

            dh.set_cover(cover, cur_dir, self.strategy, self.cover_thresh, self.ratio, self.num_sp)

            print(f"{cur_dir}: Done")

        except Exception as e:
            print(f"Error occured in Coverage Computation: {e}")


class AllocationMatrixComputation(TopologyMultiprocessing):
    """Compute all the allocation matrices per topology."""

    description = "Computation of the ALLOCATION MATRICES"

    def __init__(self, dirs, n_proc, force_recompute):
        super(AllocationMatrixComputation, self).__init__(dirs, n_proc, ALLOCATION_MATRIX, force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(AllocationMatrixComputation, self).per_dir_op(cur_dir)

            path = dh.get_full_path(cur_dir, self.data_type)

            if not os.path.exists(path) or self.force:

                graph = dh.get_graph(cur_dir)
                allocation_matrices = all_allocation_matrices(graph)
                dh.set_tm(allocation_matrices, cur_dir)

            print(f"{cur_dir}: Done")

        except Exception as e:
            print(f"Error occured in TM calculation: {e}")


class GMAAllocationComputation(TopologyMultiprocessing):

    description = "Computations of GMA Allocations"

    def __init__(self, dirs, n_proc, force_recompute, num_sp=None):
        super(GMAAllocationComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'GMAImproved'
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(GMAAllocationComputation, self).per_dir_op(cur_dir)
            if not os.path.exists(dh.get_full_path(cur_dir, ALLOCATION, self.strategy, num_sp=self.num_sp)):
                #path = dh.get_full_path(cur_dir, self.data_type, self.strategy)
                
                #if not os.path.exists(path) or self.force:
                
                graph = dh.get_graph(cur_dir)
                tm = dh.get_tm(cur_dir)
                strat = GMAImproved(graph, tm)
                
                if self.num_sp is not None:
                    sp = dh.get_k_shortest_paths(cur_dir, self.num_sp)
                    allocs = strat.compute_for_all_multipaths(sp)
                    dh.set_allocations(allocs, cur_dir, self.strategy, num_sp=self.num_sp)
                else:
                    sp = dh.get_shortest_paths(cur_dir)
                    allocs = strat.compute_for_all_single_paths(sp, False)
                    dh.set_allocations(allocs, cur_dir, self.strategy)
                
                print(f"{cur_dir}: Done")

        except Exception as e:
            print(f"Error occurred in GMA Allocation computation")


class SQoSOTComputation(TopologyMultiprocessing):

    description = "Computation of the ALLOCATIONS using Optimistic SQoS with Time Division"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None, num_sp=None):

        super(SQoSOTComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_ot'
        self.ratio = ratio
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSOTComputation, self).per_dir_op(cur_dir)

            graph = dh.get_graph(cur_dir)
            tm = dh.get_tm(cur_dir)
            counts = dh.get_pc(cur_dir, self.ratio, self.num_sp)

            strat = ScaledQoSAlgorithmOT(graph, tm, path_counts=counts, node_count=dh.sfname(cur_dir))

            if self.num_sp is not None:
                sp = dh.get_k_shortest_paths(cur_dir, self.num_sp, self.ratio)
                allocs = strat.compute_for_all_multipaths(sp)
                dh.set_allocations(allocs, cur_dir, self.strategy, ratio=self.ratio, num_sp=self.num_sp)
            else:
                sp = dh.get_shortest_paths(cur_dir, self.ratio)
                alloc = strat.compute_for_all_single_paths(sp, True)
                dh.set_allocations(alloc, cur_dir, self.strategy, ratio=self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class SQoSOBComputation(TopologyMultiprocessing):

    description = "Computation of the ALLOCATIONS using Optimistic SQoS with Bandwidth Divison"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None, num_sp=None):

        super(SQoSOBComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_ob'
        self.ratio = ratio
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSOBComputation, self).per_dir_op(cur_dir)

            graph = dh.get_graph(cur_dir)
            tm = dh.get_tm(cur_dir)
            counts = dh.get_pc(cur_dir, self.ratio, self.num_sp)

            strat = ScaledQoSAlgorithmOB(graph, tm, path_count=counts, node_count=dh.sfname(cur_dir))

            if self.num_sp is not None:
                sp = dh.get_k_shortest_paths(cur_dir, self.num_sp, self.ratio)
                allocs = strat.compute_for_all_multipaths(sp)
                dh.set_allocations(allocs, cur_dir, self.strategy, ratio=self.ratio, num_sp=self.num_sp)
            else:
                sp = dh.get_shortest_paths(cur_dir, self.ratio)
                alloc = strat.compute_for_all_single_paths(sp, True)
                dh.set_allocations(alloc, cur_dir, self.strategy, ratio=self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class SQoSPTComputation(TopologyMultiprocessing):

    description = "Computation of the ALLOCATIONS using Pessimistic SQoS with Time Divison"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None, num_sp=None):

        super(SQoSPTComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_pt'
        self.ratio = ratio
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSPTComputation, self).per_dir_op(cur_dir)
            if not os.path.exists(dh.get_full_path(cur_dir, ALLOCATION, self.strategy, num_sp=self.num_sp)):
                graph = dh.get_graph(cur_dir)
                tm = dh.get_tm(cur_dir)

                strat = ScaledQoSAlgorithmPT(graph, tm, node_count=graph.number_of_nodes())

                if self.num_sp is not None:
                    sp = dh.get_k_shortest_paths(cur_dir, self.num_sp)
                    #print('pre alloc comp')
                    allocs = strat.compute_for_all_multipaths(sp)
                    #print('here')
                    dh.set_allocations(allocs, cur_dir, self.strategy, num_sp=self.num_sp)
                else:
                    sp = dh.get_shortest_paths(cur_dir)
                    allocs = strat.compute_for_all_single_paths(sp, False)
                    dh.set_allocations(allocs, cur_dir, self.strategy)

                print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class SQoSPBComputation(TopologyMultiprocessing):

    description = "Computation of the ALLOCATIONS using Pessimistic SQoS with Bandwidth Divison"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None, num_sp=None):

        super(SQoSPBComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_pb'
        self.ratio = ratio
        self.num_sp = num_sp

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSPBComputation, self).per_dir_op(cur_dir)
            if not os.path.exists(dh.get_full_path(cur_dir, ALLOCATION, self.strategy, num_sp=self.num_sp)):
                graph = dh.get_graph(cur_dir)
                tm = dh.get_tm(cur_dir)
                path_counts = dh.get_pc(cur_dir, self.ratio, self.num_sp)

                strat = ScaledQoSAlgorithmPB(graph, tm, path_counts=path_counts, node_count=graph.number_of_nodes())
                if self.num_sp is not None:
                    sp = dh.get_k_shortest_paths(cur_dir, self.num_sp)
                    allocs = strat.compute_for_all_multipaths(sp)
                    dh.set_allocations(allocs, cur_dir, self.strategy, num_sp=self.num_sp)
                else:
                    sp = dh.get_shortest_paths(cur_dir)
                    allocs = strat.compute_for_all_single_paths(sp, False)
                    dh.set_allocations(allocs, cur_dir, self.strategy)

                print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class PathLengthComputation2(TopologyMultiprocessing):

    description = "Counting of Path shortest path lenghts"

    def __init__(self, dirs, n_proc, force_recompute):
        super(PathLengthComputation2, self).__init__(dirs, n_proc, 'path_lengths.csv', force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            #super(PathLengthComputation, self).per_dir_op(cur_dir)
            deg = dh.get_degrees(cur_dir)
            nodes = deg['nodes']
            sps = dh.get_shortest_paths(cur_dir)
            graph = dh.get_graph(cur_dir)

            pls = {}
            i = 0
            alln = len(nodes)
            for n in nodes:
                pl = {}
                for n2 in nodes:
                    pl[n2] = len(sps[n][n2])#[0])
                pls[n] = pl

            i = i+1
            if i == alln:
                print(f"{round(100 * i / alln, 4)}%")
            else:
                print(f"{round(100 * i / alln, 4)}%", end="\r")

            dh.set_pl(pls, cur_dir)

            print(f"{cur_dir}: Done")

        except Exception as e:
            print(f'Error occured in Path Length Computation: {e}')


class PathLengthComputation(TopologyMultiprocessing):

    description = "Counting of Path shortest path lenghts"

    def __init__(self, dirs, n_proc, force_recompute):
        super(PathLengthComputation, self).__init__(dirs,n_proc, 'path_lengths.csv', force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            #super(PathLengthComputation, self).per_dir_op(cur_dir)
            deg = dh.get_degrees(cur_dir)
            nodes = deg['nodes']
            sps = dh.get_shortest_paths(cur_dir)
            graph = dh.get_graph(cur_dir)

            proc = pno.PathLengthComputation(cur_dir, nodes, self.n_proc, self.force, graph)

            proc.run()

            print(f"{cur_dir}: Done")

        except Exception as e:
            print(f'Error occured in Path Length Computation: {e}')


class AddConstantCapacity(TopologyMultiprocessing):
    def __init__(self, dirs, n_proc, capacity):
        super(AddConstantCapacity, self).__init__(dirs, n_proc, TOPOLOGY, True)
        self.capacity = capacity

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        graph = dh.get_graph(cur_dir)
        fnss.set_capacities_constant(graph, self.capacity)
        graph = set_internal_cap_const(graph, self.capacity)
        dh.set_graph(graph, cur_dir)


class AddDegreeGravityCapacity(TopologyMultiprocessing):
    def __init__(self, dirs, cap_levels, n_proc):
        super(AddDegreeGravityCapacity, self).__init__(dirs, n_proc, TOPOLOGY, True)
        self.cap_levels = cap_levels

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        graph = dh.get_graph(cur_dir)
        fnss.set_capacities_degree_gravity(graph, self.cap_levels, capacity_unit=UNIT)
        graph = set_internal_cap_max_link(graph)
        dh.set_graph(graph, cur_dir)


class AddInternalFractionCapacity(TopologyMultiprocessing):
    def __init__(self, dirs, n_proc, capacity, fraction):
        super(AddInternalFractionCapacity, self).__init__(dirs, n_proc, TOPOLOGY, True)
        self.capacity = capacity
        self.fraction = fraction

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        graph = dh.get_graph(cur_dir)
        fnss.set_capacities_constant(graph, self.capacity)
        # Set internal bandwidth as minimum of the fraction and the capacity.
        graph = set_internal_cap_fraction_out(
            graph, self.fraction, min_cap=self.capacity
        )
        dh.set_graph(graph, cur_dir)

"""Parallel runners for experimentation and analysis."""
import os.path

from src.util.utility import *
from src.multiprocessing.topology.TopolgyMultiprocessing import TopologyMultiprocessing
import src.multiprocessing.node.PerNodeOperations as pno
from src.util.const import *
import src.util.data_handler as dh

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
            super(PathSampling, self).per_dir_op(cur_dir)

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
            nodes = deg['nodes']

            proc = pno.PathCounting(cur_dir, nodes, self.n_proc, self.force, sp, ratio=self.ratio)

            proc.run()

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

    def __init__(self, dirs, n_proc, strategy, force_recompute, c_thresh, ratio=None):
        super(CoverageComputation, self).__init__(dirs, n_proc, COVER, force_recompute, strategy, ratio)
        self.cover_thresh = c_thresh
        self.description = self.description + " with "+self.strategy+f" ALLOCATION MATRICES and thresh: {c_thresh}"

    def find_or_compute_precursors(self, cur_dir):
        # Needs Allocations
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(CoverageComputation, self).per_dir_op(cur_dir)

            alloc = dh.get_allocations(cur_dir, self.strategy, self.ratio)
            sp = dh.get_shortest_paths(cur_dir, self.ratio)
            deg = dh.get_degrees(cur_dir)
            nodes = deg['nodes']

            proc = pno.CoverComputation(cur_dir, nodes, self.n_proc, self.force, sp, alloc, self.cover_thresh, self.strategy, self.ratio)
            proc.run()

            #cover = alloc_to_cover(alloc, self.cover_thresh, sp)

            #dh.set_cover(cover, cur_dir, self.strategy, self.ratio)

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

    def __init__(self, dirs, n_proc, force_recompute):
        super(GMAAllocationComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'GMAImproved'

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(GMAAllocationComputation, self).per_dir_op(cur_dir)

            path = dh.get_full_path(cur_dir, self.data_type, self.strategy)

            if not os.path.exists(path) or self.force:

                sp = dh.get_shortest_paths(cur_dir)
                graph = dh.get_graph(cur_dir)
                tm = dh.get_tm(cur_dir)
                strat = GMAImproved(graph, tm)
                allocs = strat.compute_for_all_single_paths(sp)

                dh.set_allocations(allocs, cur_dir, self.strategy)

            print(f"{cur_dir}: Done")

        except Exception as e:
            print(f"Error occurred in GMA Allocation computation")


class SQoSOTComputation(TopologyMultiprocessing):
    """Compute the allocations for the k-shortest-paths case.

    This class uses the "Strategy" design pattern.
    A `PathAlgorithm` class (the strategy) computes the path allocations.
    """

    description = "Computation of the ALLOCATIONS using Optimistic SQoS with Time Division"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None):

        super(SQoSOTComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_ot'
        self.ratio = ratio

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSOTComputation, self).per_dir_op(cur_dir)
            sps = dh.get_shortest_paths(cur_dir, self.ratio)

            graph = dh.get_graph(cur_dir)
            tm = dh.get_tm(cur_dir)
            counts = dh.get_pc(cur_dir, self.ratio)

            strat = ScaledQoSAlgorithmOT(graph, tm, path_counts=counts)
            alloc = strat.compute_for_all_single_paths(sps)

            dh.set_allocations(alloc, cur_dir, self.strategy, self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class SQoSOBComputation(TopologyMultiprocessing):
    """Compute the allocations for the k-shortest-paths case.

    This class uses the "Strategy" design pattern.
    A `PathAlgorithm` class (the strategy) computes the path allocations.
    """

    description = "Computation of the ALLOCATIONS using Optimistic SQoS with Bandwidth Divison"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None):

        super(SQoSOBComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_ob'
        self.ratio = ratio

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSOBComputation, self).per_dir_op(cur_dir)
            sps = dh.get_shortest_paths(cur_dir, self.ratio)

            graph = dh.get_graph(cur_dir)
            tm = dh.get_tm(cur_dir)
            counts = dh.get_pc(cur_dir, self.ratio)

            strat = ScaledQoSAlgorithmOB(graph, tm, path_count=counts)
            alloc = strat.compute_for_all_single_paths(sps)

            dh.set_allocations(alloc, cur_dir, self.strategy, self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class SQoSPTComputation(TopologyMultiprocessing):
    """Compute the allocations for the k-shortest-paths case.

    This class uses the "Strategy" design pattern.
    A `PathAlgorithm` class (the strategy) computes the path allocations.
    """

    description = "Computation of the ALLOCATIONS using Pessimistic SQoS with Time Divison"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None):

        super(SQoSPTComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_pt'
        self.ratio = ratio

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSPTComputation, self).per_dir_op(cur_dir)
            sps = dh.get_shortest_paths(cur_dir, self.ratio)

            graph = dh.get_graph(cur_dir)
            tm = dh.get_tm(cur_dir)

            strat = ScaledQoSAlgorithmPT(graph, tm, node_count=graph.number_of_nodes())
            alloc = strat.compute_for_all_single_paths(sps)

            dh.set_allocations(alloc, cur_dir, self.strategy, self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class SQoSPBComputation(TopologyMultiprocessing):
    """Compute the allocations for the k-shortest-paths case.

    This class uses the "Strategy" design pattern.
    A `PathAlgorithm` class (the strategy) computes the path allocations.
    """

    description = "Computation of the ALLOCATIONS using Pessimistic SQoS with Bandwidth Divison"

    def __init__(self, dirs, n_proc, force_recompute, ratio=None):

        super(SQoSPBComputation, self).__init__(dirs, n_proc, ALLOCATION, force_recompute)
        self.strategy = 'sqos_pb'
        self.ratio = ratio

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            super(SQoSPBComputation, self).per_dir_op(cur_dir)
            sps = dh.get_shortest_paths(cur_dir, self.ratio)

            graph = dh.get_graph(cur_dir)
            tm = dh.get_tm(cur_dir)
            path_counts = dh.get_pc(cur_dir, self.ratio)

            strat = ScaledQoSAlgorithmPB(graph, tm, path_counts=path_counts, node_count=graph.number_of_nodes())
            alloc = strat.compute_for_all_single_paths(sps)

            dh.set_allocations(alloc, cur_dir, self.strategy, self.ratio)

            print(f"{cur_dir}: Done")
        except Exception as e:
            print(f"Error occurred in SQoS Allocation Computation: {e}")


class PathLengthComputation(TopologyMultiprocessing):

    description = "Counting of Path shortest path lenghts"

    def __init__(self, dirs, n_proc, force_recompute):
        super(PathLengthComputation, self).__init__(dirs,n_proc, 'path_lengths.csv', force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        try:
            deg = dh.get_degrees(cur_dir)
            nodes = deg['nodes']
            sps = dh.get_shortest_paths(cur_dir)
            graph = dh.get_graph(cur_dir)

            proc = pno.PathLengthComputation(cur_dir, nodes, self.n_proc, self.force, graph)

            proc.run()

            print(f"{cur_dir}: Done")

        except Exception as e:
            print(f'Error occured in Path Length Computation: {e}')


'''
class AllocationComparison(TopologyMultiprocessing):

    description = "Comparison of allocations of different Strategies"

    def __init__(self, dirs, n_proc):
        super(AllocationComparison, self).__init__(dirs, n_proc)

    def find_or_compute_precursors(self, cur_dir):
        pass

    def mat_to_json(self, mat, nodes):
        shape = np.shape(mat)
        json = {}
        for i in range(0,shape[0]):
            json[nodes[i]] = {}
            for j in range(0,shape[1]):
                json[nodes[i]][nodes[j]] = mat[i][j]

        return json

    def diff_to_json(self, a, b, nodes):
        out = {}
        for src, dsts in a.items():
            out[src] = {}
            #print('outer')
            for dst, alloc in dsts.items():
                #print('inner')
                if  src in b and dst in b[src]:
                    #print('path present')
                    out[src][dst] = alloc - b[src][dst]
        return out

    def per_dir_op(self, cur_dir):

        try:
            strategies = ['GMAImproved','ScaledQoSAlgorithm_0','ScaledQoSAlgorithm_1','ScaledQoSAlgorithm2_0','ScaledQoSAlgorithm2_1']

            nodes = load_degrees(cur_dir)[0]

            mats = {}
            for s in strategies:
                if s == 'ScaledQoSAlgorithm2_0' or s == 'ScaledQoSAlgorithm2_1':
                    mats[s] = load_k_alloc_sum_ratio(cur_dir,s,1,True, '0.1')
                else:
                    mats[s] = load_k_alloc_sum_ratio(cur_dir, s, 1,True, '1')

            diffs = {}
            for s in strategies:
                if not s == 'GMAImproved':

                    diffs[('GMAImproved',s)] = self.diff_to_json(mats['GMAImproved'],mats[s], nodes)

                    save_data(cur_dir, f"Diff_GMA_{s}.json", diffs[('GMAImproved', s)])
        except Exception as e:
            print(f'Error occured in Allocation Comparison: {e}')

'''
class DataAggregation(TopologyMultiprocessing):

    def find_or_compute_precursors(self, cur_dir):
        # Needs Allocations
        pass

    def per_dir_op(self, cur_dir):
        super(DataAggregation, self).per_dir_op(cur_dir)

    pass


class TopologyCreation(TopologyMultiprocessing):

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        super(TopologyCreation, self).per_dir_op(cur_dir)
        pass


'''
class AllShortestPathsComputation(TopologyMultiprocessing):
    description = "Computation of SHORTEST PATHS"

    def __init__(self, dirs, n_proc, force_recompute):
        super(AllShortestPathsComputation, self).__init__(dirs, n_proc, ksp_file_name('1'), force_recompute)

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):
        if not os.path.exists(os.path.join(os.path.join(os.getcwd(), cur_dir), self.out)):
            #super(AllShortestPathsComputation, self).per_dir_op(cur_dir)
            graph = load_graph(cur_dir)
            sps = all_shortest_paths(graph)
            save_data(cur_dir, ksp_file_name(1), sps)  # 1_shortest_paths
'''


'''
class SimplePathAllocationComputation(TopologyMultiprocessing):
    """Compute the allocations for the k-shortest-paths case."""

    description = "Computation of ALLOCATIONS With the SIMPLIFIED ALGORITHM"

    def __init__(self, dirs, n_proc, num_sp, strat):
        super(SimplePathAllocationComputation, self).__init__(dirs, n_proc)
        self.num_sp = num_sp
        self.strat = strat

    def find_or_compute_precursors(self, cur_dir):
        # Needs Degrees
        # Needs shortest paths
        pass

    def per_dir_op(self, cur_dir):
        super(SimplePathAllocationComputation, self).per_dir_op(cur_dir)

        sps = load_k_shortest_paths(cur_dir, self.num_sp)
        nodes, degrees = load_degrees(cur_dir)
        alloc = compute_allocations_simple(sps, nodes, degrees)
        alloc_sum = k_alloc_to_sum(alloc)
        save_data(cur_dir, kalloc_file_name_ratio(self.strat, self.num_sp, 1), alloc)
        save_data(cur_dir, kalloc_sum_file_name_ratio(self.strat, self.num_sp, 1), alloc_sum)
'''
'''
class MatrixPathAllocationComputation(TopologyMultiprocessing):
    """Compute the allocations for the k-shortest-paths case.

    This class uses the "Strategy" design pattern.
    A `GMAPathAlgorithm` class (the strategy) computes the path allocations.
    """

    description = "Computation of the ALLOCATIONS using ALLOCATION MATRICES"

    def __init__(self, dirs, n_proc, num_sp, name, strategy, force_recompute):
        super(MatrixPathAllocationComputation, self).__init__(dirs, n_proc,'', force_recompute)
        self.num_sp = num_sp
        self.strategy = strategy
        self.name = name

    def find_or_compute_precursors(self, cur_dir):
        pass

    def per_dir_op(self, cur_dir):

        try:
            sps = load_k_shortest_paths(cur_dir, self.num_sp)
            graph = load_graph(cur_dir)
            tm = load_traffic_matrices(cur_dir)

            # Compute the paths with the selected strategy
            alloc_computer = self.strategy(graph, tm)
            alloc = alloc_computer.compute_for_all_paths(sps)
            # Get the sum of the allocations in multipath
            alloc_sum = k_alloc_to_sum(alloc)

            save_data(
                cur_dir,
                kalloc_tm_file_name_ratio(self.name, self.num_sp, 1),
                alloc,
            )
            save_data(
                cur_dir,
                kalloc_sum_tm_file_name_ratio(self.name, self.num_sp, 1),
                alloc_sum,
            )
        except Exception as e:
            print(f"Error occured in Matrix Path Allocation Computation: {e}")
'''


if __name__ == "__main__":
    NotImplementedError()
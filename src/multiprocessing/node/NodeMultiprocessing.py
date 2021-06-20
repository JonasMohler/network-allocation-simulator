"""Run an operation in parallel on many directories."""
import time
from abc import abstractmethod
import src.util.data_handler as dh
from src.util.const import *
from src.util.utility import dict_from_list
from multiprocessing.pool import Pool


class NodeMultiprocessing:
    description = ""

    def __init__(self, dir, nodes, n_proc, data_type, force_recompute, strategy=None, ratio=None, thresh=None):
        print(f"----")

        self.dir = dir
        self.force = force_recompute
        self.nodes = nodes
        self.n_proc = n_proc
        self.ratio = ratio
        self.data_type = data_type
        self.strategy = strategy
        self.thresh = thresh

    'Takes current node, must return result of form {node: compRes}'
    @abstractmethod
    def per_node_op(self, cur_node):
        """Run this operation on every node selected."""
        raise NotImplementedError

    def update(self, data):
        pass

    def write(self, data):
        if self.data_type == SHORTEST_PATH:
            print('here')
            dh.set_shortest_paths(data, self.dir, self.ratio)
        elif self.data_type == PATH_COUNTS:
            dh.set_pc(data, self.dir, self.ratio)
        elif self.data_type == PATH_LENGTHS:
            dh.set_pl(data, self.dir)
        elif self.data_type == ALLOCATION_MATRIX:
            dh.set_tm(data, self.dir)
        elif self.data_type == COVER:
            dh.set_cover(data, self.dir, self.strategy, self.thresh, self.ratio)
        else:
            print('unknown data type to store results to')

    def run(self):
        """Run `per_node_op` on all nodes."""

        if VERBOSITY >= 2: print(f"START: {self.description}...")
        st = time.time()

        #path = dh.get_full_path(self.dir, self.data_type, self.strategy, self.thresh, self.ratio)

        #if self.force or not os.path.exists(path):

        pool_inner = Pool(self.n_proc)

        res = pool_inner.map_async(self.per_node_op, self.nodes)

        pool_inner.close()

        pool_inner.join()

        result = dict_from_list(res.get())

        self.write(result)

        if VERBOSITY >= 2: print(f"END: {self.description}; completed in {time.time() - st}")

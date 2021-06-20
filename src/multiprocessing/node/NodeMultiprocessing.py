"""Run an operation in parallel on many directories."""
import os
import time
import json
from abc import abstractmethod
from argparse import ArgumentParser
import src.util.data_handler as dh
from src.util.const import *
from src.util.utility import dict_from_list



#from simulation.data_loaders import *
from multiprocessing.pool import Pool
#from concurrent.futures import ProcessPoolExecutor as Pool

#import simulation.data_loaders as dl

#from simulation.simulation_const import VERBOSITY


class NodeMultiprocessing:
    description = ""

    def __init__(self, dir, nodes, n_proc, data_type, force_recompute, ratio=None, strategy=None):
        print(f"----")

        self.dir = dir
        self.force = force_recompute
        self.nodes = nodes
        self.n_proc = n_proc
        self.ratio = ratio
        self.data_type = data_type
        self.strategy = strategy

    'Takes current node, must return result of form {node: compRes}'
    @abstractmethod
    def per_node_op(self, cur_node):
        """Run this operation on every node selected."""
        raise NotImplementedError

    '''
    def listener(self, result):
        print('in listener')
        (node, res), = result.items()
        data = _get_data(self.dir, self.out)
        data[node] = res
        save_data(self.dir, self.out, data)
    '''

    '''
    def listener(self, queue):
        """listens for messages on the q, writes to file. """
        while 1:
            #print('list')
            m = queue.get()
            #print(f"m: {m}")
            if m == 'kill':
                #print('killed')
                break
            else:

                #print(f'read queue: {list(m.keys())[0]}')
                node = list(m.keys())[0]
                res = m[node]
                #print(f"got val node: {node}, res: {res}")
                #print(f"dir: {self.dir}, out: {self.out}")

                data = dl._get_data(self.dir, self.out)
                #data = _get_data(self.dir, self.out)

                #print(f"got outfile")
                data[node]=res
                #print(f"modified outfile")
                save_data(self.dir,self.out,data)
                print('wrote from queue to file')
    '''

    '''
    def writer(self, m):
        for r in m:

            node = list(r.keys())[0]

            res = r[node]
            data = dl._get_data(self.dir, self.out)
            # data = _get_data(self.dir, self.out)

            data[node] = res

            save_data(self.dir, self.out, data)
    '''

    def write(self, data):
        if self.data_type == SHORTEST_PATH:
            dh.set_shortest_paths(data, self.dir, self.ratio)
        elif self.data_type == PATH_COUNTS:
            dh.set_pc(data, self.dir, self.ratio)
        elif self.data_type == PATH_LENGTHS:
            dh.set_pl(data, self.dir)
        elif self.data_type == ALLOCATION_MATRIX:
            dh.set_tm(data, self.dir)
        elif self.data_type == COVER:
            dh.set_cover(data, self.dir, self.strategy, self.ratio)
        else:
            print('unknown data type to store results to')

    def run(self):
        """Run `per_node_op` on all nodes."""

        if VERBOSITY >= 2: print(f"START: {self.description}...")
        st = time.time()
        path = dh.get_full_path(self.dir, self.data_type, None, self.ratio)
        if self.force or not os.path.exists(path):

            '''
            # Create outfile for current operation with all nodes
            default_dict = {}
            for n in self.nodes:
                default_dict[n]=[]

            save_data(self.dir,self.out,default_dict)
            '''

            pool_inner = Pool(self.n_proc)

            res = pool_inner.map_async(self.per_node_op, self.nodes)

            pool_inner.close()

            pool_inner.join()

            result = dict_from_list(res.get())

            self.write(result)

        if VERBOSITY >= 2: print(f"END: {self.description}; completed in {time.time() - st}")

'''
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories.")
    parser.add_argument(
        "--cores",
        default=1,
        type=int,
        help="Number of cores to be used in the computation.",
    )
    args = parser.parse_args()
    proc = SubdirMultiprocessing(args.dirs, args.cores)
    proc.run()
'''
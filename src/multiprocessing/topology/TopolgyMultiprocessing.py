"""Run an operation in parallel on many directories."""
import os
import time
from abc import abstractmethod
from argparse import ArgumentParser
#from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing.pool import Pool
import src.util.data_handler as dh
from src.util.const import VERBOSITY
import random


class TopologyMultiprocessing:
    description = ""

    def __init__(self, dirs, n_proc, data_type, force_recompute, strategy=None, ratio=None):
        print(f"\n----\n")
        self.dirs = dirs
        self.n_proc = n_proc
        self.force = force_recompute
        self.data_type = data_type
        self.strategy = strategy
        self.ratio = ratio

    @abstractmethod
    def find_or_compute_precursors(self, cur_dir):
        raise NotImplementedError

    @abstractmethod
    def per_dir_op(self, cur_dir):
        """Run this operation on every directory selected."""
        try:
            pass
        except Exception as e:
            print(f"Error occured: {e}")

    def parallel_graph_ops(self, dirs_list):
        for cur_dir in dirs_list:
            self.per_dir_op(cur_dir)
            print(f"Completed {cur_dir}")

    def run(self):
        """Run `per_dir_op` on all the directories, split by core."""
        if VERBOSITY >= 2: print(f"START: {self.description}...")


        '''

        pool = Pool(self.n_proc)
        for dir in self.dirs:
            pool.apply_async(self.per_dir_op, args=(dir,))

        pool.close()
        for dir in self.dirs:
            pool.join()
        '''
        '''
        with Pool(self.n_proc) as p:
            p.map(self.per_dir_op, self.dirs)
        '''
        '''
        for dir in self.dirs:
            print(f"\n{dir}: Start")
            self.per_dir_op(dir)
        '''
        r_dirs = self.dirs
        random.shuffle(r_dirs)
        dirs_per_split = len(r_dirs) // self.n_proc + 1
        splits = [
            r_dirs[i : i + dirs_per_split]
            for i in range(0, len(r_dirs), dirs_per_split)
        ]
        st = time.time()

        with Pool(self.n_proc) as pool:
            pool.map(self.parallel_graph_ops, splits)


        if VERBOSITY >= 2: print(f"END: {self.description}; completed in {time.time() - st}")


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
    proc = TopologyMultiprocessing(args.dirs, args.cores)
    proc.run()

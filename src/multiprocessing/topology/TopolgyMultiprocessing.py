"""Run an operation in parallel on many directories."""
import os
import time
from abc import abstractmethod
from argparse import ArgumentParser
#from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing.pool import Pool
import src.util.data_handler as dh
from src.util.const import VERBOSITY


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

    def run(self):
        """Run `per_dir_op` on all the directories, split by core."""
        if VERBOSITY >= 2: print(f"START: {self.description}...")
        st = time.time()

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
        for dir in self.dirs:
            print(f"\n{dir}: Start")
            self.per_dir_op(dir)

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
    proc = SubdirMultiprocessing(args.dirs, args.cores)
    proc.run()

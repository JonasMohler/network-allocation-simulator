"""Main runner for the computation of metrics."""
from argparse import ArgumentParser

from src.path_algorithms.gma_improved import GMAImproved
from src.path_algorithms.gma_original import GMAOriginal
from src.path_algorithms.sqos_algorithm import ScaledQoSAlgorithmOT, ScaledQoSAlgorithmPT, ScaledQoSAlgorithmPB, ScaledQoSAlgorithmOB
from src.multiprocessing.topology.PerTopologyOperations import *
from src.util.const import COVER_THRESHOLD

'''
from gma.path_algorithms.gma_improved import GMAImproved
from gma.path_algorithms.gma_original import GMAOriginal
from gma.path_algorithms.sqos_algorithm import ScaledQoSAlgorithm
from gma.path_algorithms.sqos_algorithm import ScaledQoSAlgorithm2
#from simulation.metrics_computation import *
from simulation.simulation_const import COVER_THRESHOLD

from simulation.PerTopologyOperations import *
'''

'''
def strategy_picker(strat_name):
    if strat_name == "original":
        strategy = GMAOriginal
    elif strat_name == "improved":
        strategy = GMAImproved
    elif strat_name == "sqs":
        strategy = ScaledQoSAlgorithm
    elif strat_name == "sqs2":
        strategy = ScaledQoSAlgorithm2
    else:
        raise ValueError("Strategy must be in {original, improved, sqs, sqs2}")
    return strategy


def strategy_namer(strat_name):
    if strat_name == "original":
        name = 'GMAOriginal'
    elif strat_name == "improved":
        name = 'GMAImproved'
    elif strat_name == "sqs":
        name = 'ScaledQoSAlgorithm'
    elif strat_name == "sqs2":
        name = 'ScaledQoSAlgorithm2'
    else:
        raise ValueError("Strategy must be in {original, improved, sqs, sqs2}")
    return name
'''


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--directories",
        nargs="+",
        default=["Eenet(13)"],
        help="Directories."
    )
    parser.add_argument(
        "-s",
        "--strategies",
        nargs="+",
        default=["improved"],
        type=str,
        help="Strategy for the computation. {original, improved, sq1, sq2}.",
    )
    parser.add_argument(
        "-r",
        "--sampling_ratios",
        nargs="+",
        default=["0.1"],
        help="Ratios to sample topologies down to for M-Approach"
    )
    parser.add_argument(
        "--o",
        default="out",
        help="Directory for aggregates"
    )
    parser.add_argument(
        "--cores",
        default=10,
        type=int,
        help="Number of cores to be used in the computation.",
    )
    parser.add_argument(
        "--thrs",
        default=COVER_THRESHOLD,
        type=float,
        help="The threshold for cover and reach.",
    )
    parser.add_argument(
        "--fr",
        action="store_true",
        help="Force recomputation of all intermediate steps"
    )
    parser.add_argument(
        "--agg",
        action="store_true",
        help="Perform Aggregation Steps"
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Visualize Results"
    )

    args = parser.parse_args()
    return args


def main(args):

    dirs = args.directories
    ratios = args.sampling_ratios
    out = args.o
    cores = args.cores
    force = args.fr
    thresh = args.thrs

    strategies = ['GMAImproved', 'sqos_ot', 'sqos_ob', 'sqos_pt', 'sqos_pb']

    # Preprocessing:
    # - Create topologies from zoo / rand topologies, add capacities
    # TODO

    # - Compute Degrees, Diameters, shortest paths, traffic matrices, shortest path sampling, path lengths

    '''
    degs = DegreesComputation(dirs, cores, force)
    degs.run()

    #shortest_paths = ShortestPathsComputation(dirs, cores, force)
    #shortest_paths.run()
    
    shortest_paths = AllShortestPathsComputation(dirs, cores, force)
    shortest_paths.run()


    diameters = DiameterComputation(dirs, cores, force)
    diameters.run()


    tm = AllocationMatrixComputation(dirs, cores, force)
    tm.run()

    #for r in ratios:
    #    sample = PathSampling(dirs, cores, force, r)
    #    sample.run()
    #

    for r in ratios:
        sample = PathSampling2(dirs, cores, force, r)
        sample.run()
    
    count = PathCounting2(dirs, cores, force)
    count.run()

    for r in ratios:
        count = PathCounting2(dirs, cores, force, r)
        count.run()



    
    lengths = PathLengthComputation2(dirs, cores, force)
    lengths.run()

    
    gma = GMAAllocationComputation(dirs, cores, force)
    gma.run()

    
    s10 = SQoSPTComputation(dirs, cores, force)
    s10.run()

    s11 = SQoSPBComputation(dirs, cores, force)
    s11.run()

    for r in ratios:
        s20 = SQoSOTComputation(dirs, cores, force, r)
        s20.run()

    for r in ratios:
        s21 = SQoSOBComputation(dirs, cores, force, r)
        s21.run()

'''
    # Covers: TODO
    for s in strategies:
        if s == 'sqos_ob' or s == 'sqos_ot':
            for r in ratios:
                c = CoverageComputation(dirs, cores, s, force, thresh, r)
                c.run()
        else:
            c = CoverageComputation(dirs, cores, s, force, thresh)
            c.run()

    # Aggregations: TODO
    if args.agg:
        pass


    # Vizualizations: TODO
    if args.viz:
        pass

    '''
    '''

if __name__ == "__main__":
    args = parse_args()
    main(args)
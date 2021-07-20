"""Main runner for the computation of metrics."""
from argparse import ArgumentParser

from src.path_algorithms.gma_improved import GMAImproved
from src.path_algorithms.gma_original import GMAOriginal
from src.path_algorithms.sqos_algorithm import ScaledQoSAlgorithmOT, ScaledQoSAlgorithmPT, ScaledQoSAlgorithmPB, ScaledQoSAlgorithmOB
from src.multiprocessing.topology.PerTopologyOperations import *
from src.util.const import COVER_THRESHOLD, STRATEGIES


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--all",
        action="store_true",
        help="Perform all simulation computations"
    )
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
        default=[0.1],
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
        "--degs",
        action="store_true",
        help="Compute Degrees"
    )
    parser.add_argument(
        "--sp",
        action="store_true",
        help="Compute shortest paths"
    )
    parser.add_argument(
        "--ksp",
        action="store_true",
        help="Compute k-shortest paths"
    )
    parser.add_argument(
        "--n_ksp",
        default=1,
        help="Number of shortest paths to compute"
    )
    parser.add_argument(
        "--dias",
        action="store_true",
        help="Compute diameters"
    )
    parser.add_argument(
        "--ams",
        action="store_true",
        help="Compute allocation matrices"
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Compute path counts"
    )
    parser.add_argument(
        "--lens",
        action="store_true",
        help="Compute path lengths"
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Compute allocations"
    )
    parser.add_argument(
        "--cov",
        action="store_true",
        help="Compute covers"
    )
    parser.add_argument(
        "--samp",
        action="store_true",
        help="Compute samples"
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

    # - Compute Degrees, Diameters, shortest paths, traffic matrices, shortest path sampling, path lengths

    if args.degs or args.all:
        degs = DegreesComputation(dirs, cores, force)
        degs.run()

    #shortest_paths = ShortestPathsComputation(dirs, cores, force)
    #shortest_paths.run()

    if args.sp or args.all:
        shortest_paths = AllShortestPathsComputation(dirs, cores, force)
        shortest_paths.run()

    if (args.ksp or args.all) and not args.n_ksp == 1:
        k_shortest_paths = AllKShortestPathsComputation(dirs, cores, force, args.n_ksp)
        k_shortest_paths.run()

    if args.dias or args.all:
        diameters = DiameterComputation(dirs, cores, force)
        diameters.run()

    if args.ams or args.all:
        tm = AllocationMatrixComputation(dirs, cores, force)
        tm.run()


    #for r in ratios:
    #    sample = PathSampling(dirs, cores, force, r)
    #    sample.run()
    #

    if args.samp or args.all:
        if not args.n_ksp == 1:
            for r in ratios:
                sample = PathSampling2(dirs, cores, force, r, args.n_ksp)
                sample.run()

        else:
            for r in ratios:
                sample = PathSampling2(dirs, cores, force, r, None)
                sample.run()
    
    if args.count or args.all:
        if not args.n_ksp == 1:
            count = PathCounting2(dirs, cores, force, num_sp=args.n_ksp)
            count.run()
        else:
            count = PathCounting2(dirs, cores, force)
            count.run()

        for r in ratios:
            if not args.n_ksp == 1:
                count = PathCounting2(dirs, cores, force, ratio=r, num_sp=args.n_ksp)
                count.run()
            else:
                count = PathCounting2(dirs, cores, force, ratio=r)
                count.run()
    '''

    if args.count or args.all:
        count = PathCounting3(dirs, cores, force)
        count.run()

        for r in ratios:
            count = PathCounting3(dirs, cores, force, r)
            count.run()

    '''
    if args.lens or args.all:
        lengths = PathLengthComputation2(dirs, cores, force)
        lengths.run()

    if args.sim or args.all:

        if not args.n_ksp == 1:
            '''
            gma = GMAAllocationComputation(dirs, cores, force, args.n_ksp)
            gma.run()

            s10 = SQoSPTComputation(dirs, cores, force, num_sp=args.n_ksp)
            s10.run()
            
            s11 = SQoSPBComputation(dirs, cores, force, num_sp=args.n_ksp)
            s11.run()
            '''
            for r in ratios:
                s20 = SQoSOTComputation(dirs, cores, force, r, args.n_ksp)
                s20.run()

            for r in ratios:
                s21 = SQoSOBComputation(dirs, cores, force, r, args.n_ksp)
                s21.run()
        else:
            '''
            gma = GMAAllocationComputation(dirs, cores, force)
            gma.run()

            s10 = SQoSPTComputation(dirs, cores, force)
            s10.run()

            s11 = SQoSPBComputation(dirs, cores, force)
            s11.run()
            '''
            for r in ratios:
                s20 = SQoSOTComputation(dirs, cores, force, r)
                s20.run()

            for r in ratios:
                s21 = SQoSOBComputation(dirs, cores, force, r)
                s21.run()

    if args.cov or args.all:
        if not args.n_ksp == 1:
            for s in STRATEGIES:
                if s == 'sqos_ob' or s == 'sqos_ot':
                    for r in ratios:
                        c = CoverageComputation(dirs, cores, s, force, thresh, ratio=r, num_sp=args.n_ksp)
                        c.run()
                else:
                    c = CoverageComputation(dirs, cores, s, force, thresh, num_sp=args.n_ksp)
                    c.run()
        else:
            for s in STRATEGIES:
                if s == 'sqos_ob' or s == 'sqos_ot':
                    for r in ratios:
                        c = CoverageComputation(dirs, cores, s, force, thresh, ratio=r)
                        c.run()
                else:
                    c = CoverageComputation(dirs, cores, s, force, thresh)
                    c.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)

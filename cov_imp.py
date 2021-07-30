"""Main runner for the computation of metrics."""
from argparse import ArgumentParser

from src.path_algorithms.gma_improved import GMAImproved
from src.path_algorithms.gma_original import GMAOriginal
from src.path_algorithms.sqos_algorithm import ScaledQoSAlgorithmOT, ScaledQoSAlgorithmPT, ScaledQoSAlgorithmPB, \
    ScaledQoSAlgorithmOB
from src.multiprocessing.topology.PerTopologyOperations import *
from src.util.const import COVER_THRESHOLD, STRATEGIES


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
        "--cores",
        default=10,
        type=int,
        help="Number of cores to be used in the computation.",
    )
    parser.add_argument(
        "--thrs",
        default=[COVER_THRESHOLD],
        nargs="+",
        help="The thresholds for cover and reach.",
    )
    parser.add_argument(
        "--fr",
        action="store_true",
        help="Force recomputation of all intermediate steps"
    )
    parser.add_argument(
        "--n_ksps",
        nargs="+",
        default=[2],
        help="Number of shortest paths to compare to 1"
    )
    parser.add_argument(
        "--out",
        default='dat/'
    )

    args = parser.parse_args()
    return args


def main(args):
    dirs = args.directories
    cores = args.cores
    force = args.fr
    threshs = args.thrs
    num_sps = args.n_ksps

    for s in ['GMAImproved', 'sqos_pt', 'sqos_pb']:
        for nsp in num_sps:
            for t in threshs:
                proc = CoverImprovement(dirs, cores, s, t, nsp, force=force)
                proc.run()

    cover_imps = []

    for dir in dirs:
        try:
            dia = dh.get_diameter(dir)
            size = dh.sfname(dir)

            for s in ['GMAImproved', 'sqos_pt', 'sqos_pb']:
                for nsp in num_sps:
                    for t in threshs:
                        cov_imp = dh.get_cover_imp_dist(dir, s, t, nsp)
                        row = [dir, dia, size, s, t, nsp]+cov_imp
                        cover_imps.append(row)
        except Exception as e:
            print(e)

    df_cover = pd.DataFrame(columns=['Graph', 'Diameter', 'Size', 'Strategy', 'Cover Threshold', 'Num Shortest Paths',  'Min-Cover', 'Q1-Cover', 'Mean-Cover', 'Median-Cover', 'Q3-Cover', 'Max-Cover'], data=cover_imps)
    with open(os.path.join(args.out, 'agg_cov_imp.csv'), "w+") as f:
        df_cover.to_csv(f)

if __name__ == "__main__":
    args = parse_args()
    main(args)

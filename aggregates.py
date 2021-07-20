import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from src.multiprocessing.topology.PerTopologyOperations import *
import pandas as pd

from src.util.utility import *
from src.util.const import *
import src.util.data_handler as dh
#import src.util.plot_handler as ph
#import seaborn as sns


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dirs",
        nargs="+",
        default=["Eenet(13)"],
        help="Directories."
    )
    parser.add_argument(
        "--r",
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Sampling ratios",
    )
    parser.add_argument(
        "--num_sp",
        nargs="+",
        default=[1],
        help="Number of shortest paths in multipathing",
    )
    parser.add_argument(
        "--out",
        default='dat/'
    )

    args = parser.parse_args()
    return args


def main(args):
    graphs = args.dirs

    allocs = []
    covers = []

    for g in graphs:
        # For all strategies
        # get diameter, size, rand flag
        diameter = dh.get_diameter(g)
        size = dh.sfname(g)

        deg = dh.get_degrees(g)
        degrees = deg['degrees']
        ave_deg = sum(degrees)/len(degrees)

        for s in STRATEGIES:
            # TODO: For all multipaths
            for mp in args.num_sp:
                # For all relevant ratios
                if s in ['sqos_ot', 'sqos_ob']:

                    for r in args.r:
                        # get min, q1, mean, median, q3, max allocation
                        try:
                            if int(mp) == 1:
                                a_dist = dh.get_alloc_dist(g, s, r)
                            else:
                                a_dist = dh.get_alloc_dist(g, s, r, mp)

                            a_row = [g, diameter, size, ave_deg, s, r, mp]+a_dist
                            allocs.append(a_row)
                        except Exception as e:
                            print(e)

                        # for all cover threshs:
                        for t in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                            # get min, q1, mean, median, q3, max cover
                            try:
                                if int(mp) == 1:
                                    c_dist = dh.get_cover_dist(g, s, t, r)
                                else:
                                    c_dist = dh.get_cover_dist(g, s, t, r, mp)

                                c_row = [g, diameter, size, ave_deg, s, r, t, mp]+c_dist
                                covers.append(c_row)
                            except Exception as e:
                                print(e)
                else:
                    try:
                        if int(mp) == 1:
                            a_dist = dh.get_alloc_dist(g, s)
                        else:
                            a_dist = dh.get_alloc_dist(g, s, num_sp=mp)

                        a_row = [g, diameter, size, ave_deg, s, 1, mp]+a_dist
                        allocs.append(a_row)
                    except Exception as e:
                        print(e)

                    for t in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                        # get min, q1, mean, median, q3, max cover
                        try:
                            if int(mp) == 1:
                                c_dist = dh.get_cover_dist(g, s, t)
                            else:
                                c_dist = dh.get_cover_dist(g, s, t, num_sp=mp)

                            c_row = [g, diameter, size, ave_deg, s, 1, t, mp] + c_dist
                            covers.append(c_row)
                        except Exception as e:
                            print(e)

    df_alloc = pd.DataFrame(columns=['Graph', 'Diameter', 'Size', 'Average Node Degree', 'Strategy', 'Ratio', 'Num Shortest Paths', 'Min-Allocation', 'Q1-Allocation', 'Mean-Allocation', 'Median-Allocation', 'Q3-Allocation', 'Max-Allocation'], data=allocs)
    df_cover = pd.DataFrame(columns=['Graph', 'Diameter', 'Size', 'Average Node Degree', 'Strategy', 'Ratio', 'Cover Threshold', 'Num Shortest Paths',  'Min-Cover', 'Q1-Cover', 'Mean-Cover', 'Median-Cover', 'Q3-Cover', 'Max-Cover'], data=covers)

    with open(os.path.join(args.out, 'aggregate_allocations.csv'), "w+") as f:
        df_alloc.to_csv(f)

    with open(os.path.join(args.out, 'aggregate_covers.csv'), "w+") as f:
        df_cover.to_csv(f)


if __name__ == "__main__":
    args = parse_args()
    main(args)

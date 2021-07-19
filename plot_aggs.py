import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from src.multiprocessing.topology.PerTopologyOperations import *
import pandas as pd

from src.util.utility import *
from src.util.const import *
import src.util.data_handler as dh
import src.util.plot_handler as ph
import seaborn as sns


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
        "--which",
        default='dat'
    )

    args = parser.parse_args()
    return args


def main(args):
    graphs = args.dirs

    # Cover plots
    '''
    with open(os.path.join(args.which, 'aggregate_covers.csv'), "r+") as f:
        df_cover_all = pd.read_csv(f)

    # Comparison Over multipath
    # Only Data where sampling strategy is degree weighted prob, cover threshold is 1Mbps
    df_cover = df_cover_all[(df_cover_all['Ratio'] == 1) | (df_cover_all['Ratio'] == '1') | (df_cover_all['Ratio'] == '0.5') | (df_cover_all['Ratio'] == 0.5)]
    df_cover = df_cover[~df_cover.Graph.str.startswith('c_')]

    df_cover["Strategy"].replace(STRATEGY_LABEL, inplace=True)
    df_cover["Cover Threshold"].replace(THRESH_LABEL, inplace=True)

    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover)#, ax=ax, palette=C_MAP)
    plt.show()

    sns.boxplot(x='Cover Threshold', y='Mean-Cover', hue='Strategy', data=df_cover)  # , ax=ax, palette=C_MAP)
    plt.show()

    sns.boxplot(x='Cover Threshold', y='Min-Cover', hue='Strategy', data=df_cover)  # , ax=ax, palette=C_MAP)
    plt.show()

    df_cover = df_cover[df_cover['Cover Threshold'] == '1Mbps']

    print(df_cover)

    for strat in [STRATEGY_LABEL['GMAImproved'], STRATEGY_LABEL['sqos_ot'], STRATEGY_LABEL['sqos_pt']]:
        df_tmp = df_cover[(df_cover['Strategy'] == strat)]
        print(df_tmp)
        sns.lmplot(x='Size', y='Median-Cover', hue='Num Shortest Paths', data=df_tmp)#, ax=axs[i])
        plt.title(strat)
        plt.ylim(0, 1)
        plt.show()



    # TODO: Test
    # Comparison over various metrics
    # Only Data with degree weighted prob + only 1 shortest path + only threshold 1Mbps
    df_cover = df_cover[df_cover['Num Shortest Paths'] == 1]

    for xs in ['Size', 'Average Node Degree', 'Diameter']:
        for strat in [STRATEGY_LABEL['GMAImproved'], STRATEGY_LABEL['sqos_ot'], STRATEGY_LABEL['sqos_pt']]:
            df_tmp = pd.melt(df_cover[[xs, 'Min-Cover', 'Max-Cover', 'Median-Cover']][df_cover['Strategy'] == strat], id_vars=[xs], value_name='Cover', value_vars=['Min-Cover', 'Max-Cover', 'Median-Cover'], var_name='Metric')
            print(df_tmp)
            sns.lmplot(x=xs, y='Cover', data=df_tmp, hue='Metric')
            plt.title(f"{strat} Covers over {xs}")
            plt.ylim(0,1)
            plt.show()

    '''
    # Allocation Plots

    with open(os.path.join(args.which, 'aggregate_allocations.csv'), "r+") as f:
        df_alloc_all = pd.read_csv(f)

    '''
    # Comparison of different sampling strategies, only 1 shortest path
    df_alloc = df_alloc_all[df_alloc_all['Num Shortest Paths'] == 1]

    # TODO: Test
    dft = df_alloc[['Median-Allocation', 'Ratio']][(df_alloc['Ratio'] == '0.5') | (df_alloc['Ratio'] == 0.5) | (df_alloc['Ratio'] == 'u0.5') | (df_alloc['Ratio'] == 'a0.5')]
    r_map = {0.5: 'Degree weighted pick probability', '0.5': 'Degree weighted pick probability', 'u0.5': 'Uniform random sampling', 'a0.5': 'Degree weighted number of destinations'}
    dft['Ratio'].replace(r_map, inplace=True)
    sns.boxplot(y='Median-Allocation', hue='Ratio', data=dft)
    plt.show()
    '''
    df_alloc_1sp = df_alloc_all[df_alloc_all['Num Shortest Paths'] == 1]
    df_alloc_const = df_alloc_1sp[df_alloc_1sp.Graph.str.startswith('c_')]
    df_alloc_const_rand = df_alloc_const[df_alloc_const.Graph.str.startswith('c_Barabasi')]
    df_alloc_const_zoo = df_alloc_const[~df_alloc_const.Graph.str.startswith('c_Barabasi')]

    df_alloc_dw = df_alloc_1sp[~df_alloc_1sp.Graph.str.startswith('c_')]
    df_alloc_dw_rand = df_alloc_dw[df_alloc_dw.Graph.str.startswith('Barabasi')]
    df_alloc_dw_zoo = df_alloc_dw[~df_alloc_dw.Graph.str.startswith('Barabasi')]

    # compare const & dw in rand:
    fig, axs = plt.subplots(1, 2, sharey=True)

    df_alloc_dw_zoo['Capacity Model'] = 'Degree-weighted'
    df_alloc_const_zoo['Capacity Model'] = 'Constant'
    df_alloc_zoo = pd.concat([df_alloc_dw_zoo, df_alloc_const_zoo])

    sns.scatterplot(x='Size',y='Median-Allocation', hue='Capacity Model', data=df_alloc_zoo, alpha=0.4, style='Capacity Model', ax=axs[0])




    # compare const & dw in zoo:
    #print(df_alloc_dw_rand)
    #print(df_alloc_const_rand)
    df_alloc_dw_rand['Capacity Model'] = 'Degree-weighted'
    df_alloc_const_rand['Capacity Model'] = 'Constant'
    df_alloc_rand = pd.concat([df_alloc_const_rand, df_alloc_dw_rand])
    sns.scatterplot(x='Size', y='Median-Allocation', hue='Capacity Model', data=df_alloc_rand, alpha=0.4, style='Capacity Model', ax=axs[1])

    plt.yscale('log')
    plt.show()




if __name__ == "__main__":
    args = parse_args()
    main(args)

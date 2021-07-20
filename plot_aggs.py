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

    ####################################################################################################################
    '''
    Median Allocations - Constant Link Model vs. Degree-weighted link model, in Zoo and Rand graphs, 0.5 sampling for M-Approach
    '''

    df_alloc_1sp = df_alloc_all[df_alloc_all['Num Shortest Paths'] == 1]
    df_alloc_const = df_alloc_1sp[df_alloc_1sp.Graph.str.startswith('c_')]
    df_alloc_const_rand = df_alloc_const[df_alloc_const.Graph.str.startswith('c_Barabasi')]
    df_alloc_const_zoo = df_alloc_const[~df_alloc_const.Graph.str.startswith('c_Barabasi')]

    df_alloc_dw = df_alloc_1sp[~df_alloc_1sp.Graph.str.startswith('c_')]
    df_alloc_dw_rand = df_alloc_dw[df_alloc_dw.Graph.str.startswith('Barabasi')]
    df_alloc_dw_zoo = df_alloc_dw[~df_alloc_dw.Graph.str.startswith('Barabasi')]

    X_AXIS = 'Average Node Degree'

    # compare const & dw in rand:
    #fig, axs = plt.subplots(2, 2, sharey=True)

    df_alloc_dw_zoo['Capacity Model'] = 'Degree-Weighted'
    df_alloc_const_zoo['Capacity Model'] = 'Constant'
    df_alloc_zoo = pd.concat([df_alloc_dw_zoo, df_alloc_const_zoo])

    df_alloc_dw_rand['Capacity Model'] = 'Degree-Weighted'
    df_alloc_const_rand['Capacity Model'] = 'Constant'
    df_alloc_rand = pd.concat([df_alloc_const_rand, df_alloc_dw_rand])

    '''

    sns.scatterplot(x=X_AXIS,y='Median-Allocation', hue='Capacity Model', data=df_alloc_zoo[df_alloc_zoo['Strategy'] == 'GMAImproved'], alpha=0.4, style='Capacity Model', ax=axs[0][0])
    axs[0][0].set_title('GMA in Zoo Graphs')

    # compare const & dw in zoo:

    sns.scatterplot(x=X_AXIS, y='Median-Allocation', hue='Capacity Model', data=df_alloc_rand[df_alloc_rand['Strategy'] == 'GMAImproved'], alpha=0.4, style='Capacity Model', ax=axs[0][1])
    axs[0][1].set_title('GMA in Random Graphs')

    sns.scatterplot(x=X_AXIS, y='Median-Allocation', hue='Capacity Model', data=df_alloc_zoo[(df_alloc_zoo['Strategy'] == 'sqos_ot') & (df_alloc_zoo['Ratio'] == '0.5')], alpha=0.4, style='Capacity Model', ax=axs[1][0])
    axs[1][0].set_title('M-Approach in Zoo Graphs')

    sns.scatterplot(x=X_AXIS, y='Median-Allocation', hue='Capacity Model', data=df_alloc_rand[(df_alloc_rand['Strategy'] == 'sqos_ot') & (df_alloc_rand['Ratio'] == '0.5')], alpha=0.4, style='Capacity Model', ax=axs[1][1])
    axs[1][1].set_title('M-Approach in Random Graphs')

    plt.yscale('log')
    fig.suptitle('Influence of Link Model on Allocations')
    #plt.show()
    '''
    ####################################################################################################################
    '''
    As above in boxplots
    '''
    fig, axs = plt.subplots(1, 2, sharey=True)
    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Capacity Model', data=df_alloc_zoo, ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Capacity Model', data=df_alloc_rand, ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    plt.yscale('log')
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.75, -0.98))
    #plt.legend(title='Link Capacity Models', labels=['Constant', 'Degree-Weighted'], bbox_to_anchor=(1, 1.23), fancybox=True, shadow=True, ncol=5, loc='upper right')
    plt.tight_layout()
    fig.suptitle('Influence of Link-Capacity Model on Allocations')
    plt.show()

    ####################################################################################################################
    '''
    Median Allocations for sqos_ot sampling models - constant link model, 0.5 sampling 
    '''
    '''
    fig, axs = plt.subplots(1,2, sharey=True)

    df_samp_comp_zoo = df_alloc_const_zoo[(df_alloc_const_zoo['Strategy'] == 'sqos_ot') & ((df_alloc_const_zoo['Ratio'] == '0.5') | (df_alloc_const_zoo['Ratio'] == 'u0.5') | (df_alloc_const_zoo['Ratio'] == 'a0.5'))]
    sns.scatterplot(x='Size', y='Median-Allocation', hue='Ratio', data=df_samp_comp_zoo, alpha=0.4, style='Ratio', ax=axs[0])

    df_samp_comp_rand = df_alloc_const_rand[(df_alloc_const_rand['Strategy'] == 'sqos_ot') & ((df_alloc_const_rand['Ratio'] == '0.5') | (df_alloc_const_rand['Ratio'] == 'u0.5') | (df_alloc_const_rand['Ratio'] == 'a0.5'))]
    sns.scatterplot(x='Size', y='Median-Allocation', hue='Ratio', data=df_samp_comp_rand, alpha=0.4,style='Ratio', ax=axs[1])

    plt.yscale('log')
    plt.title('Influence of Sampling Model on M-Approach Allocations')
    #plt.show()
    '''
    ####################################################################################################################
    '''
    As above in boxplot
    '''
    df_samp_comp_zoo = df_alloc_const_zoo[(df_alloc_const_zoo['Strategy'] == 'sqos_ot') & ((df_alloc_const_zoo['Ratio'] == '0.5') | (df_alloc_const_zoo['Ratio'] == 'u0.5') | (df_alloc_const_zoo['Ratio'] == 'a0.5'))]
    df_samp_comp_rand = df_alloc_const_rand[(df_alloc_const_rand['Strategy'] == 'sqos_ot') & ((df_alloc_const_rand['Ratio'] == '0.5') | (df_alloc_const_rand['Ratio'] == 'u0.5') | (df_alloc_const_rand['Ratio'] == 'a0.5'))]

    fig, axs = plt.subplots(1,2, sharey=True)
    sns.boxplot(x='Ratio', y='Median-Allocation', hue='Ratio', data=df_samp_comp_zoo, ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set(xticklabels=[])
    axs[0].grid(b=True)

    sns.boxplot(x='Ratio', y='Median-Allocation', hue='Ratio', data=df_samp_comp_rand, ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set(xticklabels=[])
    axs[1].grid(b=True)

    plt.yscale('log')
    plt.legend(title='Ratios', labels=['0.5', 'u0.5'],bbox_to_anchor=(1, 1.23), fancybox=True, shadow=True, ncol=5, loc='upper right')
    plt.tight_layout()
    fig.suptitle('Influence of Sampling Strategy on M-Approach Allocations')
    plt.show()

    ####################################################################################################################
    # TODO: Adapt for appropriate graph
    '''
    Comparison of Allocations GMA vs M-Approach flavours, const capacity model, classic link model, ratio 0.5
    - Once one plot with median allocations (aggregated)
    - Once one plot with cdf of allocation ratios (one graph, no aggregates)
    '''
    '''
    fig, axs = plt.subplots(1, 2, sharey=True)

    df = df_alloc_const_zoo[(df_alloc_const_zoo['Strategy'] == 'GMAImproved') | (df_alloc_const_zoo['Strategy'] == 'sqos_pb') | (df_alloc_const_zoo['Strategy'] == 'sqos_pt') | (df_alloc_const_zoo['Ratio'] == '0.5')]
    sns.scatterplot(x='Size', y='Median-Allocation', hue='Strategy', data=df, alpha=0.4, style='Strategy', ax=axs[0])

    df = df_alloc_const_rand[(df_alloc_const_rand['Strategy'] == 'GMAImproved') | (df_alloc_const_rand['Strategy'] == 'sqos_pb') | (df_alloc_const_rand['Strategy'] == 'sqos_pt') | (df_alloc_const_rand['Ratio'] == '0.5')]
    sns.scatterplot(x='Size', y='Median-Allocation', hue='Strategy', data=df, alpha=0.4, style='Strategy', ax=axs[1])

    plt.yscale('log')
    plt.show()
    '''
    ####################################################################################################################
    '''
    As above in boxplot
    '''
    fig, axs = plt.subplots(1, 2, sharey=True)

    df = df_alloc_const_zoo[(df_alloc_const_zoo['Strategy'] == 'GMAImproved') | (df_alloc_const_zoo['Strategy'] == 'sqos_pb') | (df_alloc_const_zoo['Strategy'] == 'sqos_pt') | (df_alloc_const_zoo['Ratio'] == '0.5')]
    print(df.Strategy.unique())
    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Strategy', data=df, ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set(xticklabels=[])
    axs[0].grid(b=True)

    df = df_alloc_const_rand[(df_alloc_const_rand['Strategy'] == 'GMAImproved') | (df_alloc_const_rand['Strategy'] == 'sqos_pb') | (df_alloc_const_rand['Strategy'] == 'sqos_pt') | (df_alloc_const_rand['Ratio'] == '0.5')]
    print(df.Strategy.unique())
    sns.boxplot(x='Strategy',y='Median-Allocation', hue='Strategy', data=df, ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set(xticklabels=[])
    axs[1].grid(b=True)
    fig.suptitle('Median Allocations across Strategies')
    plt.yscale('log')
    plt.show()

    ####################################################################################################################

    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
    # Zoo: comparison in colt(153)
    data = dh.get_alloc_quots_as_df(['c_Barabasi_Albert_20_30_(1000)'], 'GMAImproved', ['sqos_ob', 'sqos_ot', 'sqos_pt', 'sqos_pb'], ratios=['0.1'])

    d_min = data['Allocation Ratio'].min()
    d_max = data['Allocation Ratio'].max()
    med = data['Allocation Ratio'].median()
    mean = data['Allocation Ratio'].mean()

    sns.ecdfplot(data=data, x='Allocation Ratio', hue="Strategies", ax=axs[0])#, palette=DIFF_C_MAP)

    axs[0].set_xlim(d_min, d_max)
    axs[0].grid(b=True)
    axs[0].axvline(x=1, c='r', linestyle='--', alpha=0.3, label='One')
    fig.suptitle('Comparison of GMA vs M-Approach Flavour Allocations in Barabasi-Albert 20 30 1000')
    plt.xscale('log')
    plt.show()

    ####################################################################################################################
    '''
    Numbers version of Allocation comparison
    '''
    # GMA vs. sqos_ob
    ob = data[data['Strategies'] == f"GMA vs. {STRATEGY_LABEL['sqos_ob']}"]
    ot = data[data['Strategies'] == f"GMA vs. {STRATEGY_LABEL['sqos_ot']}"]
    pb = data[data['Strategies'] == f"GMA vs. {STRATEGY_LABEL['sqos_pb']}"]
    pt = data[data['Strategies'] == f"GMA vs. {STRATEGY_LABEL['sqos_pt']}"]

    obs = ob['Allocation Ratio'].quantile([0.1, 0.5, 0.9])
    ots = ot['Allocation Ratio'].quantile([0.1, 0.5, 0.9])
    pbs = pb['Allocation Ratio'].quantile([0.1, 0.5, 0.9])
    pts = pt['Allocation Ratio'].quantile([0.1, 0.5, 0.9])
    print(obs)

    ####################################################################################################################

    '''
    Multipath improvements: 
    - Scatter of Aggregates for zoo and rand
    - for GMA and sqos_pb in const graphs, cdf and table of allocation ratios
    '''
    df_alloc_const_zoo = df_alloc_all[df_alloc_all.Graph.str.startswith('c_') & (~df_alloc_all.Graph.str.startswith('c_Barabasi'))]
    df_alloc_const_rand = df_alloc_all[df_alloc_all.Graph.str.startswith('c_Barabasi')]

    df_alloc_const_zoo_gma = df_alloc_const_zoo[df_alloc_const_zoo['Strategy'] == 'GMAImproved']
    df_alloc_const_rand_gma = df_alloc_const_rand[df_alloc_const_rand['Strategy'] == 'GMAImproved']

    df_alloc_const_zoo_pb = df_alloc_const_zoo[df_alloc_const_zoo['Strategy'] == 'sqos_pb']
    df_alloc_const_rand_pb = df_alloc_const_rand[df_alloc_const_rand['Strategy'] == 'sqos_pb']

    fig, axs = plt.subplots(2, 2, sharey=True)
    sns.scatterplot(x='Size', y='Median-Allocation', hue='Num Shortest Paths', data=df_alloc_const_zoo_gma, ax=axs[0][0])
    axs[0][0].set_title('GMA in Zoo Graphs')
    axs[0][0].grid(b=True)
    sns.scatterplot(x='Size', y='Median-Allocation', hue='Num Shortest Paths', data=df_alloc_const_rand_gma, ax=axs[0][1])
    axs[0][1].set_title('GMA in Rand Graphs')
    axs[0][1].grid(b=True)

    sns.scatterplot(x='Size', y='Median-Allocation', hue='Num Shortest Paths', data=df_alloc_const_zoo_pb,
                    ax=axs[1][0])
    axs[1][0].set_title('M-Approach (pb) in Zoo Graphs')
    axs[1][0].grid(b=True)

    sns.scatterplot(x='Size', y='Median-Allocation', hue='Num Shortest Paths', data=df_alloc_const_rand_pb,
                    ax=axs[1][1])
    axs[1][1].set_title('M-Approach (pb) in Rand Graphs')
    axs[1][1].grid(b=True)
    fig.suptitle('Multipath improvements across Graphs')

    plt.yscale('log')
    plt.show()



    data = dh.get_alloc_quots_multipath_as_df('c_Barabasi_Albert_20_30_(1000)', ['GMAImproved', 'sqos_pt'])
    fig, ax = plt.subplots(1,1)
    sns.ecdfplot(data=data, x='Allocation Ratio', hue='Strategy', ax=ax)
    plt.xscale('log')
    fig.suptitle('Multipath improvements for GMA and M-Approach PT Allocations in Barabasi-Albert 20_30_(1000)')
    plt.show()

    gma = data[data['Strategy'] == STRATEGY_LABEL['GMAImproved']]
    pt = data[data['Strategy'] == STRATEGY_LABEL['sqos_pt']]

    gmas = gma['Allocation Ratio'].quantile([0.1, 0.5, 0.9])
    pts = pt['Allocation Ratio'].quantile([0.1, 0.5, 0.9])

    print(gmas)
    print(pts)

    ####################################################################################################################
    '''
    Cover plots: Median Covers by thresholds, degweighted, 1sp
    '''
    with open(os.path.join(args.which, 'aggregate_covers.csv'), "r+") as f:
        df_cover_all = pd.read_csv(f)

    df_cover = df_cover_all[~df_cover_all.Graph.str.startswith('c_')]
    df_cover["Cover Threshold"].replace(THRESH_LABEL, inplace=True)
    df_cover = df_cover[df_cover['Num Shortest Paths'] == 1]
    df_cover = df_cover[(df_cover['Strategy'] == 'GMAImproved') | (df_cover['Strategy'] == 'sqos_pb') | (df_cover['Strategy'] == 'sqos_pt') | (df_cover['Ratio'] == '0.5')]

    df_cover_zoo = df_cover[~df_cover.Graph.str.startswith('c_Barabasi')]
    df_cover_rand = df_cover[df_cover.Graph.str.startswith('c_Barabasi')]

    fig, axs = plt.subplots(1, 2, sharey=True)
    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_zoo, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].grid(b=True)

    #sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_rand, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].grid(b=True)

    fig.suptitle('Strategies by Cover Thresholds Reached')

    plt.show()

    df_cover_const = df_cover_all[~df_cover_all.Graph.str.startswith('c_')]
    df_cover_const["Cover Threshold"].replace(THRESH_LABEL, inplace=True)

    df_cover_zoo = df_cover_const[~df_cover_const.Graph.str.startswith('c_Barabasi')]
    df_cover_rand = df_cover_const[df_cover_const.Graph.str.startswith('c_Barabasi')]

    fig, axs = plt.subplots(1, 2, sharey=True)

    # GMA
    sns.boxplot(y='Median-Cover', hue='Num Shortest Paths', data=df_cover_zoo[df_cover_zoo['Strategy'] == 'GMAImproved'], ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].grid(b=True)
    sns.boxplot(y='Median-Cover', hue='Num Shortest Paths', data=df_cover_rand[df_cover_rand['Strategy'] == 'GMAImproved'], ax=axs[1])
    axs[0].set_title('Rand Graphs')
    axs[0].grid(b=True)
    plt.show()




if __name__ == "__main__":
    args = parse_args()
    main(args)

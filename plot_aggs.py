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

    df_alloc_all["Strategy"].replace(STRATEGY_LABEL, inplace=True)

    df_alloc_1sp = df_alloc_all[df_alloc_all['Num Shortest Paths'] == 1]
    df_alloc_const = df_alloc_1sp[df_alloc_1sp.Graph.str.startswith('c_')]
    df_alloc_const_rand = df_alloc_const[df_alloc_const.Graph.str.startswith('c_Barabasi')]
    df_alloc_const_zoo = df_alloc_const[~df_alloc_const.Graph.str.startswith('c_Barabasi')]

    df_alloc_dw = df_alloc_1sp[~df_alloc_1sp.Graph.str.startswith('c_')]
    df_alloc_dw_rand = df_alloc_dw[df_alloc_dw.Graph.str.startswith('Barabasi')]
    df_alloc_dw_zoo = df_alloc_dw[~df_alloc_dw.Graph.str.startswith('Barabasi')]

    df_alloc_dw_zoo['Capacity Model'] = 'Degree-Weighted'
    df_alloc_const_zoo['Capacity Model'] = 'Constant'
    df_alloc_zoo = pd.concat([df_alloc_dw_zoo, df_alloc_const_zoo])

    df_alloc_dw_rand['Capacity Model'] = 'Degree-Weighted'
    df_alloc_const_rand['Capacity Model'] = 'Constant'
    df_alloc_rand = pd.concat([df_alloc_const_rand, df_alloc_dw_rand])

    sns.set(rc={"axes.facecolor":"gainsboro", "figure.facecolor": "whitesmoke"})
    #sns.set_palette((sns.color_palette('crest')))

    ####################################################################################################################
    '''
    Aggregates of allocations over size
    '''

    zoo = df_alloc_const_zoo[(df_alloc_const_zoo['Ratio'] == '0.5') | (df_alloc_const_zoo['Ratio'] == '1')]
    rand = df_alloc_const_rand[(df_alloc_const_rand['Ratio'] == '0.5') | (df_alloc_const_rand['Ratio'] == '1')]

    ####################################################################################################################

    xs = ['Diameter', 'Size', 'Average Node Degree']
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.scatterplot(x='Diameter', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.scatterplot(x='Diameter', y='Median-Allocation', hue='Strategy', data=rand, ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set_ylabel('')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    plt.yscale('log')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Median Allocations [Gbps] over Graph Diameter', fontsize='x-large')

    ####################################################################################################################
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.scatterplot(x='Size', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.scatterplot(x='Size', y='Median-Allocation', hue='Strategy', data=rand, ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set_ylabel('')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    plt.yscale('log')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Median Allocations [Gbps] over Graph Size (#Nodes)', fontsize='x-large')

    ####################################################################################################################
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.scatterplot(x='Average Node Degree', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.scatterplot(x='Average Node Degree', y='Median-Allocation', hue='Strategy', data=rand, ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set_ylabel('')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    plt.yscale('log')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Median Allocations [Gbps] over Graph Average Node Degree', fontsize='x-large')


    ####################################################################################################################
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    zoo['Average Node Degree / #Nodes'] = [row['Average Node Degree'] / (row['Size'] -1) for index, row in zoo.iterrows()]
    rand['Average Node Degree / #Nodes'] = [row['Average Node Degree'] /(row['Size'] -1) for index, row in rand.iterrows()]

    sns.scatterplot(x='Average Node Degree / #Nodes', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.scatterplot(x='Average Node Degree / #Nodes', y='Median-Allocation', hue='Strategy', data=rand, ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set_ylabel('')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    plt.yscale('log')
    plt.xscale('log')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Median Allocations [Gbps] over Graph Density', fontsize='x-large')

    plt.show()

    ####################################################################################################################
    '''
    CDF Allocations - Barabasi-Albert 20 30 1000
    '''
    dat = dh.get_allocs_as_df(['Barabasi_Albert_20_30_(1000)'], STRATEGIES, [0.5])
    dat['Allocations Gbps'] = dat['Allocations Gbps'].astype('float')

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(17, 9.27)

    sns.ecdfplot(data=dat, x='Allocations Gbps', hue='Strategy', ax=ax)

    ax.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    ax.set_ylabel('Percentiles')
    ax.set_xlabel('Allocations [Gbps]')

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Distribution of Allocations [Gbps]', fontsize='x-large')

    plt.xscale('log')
    plt.legend(labels=STRAT_ORDER, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    #plt.show()

    ####################################################################################################################
    '''
    Median Allocations - Constant Link Model vs. Degree-weighted link model, in Zoo and Rand graphs, 0.5 sampling for M-Approach
    '''

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Capacity Model', data=df_alloc_zoo, order=STRAT_ORDER, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Capacity Model', data=df_alloc_rand, order=STRAT_ORDER, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set_ylabel('')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    plt.yscale('log')
    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Influence of Link-Capacity Model on Allocations', fontsize='x-large')
    #fig.show(bbox_extra_artists=(lgd,))
    #plt.show()

    ####################################################################################################################
    '''
    Median Allocations for sqos_ot sampling models - constant link model, 0.5 sampling 
    '''
    r_map = {
        '0.5': 'Degree-weighted probability distribution',
        'u0.5': 'Uniform probability distribution',
        'a0.5': 'Degree-weighted #paths per node'

    }
    df_samp_comp_zoo = df_alloc_const_zoo[((df_alloc_const_zoo['Strategy'] == STRATEGY_LABEL['sqos_ot']) | (df_alloc_const_zoo['Strategy'] == STRATEGY_LABEL['sqos_ob'])) & ((df_alloc_const_zoo['Ratio'] == '0.5') | (df_alloc_const_zoo['Ratio'] == 'u0.5') | (df_alloc_const_zoo['Ratio'] == 'a0.5'))]
    df_samp_comp_zoo['Ratio'].replace(r_map, inplace=True)
    df_samp_comp_rand = df_alloc_const_rand[((df_alloc_const_rand['Strategy'] == STRATEGY_LABEL['sqos_ot']) | (df_alloc_const_rand['Strategy'] == STRATEGY_LABEL['sqos_ob']))& ((df_alloc_const_rand['Ratio'] == '0.5') | (df_alloc_const_rand['Ratio'] == 'u0.5') | (df_alloc_const_rand['Ratio'] == 'a0.5'))]
    df_samp_comp_rand['Ratio'].replace(r_map, inplace=True)

    fig, axs = plt.subplots(1,2, sharey=True)
    fig.set_size_inches(17, 9.27)
    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Ratio', data=df_samp_comp_zoo, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].get_legend().remove()
    #axs[0].set(xticklabels=[])
    axs[0].grid(b=True)

    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Ratio', data=df_samp_comp_rand, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Random Graphs')
    axs[1].set_ylabel('')
    axs[1].get_legend().remove()
    #axs[1].set(xticklabels=[])
    axs[1].grid(b=True)

    plt.yscale('log')
    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Influence of Sampling Strategy on M-Approach Allocations')
    #plt.show()

    ####################################################################################################################
    # TODO: Adapt for appropriate graph
    '''
    Comparison of Allocations GMA vs M-Approach flavours, const capacity model, classic sampling model, ratio 0.5
    - Once one plot with median allocations (aggregated)
    - Once one plot with cdf of allocation ratios (one graph, no aggregates)
    '''

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    df = df_alloc_const_zoo[(df_alloc_const_zoo['Strategy'] == STRATEGY_LABEL['GMAImproved']) | (df_alloc_const_zoo['Strategy'] == STRATEGY_LABEL['sqos_pb']) | (df_alloc_const_zoo['Strategy'] == STRATEGY_LABEL['sqos_pt']) | (df_alloc_const_zoo['Ratio'] == '0.5')]
    df = pd.melt(df, id_vars=['Strategy'], value_vars=['Min-Allocation', 'Median-Allocation', 'Max-Allocation'], value_name='Allocation Size [Gbps]', var_name='Metric')
    sns.boxplot(x='Metric', y='Allocation Size [Gbps]', hue='Strategy', data=df, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])

    axs[0].set_xticklabels(['Min','Median', 'Max'])
    axs[0].set_xlabel('')
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Allocation Size [Gbps]')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    df = df_alloc_const_rand[(df_alloc_const_rand['Strategy'] == STRATEGY_LABEL['GMAImproved']) | (df_alloc_const_rand['Strategy'] == STRATEGY_LABEL['sqos_pb']) | (df_alloc_const_rand['Strategy'] == STRATEGY_LABEL['sqos_pt']) | (df_alloc_const_rand['Ratio'] == '0.5')]
    df = pd.melt(df, id_vars=['Strategy'], value_vars=['Min-Allocation', 'Median-Allocation', 'Max-Allocation'], value_name='Allocation Size [Gbps]', var_name='Metric')
    sns.boxplot(x='Metric', y='Allocation Size [Gbps]', hue='Strategy', data=df, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])

    axs[1].set_xticklabels(['Min', 'Median', 'Max'])
    axs[1].set_xlabel('')
    axs[1].set_title('Random Graphs')
    axs[1].set_ylabel('')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=3, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Median Allocations across Strategies')
    plt.yscale('log')
    #plt.show()

    ####################################################################################################################

    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.set_size_inches(17, 9.27)
    # Zoo: comparison in colt(153)
    data = dh.get_alloc_quots_as_df(['c_Barabasi_Albert_20_30_(1000)'], 'GMAImproved', ['sqos_ob', 'sqos_ot', 'sqos_pt', 'sqos_pb'], ratios=['0.1'])

    d_min = data['Allocation Ratio'].min()
    d_max = data['Allocation Ratio'].max()
    med = data['Allocation Ratio'].median()
    mean = data['Allocation Ratio'].mean()

    axs.set_title('Barabasi-Albert 20 30 1000')
    axs.set_xlim(d_min, d_max)
    axs.grid(b=True)
    axs.axvline(x=1, c='r', linestyle='--', alpha=0.3)

    l = sns.ecdfplot(data=data, x='Allocation Ratio', hue="Strategies", ax=axs)#, palette=DIFF_C_MAP)

    handles, labels = axs.get_legend_handles_labels()
    lgd = axs.legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    axs.set_xlabel('Factor MA Flavor : GMA')
    axs.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    #fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    #fig.legend(labels, loc='upper center', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.suptitle('Comparison of Allocations: GMA vs MA Flavours ')

    plt.legend(labels=STRAT_ORDER, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.xscale('log')
    #plt.show()

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

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Num Shortest Paths', data=df_alloc_const_zoo, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].set_xlabel('')
    axs[0].grid(b=True)
    axs[0].get_legend().remove()

    sns.boxplot(x='Strategy', y='Median-Allocation', hue='Num Shortest Paths', data=df_alloc_const_rand, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].grid(b=True)
    axs[1].get_legend().remove()

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5), title='#Shortest Paths')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Multipath improvements across Graphs')
    plt.yscale('log')
    #plt.show()

    ####################################################################################################################
    '''
    Allocation multipath gain: cdf single graph
    '''

    data = dh.get_alloc_quots_multipath_as_df('c_Barabasi_Albert_20_30_(1000)', ['GMAImproved', 'sqos_pt'])
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(17, 9.27)

    sns.ecdfplot(data=data, x='Allocation Ratio', hue='Strategy', ax=ax)

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)

    plt.xscale('log')
    fig.suptitle('Multipath improvements for GMA and M-Approach PT Allocations in Barabasi-Albert 20_30_(1000)')
    #plt.show()

    ####################################################################################################################

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
    df_cover["Strategy"].replace(STRATEGY_LABEL, inplace=True)
    print(df_cover.Strategy.unique())
    df_cover = df_cover[(df_cover['Strategy'] == STRATEGY_LABEL['GMAImproved']) | (df_cover['Strategy'] == STRATEGY_LABEL['sqos_pb']) | (df_cover['Strategy'] == STRATEGY_LABEL['sqos_pt']) | (df_cover['Ratio'] == '0.5')]

    print(df_cover)

    df_cover_zoo = df_cover[~df_cover.Graph.str.startswith('Barabasi')]
    print(df_cover_zoo)
    df_cover_rand = df_cover[df_cover.Graph.str.startswith('Barabasi')]
    print(df_cover_rand)

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_zoo, hue_order=STRAT_ORDER, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_rand, hue_order=STRAT_ORDER, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].grid(b=True)
    axs[1].get_legend().remove()

    handles, labels = axs[1].get_legend_handles_labels()
    axs[0].set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Strategies by Cover Thresholds Reached')
    #plt.show()

    ####################################################################################################################
    '''
    Cover multipath gain
    '''
    df_cover_const = df_cover_all[~df_cover_all.Graph.str.startswith('c_')]
    df_cover_const["Cover Threshold"].replace(THRESH_LABEL, inplace=True)

    df_cover_zoo = df_cover_const[~df_cover_const.Graph.str.startswith('Barabasi')]
    df_cover_zoo = df_cover_zoo[df_cover_zoo['Cover Threshold'] == THRESH_LABEL[1e-5]]
    df_cover_rand = df_cover_const[df_cover_const.Graph.str.startswith('Barabasi')]
    df_cover_rand = df_cover_rand[df_cover_rand['Cover Threshold'] == THRESH_LABEL[1e-5]]

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    # GMA
    sns.boxplot(x='Strategy', y='Median-Cover', hue='Num Shortest Paths', data=df_cover_zoo, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].grid(b=True)
    axs[0].get_legend().remove()

    sns.boxplot(x='Strategy', y='Median-Cover', hue='Num Shortest Paths', data=df_cover_rand, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].grid(b=True)
    axs[1].get_legend().remove()

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    fig.suptitle('Multipath improvements in Median Covers for 10 Kbps Threshold')
    plt.show()




if __name__ == "__main__":
    args = parse_args()
    main(args)

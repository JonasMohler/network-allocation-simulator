import os
from argparse import ArgumentParser
import matplotlib
#matplotlib.use('Agg')
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

    plt.rcParams.update({
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large'
                         })

    plt.rc('legend', fontsize='large')

    ####################################################################################################################
    '''
    OMA Ratios vs GMA
    '''
    # TODO
    comp = dh.get_alloc_quots_as_df(['Barabasi_Albert_20_30_(1000)'], 'GMAImproved', ['sqos_ot'], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    fig, axs = plt.subplots(1, 1)
    fig.set_size_inches(17, 9.27)

    sns.ecdfplot(comp, x='Allocation Ratio', hue='Ratio', ax=axs)
    plt.xscale('log')
    plt.show()

    ####################################################################################################################
    '''
    Cover improvement multipath in single graph
    '''
    # TODO: Server
    '''
    cov_imp = dh.get_cover_mp_improv('Colt(153)', [1,2,3,5], 'sqos_pt', '1e-4')

    sns.ecdfplot(data=cov_imp, x='improvement', hue='num_sp')
    plt.show()
    '''
    ####################################################################################################################
    '''
    Allocs over ave node degree
    '''
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    zoo = df_alloc_const_zoo[(df_alloc_const_zoo['Ratio'] == '0.5') | (df_alloc_const_zoo['Ratio'] == '1')]
    rand = df_alloc_const_rand[(df_alloc_const_rand['Ratio'] == '0.5') | (df_alloc_const_rand['Ratio'] == '1')]

    sns.scatterplot(x='Average Node Degree', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[0])
    axs[0].get_legend().remove()
    axs[0].set_title('Zoo Graphs')#, fontsize=12)
    #axs[0].set_xlabel('Average Node Degree', fontsize=12)
    #axs[0].set_ylabel('Median Allocation [Gbps]', fontsize=12)
    axs[0].set_yscale('log')
    sns.scatterplot(x='Average Node Degree', y='Median-Allocation', hue='Strategy', data=rand, ax=axs[1])
    axs[1].get_legend().remove()
    axs[1].set_title('Rand Graphs')

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)


    plt.show()
    ####################################################################################################################
    '''
    Design: How do the graphs look like
    '''

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(17, 9.27)

    zoo_t = zoo[['Average Node Degree', 'Diameter', 'Graph']]
    zoo_t['Network Type'] = 'Topology Zoo'
    rand_t = rand[['Average Node Degree', 'Diameter', 'Graph']]
    rand_t['Network Type'] = 'Random Networks'

    df_t = pd.concat([zoo_t, rand_t])

    sns.jointplot(data=df_t, x='Average Node Degree', y='Diameter', hue='Network Type', ax=ax)

    ax.legend(title='')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)

    plt.tight_layout()

    plt.xlim(0, )
    plt.ylim(0, )

    plt.show()
    ####################################################################################################################
    '''
    Results: Behaviour over graph metrics
    '''

    xs = ['Diameter', 'Size', 'Average Node Degree']
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.scatterplot(x='Diameter', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[0])
    axs[0].set_ylabel('Median Allocation [Gbps]')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.scatterplot(x='Size', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[1])
    axs[1].set_xlabel('#Nodes')
    axs[1].get_legend().remove()
    axs[1].grid(b=True)

    zoo['Average Node Degree / #Nodes'] = [row['Average Node Degree'] / (row['Size'] - 1) for index, row in
                                           zoo.iterrows()]

    sns.scatterplot(x='Average Node Degree / #Nodes', y='Median-Allocation', hue='Strategy', data=zoo, ax=axs[2])
    axs[2].set_xlabel('Graph Density')
    axs[2].get_legend().remove()
    axs[2].grid(b=True)

    handles, labels = axs[2].get_legend_handles_labels()
    lgd = axs[2].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    plt.yscale('log')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)

    plt.show()

    ####################################################################################################################
    '''
    Median Allocations - Constant Link Model vs. Degree-weighted link model, in Zoo and Rand graphs, 0.5 sampling for M-Approach
    '''
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
    '''

    ####################################################################################################################
    '''
    Median Allocations for sqos_ot sampling models - constant link model, 0.5 sampling 
    '''
    '''
    r_map = {
        '0.5': 'Degree-weighted probability distribution',
        'u0.5': 'Uniform probability distribution',
        'a0.5': 'Degree-weighted #paths per node'

    }
    df_samp_comp_zoo = df_alloc_const_zoo[((df_alloc_const_zoo['Strategy'] == STRATEGY_LABEL['sqos_ot']) | (df_alloc_const_zoo['Strategy'] == STRATEGY_LABEL['sqos_ob'])) & ((df_alloc_const_zoo['Ratio'] == '0.5') | (df_alloc_const_zoo['Ratio'] == 'u0.5'))]
    df_samp_comp_zoo['Ratio'].replace(r_map, inplace=True)
    df_samp_comp_rand = df_alloc_const_rand[((df_alloc_const_rand['Strategy'] == STRATEGY_LABEL['sqos_ot']) | (df_alloc_const_rand['Strategy'] == STRATEGY_LABEL['sqos_ob']))& ((df_alloc_const_rand['Ratio'] == '0.5') | (df_alloc_const_rand['Ratio'] == 'u0.5') )]
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
    #fig.suptitle('Influence of Sampling Strategy on M-Approach Allocations')
    #plt.show()
    '''


    ####################################################################################################################
    '''
    Cover plots: Median Covers by thresholds, degweighted, 1sp: GMA vs PT
    '''

    with open(os.path.join(args.which, 'aggregate_covers.csv'), "r+") as f:
        df_cover_all = pd.read_csv(f)

    df_cover = df_cover_all[~df_cover_all.Graph.str.startswith('c_')]
    df_cover["Cover Threshold"].replace(THRESH_LABEL, inplace=True)
    df_cover = df_cover[df_cover['Num Shortest Paths'] == 1]
    df_cover["Strategy"].replace(STRATEGY_LABEL, inplace=True)
    df_cover = df_cover[(df_cover['Strategy'] == STRATEGY_LABEL['GMAImproved']) | (df_cover['Strategy'] == STRATEGY_LABEL['sqos_pt'])]# | (df_cover['Strategy'] == STRATEGY_LABEL['sqos_pt']) | (df_cover['Ratio'] == '0.5')]

    df_cover_zoo = df_cover[~df_cover.Graph.str.startswith('Barabasi')]
    df_cover_rand = df_cover[df_cover.Graph.str.startswith('Barabasi')]
    '''
    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_zoo, hue_order=['GMA','PMA\nMax'], flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_rand, hue_order=['GMA','PMA\nMax'], flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].set_ylabel('')
    axs[1].grid(b=True)
    axs[1].get_legend().remove()

    handles, labels = axs[1].get_legend_handles_labels()
    axs[0].set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    #fig.suptitle('Strategies by Cover Thresholds Reached')
    plt.show()
    '''
    ####################################################################################################################
    '''
    Cover plots: Median Covers by thresholds, degweighted, 1sp: PT vs flavors
    '''

    df_cover = df_cover_all[~df_cover_all.Graph.str.startswith('c_')]
    df_cover["Cover Threshold"].replace(THRESH_LABEL, inplace=True)
    df_cover = df_cover[df_cover['Num Shortest Paths'] == 1]
    df_cover["Strategy"].replace(STRATEGY_LABEL, inplace=True)
    df_cover = df_cover[(df_cover['Strategy'] == STRATEGY_LABEL['sqos_pt']) | (df_cover['Strategy'] == STRATEGY_LABEL['sqos_pb']) | (df_cover['Ratio'] == '0.5')]

    df_cover_zoo = df_cover[~df_cover.Graph.str.startswith('Barabasi')]
    df_cover_rand = df_cover[df_cover.Graph.str.startswith('Barabasi')]

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_zoo, hue_order=STRAT_ORDER[1:], flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].get_legend().remove()
    axs[0].grid(b=True)

    sns.boxplot(x='Cover Threshold', y='Median-Cover', hue='Strategy', data=df_cover_rand, hue_order=STRAT_ORDER[1:], flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].set_ylabel('')
    axs[1].grid(b=True)
    axs[1].get_legend().remove()

    handles, labels = axs[1].get_legend_handles_labels()
    axs[0].set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    #fig.suptitle('Strategies by Cover Thresholds Reached')
    plt.show()

    ####################################################################################################################
    '''
    Cover box single graph: by thresholds
    '''
    '''
    covs = dh.get_covers_as_df(['Core(10000)'], STRATEGIES, [0.1], ['0.01', '0.001', '0.0001', '1e-05', '1e-06'])
    covs = covs.compute()

    covs["Cover Threshold"] = covs["Cover Threshold"].astype('float')
    covs["Cover Threshold"].replace(THRESH_LABEL, inplace=True)

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Cover Threshold', y='Cover [%]', hue='Strategy', data=covs, hue_order=STRAT_ORDER, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=ax)
    ax.grid(b=True)
    plt.savefig(os.path.join(dh.get_general_figure_path(), 'covers_b_threshs_core.png'))
    '''
    ####################################################################################################################
    '''
    Results: GMA vs PMA Max (Core)
    '''
    '''
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.set_size_inches(17, 9.27)

    data = dh.get_alloc_quots_as_df(['Core(10000)'], 'GMAImproved', ['sqos_pt'], ratios=['0.1'])

    d_min = data['Allocation Ratio'].min()
    d_max = data['Allocation Ratio'].max()
    med = data['Allocation Ratio'].median()
    mean = data['Allocation Ratio'].mean()

    axs.set_xlim(d_min, d_max)
    axs.grid(b=True)
    axs.axvline(x=1, c='r', linestyle='--', alpha=0.3)

    sns.ecdfplot(data=data, x='Allocation Ratio', hue="Strategies", ax=axs)#, palette=DIFF_C_MAP)

    axs.set_xlabel('Ratio PMA Max : GMA')
    axs.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    plt.tight_layout()
    plt.xscale('log')
    plt.savefig(os.path.join(dh.get_general_figure_path(),'gma_v_pmam_core.png'))
    '''
    ####################################################################################################################
    '''
    Results: PMA Max vs OMA Max & PMA Max vs PMA Concurrent: Core
    '''
    '''
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
    fig.set_size_inches(17, 9.27)

    data1 = dh.get_alloc_quots_as_df(['Core(10000)'], 'sqos_pt', ['sqos_ot'], ratios=['0.5'])
    data2 = dh.get_alloc_quots_as_df(['Core(10000)'], 'sqos_pt', ['sqos_pb'])
    print(data2)

    d_min1 = data1['Allocation Ratio'].min()
    d_max1 = data1['Allocation Ratio'].max()

    sns.ecdfplot(data=data1, x='Allocation Ratio', hue="Strategies", ax=axs[0])

    axs[0].set_title('PMA Max vs OMA Max')
    #axs[0].set_xlim(d_min1, d_max1)
    axs[0].grid(b=True)
    axs[0].axvline(x=1, c='r', linestyle='--', alpha=0.3)
    axs[0].set_xlabel('Ratio OMA Max : PMA Max')

    axs[0].set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])


    d_min2 = data2['Allocation Ratio'].min()
    d_max2 = data2['Allocation Ratio'].max()

    sns.ecdfplot(data=data2, x='Allocation Ratio', hue="Strategies", ax=axs[1])

    axs[1].set_title('PMA Max vs PMA Concurrent')
    #axs[1].set_xlim(d_min2, d_max2)
    axs[1].grid(b=True)
    axs[1].axvline(x=1, c='r', linestyle='--', alpha=0.3)
    axs[1].set_xlabel('Ratio PMA Concurrent: PMA Max')

    #handles, labels = axs[1].get_legend_handles_labels()
    #lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5), title='')
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)

    plt.xscale('log')
    plt.savefig(os.path.join(dh.get_general_figure_path(),'pmam_v_fl_core.png'))
    '''
    ####################################################################################################################
    '''
    Results: Allocations in Barabasi-Albert 15 25 1000 (cdf)
    '''
    '''
    dat = dh.get_allocs_as_df(['c_Barabasi_Albert_15_25_(1000)'], STRATEGIES, [0.5])
    dat['Allocations Gbps'] = dat['Allocations Gbps'].astype('float')

    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(17, 9.27)

    sns.ecdfplot(data=dat, x='Allocations Gbps', hue='Strategy', hue_order=STRAT_ORDER, ax=ax)

    ax.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    ax.set_ylabel('Percentiles')
    ax.set_xlabel('Allocations [Gbps]')

    plt.xscale('log')
    plt.savefig(os.path.join(dh.get_general_figure_path(),'all_allocs_cdf_barab.png'))

    '''
    ####################################################################################################################
    '''
    Results: GMA vs PMA Max
    '''
    '''
    fig, axs = plt.subplots(1, 1, sharey=True, sharex=True)
    fig.set_size_inches(17, 9.27)

    data = dh.get_alloc_quots_as_df(['c_Barabasi_Albert_15_25_(1000)'], 'GMAImproved', ['sqos_pt'], ratios=['0.1'])

    d_min = data['Allocation Ratio'].min()
    d_max = data['Allocation Ratio'].max()
    med = data['Allocation Ratio'].median()
    mean = data['Allocation Ratio'].mean()

    axs.set_xlim(d_min, d_max)
    axs.grid(b=True)
    axs.axvline(x=1, c='r', linestyle='--', alpha=0.3)

    sns.ecdfplot(data=data, x='Allocation Ratio', hue="Strategies", ax=axs)#, palette=DIFF_C_MAP)

    axs.set_xlabel('Ratio PMA Max : GMA')
    axs.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])

    plt.tight_layout()
    plt.xscale('log')
    plt.savefig(os.path.join(dh.get_general_figure_path(),'gma_v_pmam_barab.png'))
    '''
    ####################################################################################################################
    '''
    Results: PMA Max vs OMA Max & PMA Max vs PMA Concurrent
    '''
    '''
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True)
    fig.set_size_inches(17, 9.27)

    data1 = dh.get_alloc_quots_as_df(['c_Barabasi_Albert_15_25_(1000)'], 'sqos_pt', ['sqos_ot'], ratios=['0.5'])
    data2 = dh.get_alloc_quots_as_df(['c_Barabasi_Albert_15_25_(1000)'], 'sqos_pt', ['sqos_pb'])
    print(data2)

    d_min1 = data1['Allocation Ratio'].min()
    d_max1 = data1['Allocation Ratio'].max()

    sns.ecdfplot(data=data1, x='Allocation Ratio', hue="Strategies", ax=axs[0])

    axs[0].set_title('PMA Max vs OMA Max')
    #axs[0].set_xlim(d_min1, d_max1)
    axs[0].grid(b=True)
    axs[0].axvline(x=1, c='r', linestyle='--', alpha=0.3)
    axs[0].set_xlabel('Ratio OMA Max : PMA Max')

    axs[0].set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])


    d_min2 = data2['Allocation Ratio'].min()
    d_max2 = data2['Allocation Ratio'].max()

    sns.ecdfplot(data=data2, x='Allocation Ratio', hue="Strategies", ax=axs[1])

    axs[1].set_title('PMA Max vs PMA Concurrent')
    #axs[1].set_xlim(d_min2, d_max2)
    axs[1].grid(b=True)
    axs[1].axvline(x=1, c='r', linestyle='--', alpha=0.3)
    axs[1].set_xlabel('Ratio PMA Concurrent: PMA Max')

    #handles, labels = axs[1].get_legend_handles_labels()
    #lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5), title='')
    #fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)

    plt.xscale('log')
    plt.savefig(os.path.join(dh.get_general_figure_path(),'pmam_v_fl_barab.png'))
    '''

    ####################################################################################################################
    '''
    Results: Sampling ratios for each model: (cdf)
    '''
    '''
    r_map = {
        'u0.1': '0.1',
        'u0.2': '0.2',
        'u0.3': '0.3',
        'u0.4': '0.4',
        'u0.5': '0.5',
        'u0.6': '0.6',
        'u0.7': '0.7',
        'u0.8': '0.8',
        'u0.9': '0.9'
    }
    data_classic = dh.get_allocs_as_df(['c_Barabasi_Albert_15_25_(1000)'], ['sqos_ob'], ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    data_unif = dh.get_allocs_as_df(['c_Barabasi_Albert_15_25_(1000)'], ['sqos_ob'],
                                       ['u0.1', 'u0.2', 'u0.3', 'u0.4', 'u0.5', 'u0.6', 'u0.7', 'u0.8', 'u0.9'])
    data_unif['Ratio'] = data_unif['Ratio'].replace(r_map)

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    d_min = data_classic['Allocations Gbps'].min()
    d_max = data_classic['Allocations Gbps'].max()

    sns.ecdfplot(data=data_classic, x='Allocations Gbps', hue='Ratio', ax=axs[0])
    axs[0].set_xscale('log')
    axs[0].set_xlabel('Allocations [Gbps]')
    axs[0].set_title('Degree-weighted')

    sns.ecdfplot(data=data_unif, x='Allocations Gbps', hue='Ratio', ax=axs[1])
    axs[1].set_xscale('log')
    axs[1].set_xlabel('Allocations [Gbps]')
    axs[1].set_title('Uniform')

    plt.show()
    plt.savefig(os.path.join(dh.get_general_figure_path(),'samps.png'))
    '''
    ####################################################################################################################
    '''
    Results: Allocation multipath gain: cdf single graph
    '''
    #TODO: On Server
    '''
    data = dh.get_alloc_quots_multipath_as_df('c_Barabasi_Albert_20_30_(1000)', ['GMAImproved', 'sqos_pt'])
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(17, 9.27)

    sns.ecdfplot(data=data, x='Allocation Ratio', hue='Strategy', ax=ax)
    ax.set_yticks([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)

    plt.xscale('log')
    fig.suptitle('Multipath improvements for GMA and M-Approach PT Allocations in Barabasi-Albert 20_30_(1000)')
    #plt.show()
    '''

    ####################################################################################################################
    '''
    Results: Multipath improvements in allocations  
    '''
    #TODO: On Server
    df_alloc_gma_p = df_alloc_all[(df_alloc_all['Strategy'] == STRATEGY_LABEL['GMAImproved']) | (df_alloc_all['Strategy'] == STRATEGY_LABEL['sqos_pb']) | (df_alloc_all['Strategy'] == STRATEGY_LABEL['sqos_pt'])]
    print(df_alloc_gma_p)
    df_alloc_const_zoo = df_alloc_gma_p[df_alloc_gma_p.Graph.str.startswith('c_') & (~df_alloc_gma_p.Graph.str.startswith('c_Barabasi'))]
    print(df_alloc_const_zoo)
    df_alloc_const_rand = df_alloc_gma_p[df_alloc_gma_p.Graph.str.startswith('c_Barabasi')]
    print(df_alloc_const_rand)
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
    #fig.suptitle('Multipath improvements across Graphs')
    plt.yscale('log')
    plt.show()

    ####################################################################################################################
    '''
    Cover multipath improvement: Degree-weighted graphs
    '''
    df_cover = df_cover_all[~df_cover_all.Graph.str.startswith('c_')]
    df_cover["Cover Threshold"].replace(THRESH_LABEL, inplace=True)
    df_cover_gma_p = df_cover[(df_cover['Strategy'] == 'GMAImproved') | (df_cover['Strategy'] == 'sqos_pb') | (df_cover['Strategy'] == 'sqos_pt')]
    df_cover_gp_zoo = df_cover_gma_p[~df_cover_gma_p.Graph.str.startswith('Barabasi_Albert')]
    df_cover_gp_rand = df_cover_gma_p[df_cover_gma_p.Graph.str.startswith('Barabasi_Albert')]

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Strategy', y='Median-Cover', hue='Num Shortest Paths', data=df_cover_gp_zoo, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Cover [%]')
    axs[0].set_xlabel('')
    axs[0].grid(b=True)
    axs[0].get_legend().remove()

    sns.boxplot(x='Strategy', y='Median-Cover', hue='Num Shortest Paths', data=df_cover_gp_rand, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].grid(b=True)
    axs[1].get_legend().remove()

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5), title='#Shortest Paths')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    #fig.suptitle('Multipath improvements across Graphs')

    plt.show()

    ####################################################################################################################
    '''
    Cover multipath improvement: Const graphs
    '''
    df_cover = df_cover_all[df_cover_all.Graph.str.startswith('c_')]
    df_cover["Cover Threshold"].replace(THRESH_LABEL, inplace=True)
    df_cover_gma_p = df_cover[(df_cover['Strategy'] == 'GMAImproved') | (df_cover['Strategy'] == 'sqos_pb') | (df_cover['Strategy'] == 'sqos_pt')]
    df_cover_gp_zoo = df_cover_gma_p[~df_cover_gma_p.Graph.str.startswith('c_Barabasi_Albert')]
    df_cover_gp_rand = df_cover_gma_p[df_cover_gma_p.Graph.str.startswith('c_Barabasi_Albert')]

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    sns.boxplot(x='Strategy', y='Median-Cover', hue='Num Shortest Paths', data=df_cover_gp_zoo, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[0])
    axs[0].set_title('Zoo Graphs')
    axs[0].set_ylabel('Median Cover [%]')
    axs[0].set_xlabel('')
    axs[0].grid(b=True)
    axs[0].get_legend().remove()

    sns.boxplot(x='Strategy', y='Median-Cover', hue='Num Shortest Paths', data=df_cover_gp_rand, flierprops=dict(markerfacecolor='0.50', markersize=2), ax=axs[1])
    axs[1].set_title('Rand Graphs')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].grid(b=True)
    axs[1].get_legend().remove()

    handles, labels = axs[1].get_legend_handles_labels()
    lgd = axs[1].legend(handles, labels, loc='center left', ncol=1, bbox_to_anchor=(1.05, .5), title='#Shortest Paths')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2)
    #fig.suptitle('Multipath improvements across Graphs')

    plt.show()
    ####################################################################################################################
    '''
    Cover improvement multipath: percent improvement in medians, scatter
    '''
    df_cover = df_cover_all[df_cover_all.Graph.str.startswith('c_')]
    print(df_cover.columns)
    df_cover["Cover Threshold"].replace(THRESH_LABEL, inplace=True)
    df_cover = df_cover[df_cover['Cover Threshold'] == '10Mbps']
    df_cover_gma_p = df_cover[(df_cover['Strategy'] == 'GMAImproved') | (df_cover['Strategy'] == 'sqos_pb') | (df_cover['Strategy'] == 'sqos_pt')]
    df1 = df_cover_gma_p[df_cover_gma_p['Num Shortest Paths'] == 1]
    df2 = df_cover_gma_p[df_cover_gma_p['Num Shortest Paths'] == 2]
    df3 = df_cover_gma_p[df_cover_gma_p['Num Shortest Paths'] == 3]
    df5 = df_cover_gma_p[df_cover_gma_p['Num Shortest Paths'] == 5]

    print(df1)
    print(df2)
    dfall = pd.merge(df1[['Graph', 'Size', 'Strategy', 'Median-Cover']], df2[['Graph', 'Strategy', 'Median-Cover']], how='left',left_on=['Graph', 'Strategy'], right_on=['Graph', 'Strategy'], suffixes=['_1','_2'])
    dfall = pd.merge(dfall[['Graph', 'Size', 'Strategy', 'Median-Cover_1', 'Median-Cover_2']], df3[['Graph', 'Strategy', 'Median-Cover']], how='left', left_on=['Graph', 'Strategy'], right_on=['Graph', 'Strategy']).rename(columns={'Median-Cover': 'Median-Cover_3'})
    dfall = pd.merge(dfall[['Graph', 'Size', 'Strategy', 'Median-Cover_1', 'Median-Cover_2', 'Median-Cover_3']], df5[['Graph', 'Strategy', 'Median-Cover']], how='left',left_on=['Graph', 'Strategy'], right_on=['Graph', 'Strategy']).rename(columns={'Median-Cover': 'Median-Cover_5'})

    dfall = dfall.dropna()
    print(dfall)
    '''
    dfall['Median-Cover_2'] = ((dfall['Median-Cover_2']-dfall['Median-Cover_1'])/dfall['Median-Cover_1'])*100
    dfall['Median-Cover_3'] = ((dfall['Median-Cover_3'] - dfall['Median-Cover_1']) / dfall['Median-Cover_1'])*100
    dfall['Median-Cover_5'] = ((dfall['Median-Cover_5'] - dfall['Median-Cover_1']) / dfall['Median-Cover_1'])*100
    '''

    dfall = dfall.rename(columns={'Median-Cover_2': '2', 'Median-Cover_3': '3', 'Median-Cover_5': '5'})
    dfall = dfall.drop(axis=1, columns='Median-Cover_1')

    #dfall = pd.melt(dfall, id_vars=['Graph', 'Size', 'Strategy'], value_vars=['2', '3', '5'], var_name='# Shortest Paths', value_name='% Improvement')
    dfall = pd.melt(dfall, id_vars=['Graph', 'Size', 'Strategy'], value_vars=['2', '3', '5'], var_name='# Shortest Paths', value_name='Cover %')


    print(dfall)

    fig, axs = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(17, 9.27)

    dfall_zoo = dfall[~dfall.Graph.str.startswith('c_Barabasi_Albert')]
    dfall_rand = dfall[dfall.Graph.str.startswith('c_Barabasi_Albert')]


    sns.scatterplot(data=dfall_zoo, x='Size', y='Cover %', hue='# Shortest Paths', ax=axs[0])
    sns.scatterplot(data=dfall_rand, x='Size', y='Cover %', hue='# Shortest Paths', ax=axs[1])
    plt.show()

    pb_rand = dfall_rand[dfall_rand['Strategy'] == 'sqos_pb']
    sns.scatterplot(data=pb_rand, x='Size', y='Cover %', hue='# Shortest Paths')
    plt.show()


    ####################################################################################################################
    '''
    Results: Multipath Gain: Percentage cover improvement (scatter) (Barabasi Albert 15 25 1000)
    '''
    '''
    g_cov_1sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'GMAImproved', '1e-6')
    g_cov_2sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'GMAImproved', '1e-6', num_sp=2)
    g_cov_3sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'GMAImproved', '1e-6', num_sp=3)
    g_cov_5sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'GMAImproved', '1e-6', num_sp=5)



    m_cov_1sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'sqos_pt', '1e-6')
    m_cov_2sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'sqos_pt', '1e-6', num_sp=2)
    m_cov_3sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'sqos_pt', '1e-6', num_sp=3)
    m_cov_5sp = dh.get_cover('c_Barabasi_Albert_15_25_(1000)', 'sqos_pt', '1e-6', num_sp=5)
    # TODO
    '''

if __name__ == "__main__":
    args = parse_args()
    main(args)

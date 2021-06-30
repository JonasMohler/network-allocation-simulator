import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import pandas as pd
import os
import src.util.data_handler as dh
from src.util.const import UNIT, FIGURE_PATH
from src.util.naming import *
import seaborn as sns
#sns.set_theme(style="ticks", palette="pastel")
_PALETTE="Accent"
sns.set_theme(style="ticks", palette=_PALETTE)

_RATIOS = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

_STRATEGIES = [
    'GMAImproved',
    'sqos_ot',
    'sqos_pt',
    'sqos_ob',
    'sqos_pb'
]


_LABELS = [
    'GMA',
    'Optimistic M-Approach w/ Time Division',
    'Pessimistic M-Approach w/ Time Division',
    'Optimistic M-Approach w/ Bandwidth Division',
    'Pessimistic M-Approach w/ Bandwidth Division'
]

_C_MAP = dict(zip(_LABELS, sns.color_palette(_PALETTE, 5)))
print(_C_MAP)

_GRAPHS = [
    'Eenet(13)',
    'Epoch(6)',
    'Aarnet(19)',
    'Bics(33)',
    'Amres(25)',
    'Barabasi_Albert_2_20_(500)'
]
_STOL = dict(zip(_STRATEGIES, _LABELS))

_MARKERS = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V']
_STRATEGY_MARKERS = [_MARKERS[i] for i in range(len(_STRATEGIES))]

_ALL_GRAPHS = os.listdir('dat/topologies/')
_ALL_GRAPHS = [i for i in _ALL_GRAPHS if not i == 'Barabasi_Albert_20_50_(2500)'
               and not i == 'Erdos_Renyi_0.05_(250)'
               and not i == 'Erdos_Renyi_0.2_(500)'
               and not i == 'Barabasi_Albert_1_25_(500)'
               and not i == 'Barabasi_Albert_10_100_(250)'
               and not i == 'Barabasi_Albert_10_100_(500)'
               and not i == 'Barabasi_Albert_10_75_(250)'
               and not i == 'Barabasi_Albert_15_100_(250)'
               and not i == 'Barabasi_Albert_20_(250)']

_THRESH = '0.01'

_ADD_LINKS = [
    1,
    2,
    5,
    10,
    15
]
_INIT_NODES = [
    20,
    25,
    50,
    75,
    100
]
P_LINK_CREATE = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]
RAND_N_NODES = 250
#_BARAB = [list(zip(each_permutation, _ADD_LINKS)) for each_permutation in itertools.permutations(_INIT_NODES, len(_ADD_LINKS))]
#_BARAB = [f'Barabasi_Albert_{add_link}_{init_l}_({RAND_N_NODES})' for (init_l, add_link) in _BARAB]
#_ERDOS = [f"Erdos_Renyi_{p}_({RAND_N_NODES})" for p in P_LINK_CREATE]

#_RAND_GRAPHS = _BARAB + _ERDOS

_BOX_ALPHA = .3


def sfname(graph):
    size = int(graph.split('(')[1].split(')')[0])
    return size


def alloc_difference(a1, a2):

    diffs = {}
    for src, dests in a1.items():
        for dst in dests.keys():
            if src in a2 .keys() and dst in a2[src].keys():
                if src not in diffs.keys():
                    diffs[src] = {}
                diffs[src][dst] = a1[src][dst][0] - a2[src][dst][0]
    return diffs


def alloc_difference_list(a1,a2):
    diffs = []
    for src, dests in a1.items():
        for dst in dests.keys():
            if src in a2 .keys() and dst in a2[src].keys():
                diffs.append(a1[src][dst][0] - a2[src][dst][0])
    return diffs


def cover_difference_list(c1, c2):
    diffs = []
    for node, cover in c1.items():
        if node in c2.keys():
            diffs.append(c1[node]-c2[node])
    return diffs


# CDF Histo
def cdf_alloc_1v4(graph, ratio=0.5):
    s1 = _STRATEGIES[0]
    a1 = dh.get_allocations(graph, s1)
    df = pd.DataFrame(columns=['Strategies', f"Allocation Difference [{UNIT}]"])
    for s2 in _STRATEGIES[1:]:
        if s2 in ['sqos_ot', 'sqos_ob']:
            a2 = dh.get_allocations(graph, s2, ratio)
        else:
            a2 = dh.get_allocations(graph, s2)
        diff = alloc_difference_list(a1, a2)
        df_small = pd.DataFrame()
        df_small[f"Allocation Difference [{UNIT}]"] = diff
        df_small['Strategies'] = f"{_STOL[s1]} vs. {_STOL[s2]}"

        df = pd.concat([df, df_small], axis=0)

    fig, ax = plt.subplots()
    sns.ecdfplot(data=df, x=f"Allocation Difference [{UNIT}]", hue="Strategies")
    fig.suptitle(f"Cumulative Differences of Allocations [GBpS] between GMA and M-Approach Strategies")
    plt.show()


def cdf_cover_1v4(graph, ratio=0.5, thresh='0.01'):
    s1 = _STRATEGIES[0]
    c1 = dh.get_cover(graph, s1, thresh)
    df = pd.DataFrame(columns=['Strategies', f"Cover Difference"])
    for s2 in _STRATEGIES[1:]:
        if s2 in ['sqos_ot', 'sqos_ob']:
            c2 = dh.get_cover(graph, s2, thresh, ratio)
        else:
            c2 = dh.get_cover(graph, s2, thresh, None)
        diff = cover_difference_list(c1, c2)
        df_small = pd.DataFrame()
        df_small[f"Cover Difference"] = diff
        df_small['Strategies'] = f"{_STOL[s1]} vs. {_STOL[s2]}"

        df = pd.concat([df, df_small], axis=0)

    d_min = df['Cover Difference'].min()
    d_max = df['Cover Difference'].max()

    fig, ax = plt.subplots()
    sns.ecdfplot(data=df, x=f"Cover Difference", hue="Strategies", ax=ax)
    ax.set_xlim(d_min, d_max)
    ax.grid(b=True)
    ax.axvline(c='r', linestyle='--', alpha=0.3)
    fig.suptitle(f"Cumulative Differences of Covers between GMA and M-Approach Strategies")
    plt.show()


def cdf_cover_ratios_multigraph(graphs, thresh='0.01'):
    s1 = 'GMAImproved'
    s2 = 'sqos_ot'
    df = pd.DataFrame(columns=['Strategies', f"Cover Difference"])
    for graph in graphs:
        c1 = dh.get_cover(graph, s1, thresh)
        for r in _RATIOS:
            c2 = dh.get_cover(graph, s2, thresh, r)
            diff = cover_difference_list(c1, c2)
            df_small = pd.DataFrame()
            df_small[f"Cover Difference"] = diff
            df_small['Strategies'] = f"{_STOL[s1]} vs. {r}% {_STOL[s2]}"

            df = pd.concat([df, df_small], axis=0)

    d_min = df['Cover Difference'].min()
    d_max = df['Cover Difference'].max()

    fig, ax = plt.subplots()
    sns.ecdfplot(data=df, x=f"Cover Difference", hue="Strategies", ax=ax)
    ax.set_xlim(d_min, d_max)
    ax.grid(b=True)
    ax.axvline(c='r', linestyle='--', alpha=0.3)
    fig.suptitle(f"Cumulative Differences of Covers")
    plt.show()


def cdf_cover_gma_vs_sqos_ot_ratios(graph, thresh='0.01'):
    s1 = _STRATEGIES[0]
    c1 = dh.get_cover(graph, s1, thresh)
    df = pd.DataFrame(columns=['Strategies', f"Cover Difference"])
    s2 = 'sqos_ot'
    for r in _RATIOS:
        c2 = dh.get_cover(graph, s2, thresh, r)
        diff = cover_difference_list(c1, c2)
        df_small = pd.DataFrame()
        df_small[f"Cover Difference"] = diff
        df_small['Strategies'] = f"{_STOL[s1]} vs. {r*100}% {_STOL[s2]}"

        df = pd.concat([df, df_small], axis=0)


    d_min = df['Cover Difference'].min()
    d_max = df['Cover Difference'].max()

    fig, ax = plt.subplots()
    sns.ecdfplot(data=df, x=f"Cover Difference", hue="Strategies", ax=ax)
    ax.set_xlim(d_min, d_max)
    ax.grid(b=True)
    ax.axvline(c='r', linestyle='--', alpha=0.3)
    fig.suptitle(f"Cumulative Differences of Covers between GMA and Optimistic M-Approach w/ Time-Division Sampling Ratios")
    plt.show()


# Multi bar plot
def average_allocation_by_path_length():
    pass


# Scatter
def scatter_allocs_by_pl(graphs, strategies=_STRATEGIES, ratio=0.5):

    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])

    data = []

    for g in graphs:
        path_lengths = dh.get_pl(g)
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                alloc = dh.get_allocations(g, s, ratio)
            else:
                alloc = dh.get_allocations(g, s)

            for src, dests in alloc.items():
                for dst in dests.keys():
                    data.append((alloc[src][dst][0], path_lengths[src][dst]))

            df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    print(df)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy")#, palette=_C_MAP)
    plt.yscale('log')
    fig.suptitle(f"Allocations [GBpS] over Path Lengths")
    plt.show()


def scatter_allocs_by_pl_split(graphs, strategies=_STRATEGIES, ratio=0.5):

    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])

    data = []

    for g in graphs:
        path_lengths = dh.get_pl(g)
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                alloc = dh.get_allocations(g, s, ratio)
            else:
                alloc = dh.get_allocations(g, s)

            for src, dests in alloc.items():
                for dst in dests.keys():
                    data.append((alloc[src][dst][0], path_lengths[src][dst]))

            df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    print(df)
    fig, axs = plt.subplots(5, sharey=True)
    i = 0
    for s in strategies:
        sns.scatterplot(data=df[df['Strategy'] == _STOL[s]], x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy", ax=axs[i], palette=_C_MAP)
        i = i+1
    #plt.yscale('log')
    fig.suptitle(f"Allocations [GBpS] over Path Lengths")
    plt.show()


def scatter_alloc_by_pl(graphs, strategy, ratio=0.5):

    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])

    data = []

    for g in graphs:
        path_lengths = dh.get_pl(g)
        if strategy in ['sqos_ot', 'sqos_ob']:
            alloc = dh.get_allocations(g, strategy, ratio)
        else:
            alloc = dh.get_allocations(g, strategy)

        for src, dests in alloc.items():
            for dst in dests.keys():
                data.append((alloc[src][dst][0], path_lengths[src][dst]))

        df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
        df_small['Strategy'] = _STOL[strategy]
        df = pd.concat([df, df_small], axis=0)

    print(df)
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy", palette=_C_MAP)
    plt.yscale('log')
    fig.suptitle(f"Allocations [GBpS] over Path Lengths")
    plt.show()


def lm_alloc_by_pl(graphs, strategy, ratio=0.5):

    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])

    data = []

    for g in graphs:
        path_lengths = dh.get_pl(g)
        if strategy in ['sqos_ot', 'sqos_ob']:
            alloc = dh.get_allocations(g, strategy, ratio)
        else:
            alloc = dh.get_allocations(g, strategy)

        for src, dests in alloc.items():
            for dst in dests.keys():
                data.append((alloc[src][dst][0], float(path_lengths[src][dst])))

        df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
        df_small['Strategy'] = _STOL[strategy]
        df = pd.concat([df, df_small], axis=0)

    print(df)
    fig, ax = plt.subplots()
    sns.lmplot(data=df, x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy", palette=_C_MAP, x_ci="sd")#, fit_reg=True)
    plt.yscale('log')
    fig.suptitle(f"Allocations [GBpS] over Path Lengths")
    plt.show()


def box_alloc_by_pl(graphs, strategies=_STRATEGIES, ratio=0.5):

    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])


    data = []

    for g in graphs:
        path_lengths = dh.get_pl(g)
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                alloc = dh.get_allocations(g, s, ratio)
            else:
                alloc = dh.get_allocations(g, s)

            for src, dests in alloc.items():
                for dst in dests.keys():
                    data.append((alloc[src][dst][0], path_lengths[src][dst]))

            df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    print(df)
    fig, ax = plt.subplots()
    sns.boxplot(data=df, x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy")
    plt.yscale('log')
    fig.suptitle(f"Allocations [GBpS] over Path Lengths")
    plt.show()


def box_alloc_by_pl_split(graphs, strategies=_STRATEGIES, ratio=0.5):

    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])


    data = []

    for g in graphs:
        path_lengths = dh.get_pl(g)
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                alloc = dh.get_allocations(g, s, ratio)
            else:
                alloc = dh.get_allocations(g, s)

            for src, dests in alloc.items():
                for dst in dests.keys():
                    data.append((alloc[src][dst][0], path_lengths[src][dst]))

            df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    print(df)
    fig, axs = plt.subplots(5, sharey=True)
    i = 0
    for s in strategies:
        sns.boxplot(data=df[df['Strategy'] == _STOL[s]], x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy",
                        ax=axs[i], palette=_C_MAP)
        i = i + 1
    plt.yscale('log')
    fig.suptitle(f"Allocations [GBpS] over Path Lengths")
    plt.savefig('dat/figures/graph/Core(10000)/b_al_pl_s.eps', format='eps')
    #plt.show()


def box_cover_by_diameter(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.01'):
    # Load relevant covers
    df = pd.DataFrame(columns=['Diameter', 'Strategy', 'Cover'])
    for g in graphs:
        diameter = int(dh.get_diameter(g))
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['Diameter'] = diameter
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)
    print(df)
    fig, ax = plt.subplots()
    sns.boxplot(x='Diameter', y='Cover', hue='Strategy', data=df, ax=ax)
    for patch in ax.artists:
        r, g, b, a, = patch.get_facecolor()
        patch.set_facecolor((r, g, b, _BOX_ALPHA))
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Graph Diameter")
    plt.show()


def box_cover_by_size(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.01'):
    # Load relevant covers
    df = pd.DataFrame(columns=['# Nodes', 'Strategy', 'Cover'])
    for g in graphs:
        size = sfname(g)
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['# Nodes'] = size
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)
    print(df)
    fig, ax = plt.subplots()
    sns.boxplot(x='# Nodes', y='Cover', hue='Strategy', data=df)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Graph Size")
    plt.show()


def lm_cover_by_diameter(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.01'):
    # Load relevant covers
    df = pd.DataFrame(columns=['Diameter', 'Strategy', 'Cover'])
    for g in graphs:
        diameter = dh.get_diameter(g)
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['Diameter'] = diameter
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    df[['Diameter', 'Cover']] = df[['Diameter', 'Cover']].astype(float)
    print(df)
    fig, ax = plt.subplots()
    sns.lmplot(x='Diameter', y='Cover', hue='Strategy', data=df)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Graph Diameter")
    plt.show()


def scatter_cover_by_diameter(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.01'):
    # Load relevant covers
    df = pd.DataFrame(columns=['Diameter', 'Strategy', 'Cover'])
    for g in graphs:
        diameter = dh.get_diameter(g)
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['Diameter'] = int(diameter)
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    #df[['Diameter', ' Cover']] = df[['Diameter', 'Cover']].astype(float)
    print(df)
    fig, ax = plt.subplots()
    sns.scatterplot(x='Diameter', y='Cover', hue='Strategy', data=df, markers=_STRATEGY_MARKERS)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Graph Diameter")
    plt.show()


def box_cover_single_strat_by_diameter(graphs, strategy, ratio=0.5, thresh='0.001'):
    df = pd.DataFrame(columns=['Diameter', 'Strategy', 'Cover'])
    for g in graphs:
        diameter = dh.get_diameter(g)
        if strategy in ['sqos_ot', 'sqos_ob']:
            cover = dh.get_cover(g, strategy, thresh, ratio)
        else:
            cover = dh.get_cover(g, strategy, thresh, None)
        c = list(cover.values())
        df_small = pd.DataFrame(c, columns=['Cover'])
        df_small['Diameter'] = int(diameter)
        df_small['Strategy'] = _STOL[strategy]
        df = pd.concat([df, df_small], axis=0)

    print(df)
    fig, ax = plt.subplots()
    sns.boxplot(x='Diameter', y='Cover', hue='Strategy', data=df)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Graph Diameter")
    plt.show()


# TODO
def box_cover_by_node_degree(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.001'):
    df = pd.DataFrame(columns=['Node Degree', 'Strategy', 'Cover'])
    for g in graphs:
        deg = dh.get_degrees(g)
        degrees = deg['degrees']
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['Node Degree'] = degrees
            df_small['Node Degree'] = df_small['Node Degree'].astype(int)
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    # df[['Diameter', ' Cover']] = df[['Diameter', 'Cover']].astype(float)
    print(df)
    fig, ax = plt.subplots()
    sns.boxplot(x='Node Degree', y='Cover', hue='Strategy', data=df)#, markers=_STRATEGY_MARKERS)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Node Degrees")
    plt.show()


def scatter_cover_by_node_degree(graphs=_ALL_GRAPHS, strategies=_STRATEGIES, ratio=0.5, thresh='0.001'):
    df = pd.DataFrame(columns=['Node Degree', 'Strategy', 'Cover'])
    for g in graphs:
        deg = dh.get_degrees(g)
        degrees = deg['degrees']
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['Node Degree'] = degrees
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    # df[['Diameter', ' Cover']] = df[['Diameter', 'Cover']].astype(float)
    print(df)
    fig, ax = plt.subplots()
    sns.scatterplot(x='Node Degree', y='Cover', hue='Strategy', data=df, markers=_STRATEGY_MARKERS)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Node Degrees")
    plt.show()


def lm_cover_by_node_degree(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.001'):
    df = pd.DataFrame(columns=['Node Degree', 'Strategy', 'Cover'])
    for g in graphs:
        deg = dh.get_degrees(g)
        degrees = deg['degrees']
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['Node Degree'] = degrees
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    # df[['Diameter', ' Cover']] = df[['Diameter', 'Cover']].astype(float)
    print(df)
    fig, ax = plt.subplots()
    sns.lmplot(x='Node Degree', y='Cover', hue='Strategy', data=df, markers=_STRATEGY_MARKERS)
    fig.suptitle(f"Cover with {thresh} [GBpS] Threshold Over Node Degrees")
    plt.show()

_XS = {
    'degree': 'Node Degree',
    'size': 'Graph Size',
    'diameter': 'Graph Diameter',
    'pl': 'Path Length'
}

_YS = {
    'cover': 'Cover [%]',
    'alloc': f'Allocations {[UNIT]}'
}

'''
def get_x_metric(graph, kind):

    if kind == 'degree':
        x = dh.get_degrees(graph)
        x = x['degrees']
    elif kind == 'size':
        x = sfname(graph)
    elif kind == 'diameter':
        x = dh.get_diameter(graph)
    elif kind == 'pl':
        x = dh.get_pl(graph)
    else:
        raise ValueError('Error getting x metric: Unknown metric')

    return x

def get_y_metric(graph, kind, strat, thresh, ratio):

    if kind == 'cover':
        if strat in ['sqos_ot', 'sqos_ob']:
            cover = dh.get_cover(g, strat, thresh, ratio)
        else:
            cover = dh.get_cover(g, strat, thresh, None)
        c = list(cover.values())
        pass
    if kind == 'alloc':
        pass

def get_data_as_df(graphs, strategies, ratio, thresh, xs, ys):
     df = pd.DataFrame(columns=[_XS[xs], 'Strategy', _YS[ys]])
     for g in graphs:
         get_x_metric(g, xs)
         for s in strategies:
             get_y_metric(g, ys)

     if xs == 'degree':
         if ys == 'cover':
             df = pd.DataFrame(columns=['Node Degree', 'Strategy', 'Cover'])
             for g in graphs:
                 deg = dh.get_degrees(g)
                 degrees = deg['degrees']
                 for s in strategies:
                     if s in ['sqos_ot', 'sqos_ob']:
                         cover = dh.get_cover(g, s, thresh, ratio)
                     else:
                         cover = dh.get_cover(g, s, thresh, None)
                     c = list(cover.values())
                     df_small = pd.DataFrame(c, columns=['Cover'])
                     df_small['Node Degree'] = degrees
                     df_small['Strategy'] = _STOL[s]
                     df = pd.concat([df, df_small], axis=0)
         elif ys =='alloc':
             pass
         else:
             raise ValueError('Error: Unknown Y metric in DF get')
     elif xs == 'size':
         pass
     elif xs == 'diameter':
         pass
     elif xs == 'pl':
         pass
     else:
         raise ValueError('Error: Unknown X metric in DF get')


def make_plot(graphs=_ALL_GRAPHS, strategies = _STRATEGIES, ratio=0.1, thresh='0.01', x_metric='degree', y_metric='cover', plot_kind='scatter'):

    if y_metric == 'cover':
       pass

    pass
'''


def c_plot(store=True):
    # load appropriate data

    # create specified plot

    # store or show

    pass


#box_cover_by_node_degree(['Core(10000)'], ['sqos_ot'], ratio=0.1)

#scatter_cover_by_node_degree(ratio=0.1)

box_alloc_by_pl_split(['Core(10000)'], ratio=0.1)

'''
# Create all necessary dirs if not there yet
gen_path = os.path.join(FIGURE_PATH, 'general/')
graph_path = os.path.join(FIGURE_PATH, 'graph/')

if not os.path.exists(gen_path):
    os.mkdir(gen_path)
if not os.path.exists(graph_path):
    os.mkdir(graph_path)
for g in _ALL_GRAPHS:
    if not os.path.exists(get_graph_figure_path(g)):
        os.mkdir(get_graph_figure_path(g))


# Per Topology figures
for g in _ALL_GRAPHS:
    # Allocation scatters
    for s in _STRATEGIES:
        scatter_alloc_by_pl([g], s)
    scatter_allocs_by_pl([g])
    scatter_allocs_by_pl_split([g])

    # Allocation boxes
    box_alloc_by_pl([g])
    box_alloc_by_pl_split([g])

    # Allocation lms
    lm_alloc_by_pl([g], _STRATEGIES)

    # Cover by node degree scatter
    scatter_cover_by_node_degree([g])

    # Cover by node degree box
    box_cover_by_node_degree([g])

    # Cover by node degree lm
    lm_cover_by_node_degree([g])

    # Alloc cdf
    cdf_alloc_1v4(g)
    # Cover cdf
    cdf_cover_1v4(g)
    # Cover cdf by ratios
    cdf_cover_gma_vs_sqos_ot_ratios(g)

    pass


# Group figures rand
# TODO

# Cover by diameter box
box_cover_by_diameter(_ALL_GRAPHS, _STRATEGIES, thresh='0.001')

# Cover by diameter lm
lm_cover_by_diameter(_ALL_GRAPHS, _STRATEGIES, thresh='0.001')

# Cover by diameter scatter
scatter_cover_by_diameter(_ALL_GRAPHS, _STRATEGIES, thresh='0.001')

# Alloc by pl all
scatter_allocs_by_pl()
# Alloc by pl single
scatter_alloc_by_pl()
# Alloc by pl split
scatter_allocs_by_pl_split()

# Alloc cdf
cdf_alloc_1v4()
# Cover cdf
cdf_cover_1v4()


# Group figures zoo
# Cover by diameter box
# Cover by diameter lm
# Cover by diameter scatter

# Alloc by pl all
# Alloc by pl single
# Alloc by pl split

# Alloc cdf
# Cover cdf

# Group figures general
# Cover by diameter box
# Cover by diameter lm
# Cover by diameter scatter

# Alloc by pl all
# Alloc by pl single
# Alloc by pl split

# Alloc cdf
# Cover cdf




'''
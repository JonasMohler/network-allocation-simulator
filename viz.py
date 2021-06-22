import json
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import src.util.data_handler as dh
from src.util.const import UNIT
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

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
_GRAPHS = [
    'Eenet(13)',
    'Epoch(6)',
    'Aarnet(19)',
    'Bics(33)',
    'Amres(25)',
    'Barabasi_Albert_2_20_(500)'
]
_STOL = dict(zip(_STRATEGIES, _LABELS))

marker = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V']
_STRATEGY_MARKERS = [marker[i] for i in range(len(_STRATEGIES))]

_ALL_GRAPHS = os.listdir('dat/topologies/')

_THRESH = '0.01'


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

    sns.ecdfplot(data=df, x=f"Allocation Difference [{UNIT}]", hue="Strategies")
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
    ax.axvline(c='r')
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

    sns.ecdfplot(data=df, x=f"Cover Difference", hue="Strategies")
    plt.show()


# Multi bar plot
def average_allocation_by_path_length():
    pass


# Scatter
def scatter_alloc_by_pl(graph, strategies=_STRATEGIES, ratio=0.5):

    path_lengths = dh.get_pl(graph)
    data = []
    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])

    for s in strategies:
        if s in ['sqos_ot', 'sqos_ob']:
            alloc = dh.get_allocations(graph, s, ratio)
        else:
            alloc = dh.get_allocations(graph, s)

        for src, dests in alloc.items():
            for dst in dests.keys():
                data.append((alloc[src][dst][0], path_lengths[src][dst]))

        df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
        df_small['Strategy'] = _STOL[s]
        df = pd.concat([df, df_small], axis=0)
    print(df)
    sns.scatterplot(data=df, x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy", markers=_STRATEGY_MARKERS)
    plt.show()


def box_alloc_by_pl(graph, strategies=_STRATEGIES, ratio=0.5):

    path_lengths = dh.get_pl(graph)
    data = []
    df = pd.DataFrame(columns=['Strategy', f"Allocation [{UNIT}]", 'Path Length'])

    for s in strategies:
        if s in ['sqos_ot', 'sqos_ob']:
            alloc = dh.get_allocations(graph, s, ratio)
        else:
            alloc = dh.get_allocations(graph, s)

        for src, dests in alloc.items():
            for dst in dests.keys():
                data.append((alloc[src][dst][0], path_lengths[src][dst]))

        df_small = pd.DataFrame(data, columns=[f"Allocation [{UNIT}]", "Path Length"])
        df_small['Strategy'] = _STOL[s]
        df = pd.concat([df, df_small], axis=0)
    print(df)
    sns.boxplot(data=df, x="Path Length", y=f"Allocation [{UNIT}]", hue="Strategy")
    plt.yscale('log')
    plt.show()


# Aggregation of cover stats
def cover_stats():
    pass


# cover min/med/max by graph characteristics ( As boxplot?)
def box_cover_by_graph(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.01'):
    # Load relevant covers
    df = pd.DataFrame(columns=['Graph', 'Strategy', 'Cover'])
    for g in graphs:
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                cover = dh.get_cover(g, s, thresh, ratio)
            else:
                cover = dh.get_cover(g, s, thresh, None)
            c = list(cover.values())
            df_small = pd.DataFrame(c, columns=['Cover'])
            df_small['Graph'] = g
            df_small['Strategy'] = _STOL[s]
            df = pd.concat([df, df_small], axis=0)

    sns.boxplot(x='Graph', y='Cover', hue='Strategy', data=df)
    plt.show()


def box_cover_by_diameter(graphs, strategies=_STRATEGIES, ratio=0.5, thresh='0.01'):
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
    print(df)
    sns.boxplot(x='Diameter', y='Cover', hue='Strategy', data=df)
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
    sns.boxplot(x='# Nodes', y='Cover', hue='Strategy', data=df)
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
    sns.lmplot(x='Diameter', y='Cover', hue='Strategy', data=df)
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
    sns.scatterplot(x='Diameter', y='Cover', hue='Strategy', data=df)
    plt.show()


#lm_cover_by_diameter(_ALL_GRAPHS, thresh='0.001')

#scatter_cover_by_diameter(_ALL_GRAPHS, thresh='0.001')

#box_cover_by_diameter(_ALL_GRAPHS, thresh='0.001')

#cdf_cover_gma_vs_sqos_ot_ratios('Eenet(13)', thresh='0.001')

cdf_cover_1v4('Colt(153)', thresh='0.001')

#box_cover_by_size(_ALL_GRAPHS, thresh='0.001')

#cdf_cover_1v4('Eenet(13)')

#cdf_alloc_1v4('Eenet(13)')

#box_alloc_by_pl('Eenet(13)')

#scatter_alloc_by_pl('Eenet(13)')

#cover_box_by_strategies(_GRAPHS)

#cover_box_by_diameter(_ALL_GRAPHS, thresh='1e-3')




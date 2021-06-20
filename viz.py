import json
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import src.util.data_handler as dh
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")


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


_THRESH = '0.01'


def alloc_difference(a1, a2):
    pass


# CDF Histo
def plot_cdf():
    pass


# Multi bar plot
def average_allocation_by_path_length():
    pass


# Scatter
def scatter_alloc_by_pl():
    pass


# Aggregation of cover stats
def cover_stats():
    pass


# cover min/med/max by graph characteristics ( As boxplot?)
def cover_box_by_strategies(graphs, strategies, ratio=0.5, thresh='0.01'):
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

    sns.boxplot(x='graph', y='value', hue='strategy', data=df)
    plt.show()


def cover_box_by_size(graphs, strategies, ratio=0.5, thresh='0.01'):
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

    sns.boxplot(x='graph', y='value', hue='strategy', data=df)
    plt.show()


cover_box_by_strategies(_GRAPHS, _STRATEGIES)




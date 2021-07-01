import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import src.util.data_handler as dh
from src.util.const import UNIT, FIGURE_PATH
from src.util.naming import *
import seaborn as sns

_PALETTE = "Accent"
sns.set_theme(style="ticks", palette=_PALETTE)

_FORMAT = 'png'

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
_STOL = dict(zip(_STRATEGIES, _LABELS))

_MARKERS = ['o', 'x', '^', '+', '*', '8', 's', 'p', 'D', 'V']
_STRATEGY_MARKERS = [_MARKERS[i] for i in range(len(_STRATEGIES))]

_ALL_GRAPHS = os.listdir('dat/topologies/')
_THRESH = '0.01'
_RAND_GRAPHS = [x for x in os.listdir('dat/topologies') if x.startswith('Barabasi') or x.startswith('Erdos')]
_ZOO_GRAPHS = [x for x in _ALL_GRAPHS if x not in _RAND_GRAPHS and not x == 'Core(10000)']
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


def make_fig_single(x_name, y_name, data, title, p_type='scatter', save=False, path='', logy=False, logx=False, strat='alls'):

    name = f'{strat}_{y_name}_{x_name}'
    fig, ax = plt.subplots()

    if p_type == 'scatter':
        name = f"{name}_s_single.{_FORMAT}"

        sns.scatterplot(data=data, x=x_name, y=y_name, hue="Strategy", palette=_C_MAP)

    elif p_type == 'box':
        name = f"{name}_b_single.{_FORMAT}"

        sns.boxplot(x=x_name, y=y_name, hue='Strategy', data=data, ax=ax)
        '''
        for patch in ax.artists:
            r, g, b, a, = patch.get_facecolor()
            patch.set_facecolor((r, g, b, _BOX_ALPHA))
        '''
    elif p_type == 'lm':
        name = f"{name}_l_single.{_FORMAT}"

        sns.lmplot(data=data, x=x_name, y=y_name, hue="Strategy", palette=_C_MAP, x_ci="sd")

    else:
        raise ValueError('Error: Unsupported Plot Type')

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')

    fig.suptitle(title)

    if save:
        plt.savefig(os.path.join(path, name))
        plt.close()
    else:
        plt.show()


def make_fig_split(x_name, y_name, data, title, strategies, p_type='scatter', save=False, path='', logy=False, logx=False):

    name = f'{y_name}_{x_name}'
    fig, axs = plt.subplots(len(strategies), sharey=True)

    if p_type == 'scatter':
        name = f"{name}_s_split.{_FORMAT}"

        i = 0
        for s in strategies:
            sns.scatterplot(data=data[(data["Strategy"]==_STOL[s])], x=x_name, y=y_name, hue="Strategy", ax=axs[i], palette=_C_MAP)

            i = i + 1

    elif p_type == 'box':
        name = f"{name}_b_split.{_FORMAT}"

        i = 0
        for s in strategies:
            sns.boxplot(data=data[(data["Strategy"]==_STOL[s])], x=x_name, y=y_name, hue="Strategy", ax=axs[i], palette=_C_MAP)

            '''
            for patch in axs[i].artists:
                r, g, b, a, = patch.get_facecolor()
                patch.set_facecolor((r, g, b, _BOX_ALPHA))
            '''
            i = i + 1

    elif p_type == 'lm':
        name = f"{name}_l_split.{_FORMAT}"

        i = 0
        for s in strategies:
            if s in ['sqos_ob', 'sqos_ot']:
                sns.lmplot(data=data[(data["Strategy"]==_STOL[s])], x=x_name, y=y_name, hue="Strategy", ax=axs[i], palette=_C_MAP)
            i = i + 1

    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')

    fig.suptitle(title)

    if save:
        plt.savefig(os.path.join(path, name))
        plt.close()
    else:
        plt.show()


def make_fig(x_name, y_name, data, title, type='scatter', split=False, save=False, path='', logy=False, logx=False):
    if split:
        make_fig_split(x_name, y_name, data, title, type=type, save=save, path=path, logy=logy, logx=logx)
    else:
        make_fig_single(x_name, y_name, data, title, type=type, save=save, path=path, logy=logy, logx=logx)


def make_cover_cdf(data, title, save=False, path='', logx=False):

    name = f"cov_cdf_strats.{_FORMAT}"

    d_min = data['Cover Difference'].min()
    d_max = data['Cover Difference'].max()

    fig, ax = plt.subplots()
    sns.ecdfplot(data=data, x=f"Cover Difference", hue="Strategies", ax=ax)
    ax.set_xlim(d_min, d_max)
    ax.grid(b=True)
    ax.axvline(c='r', linestyle='--', alpha=0.3)

    fig.suptitle(title)

    if logx:
        plt.xscale('log')

    if save:
        plt.savefig(os.path.join(path, name))
        plt.close()
    else:
        plt.show()


def make_alloc_cdf(data, title, save=False, path='', logx=False):

    name = f"alloc_cdf_strats.{_FORMAT}"

    d_min = data[f"Allocation Difference [{UNIT}]"].min()
    d_max = data[f"Allocation Difference [{UNIT}]"].max()

    fig, ax = plt.subplots()
    sns.ecdfplot(data=data, x=f"Allocation Difference [{UNIT}]", hue="Strategies", ax=ax)
    ax.set_xlim(d_min, d_max)
    ax.grid(b=True)
    ax.axvline(c='r', linestyle='--', alpha=0.3)

    fig.suptitle(title)

    if logx:
        plt.xscale('log')

    if save:
        plt.savefig(os.path.join(path, name))
        plt.close()
    else:
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


def make_heat_scatter(data, path, s, save=False):
    name = f'heatmap_{s}.{_FORMAT}'
    fig, ax = plt.subplots()
    sc = ax.scatter(data['Source Degree'], data['Destination Degree'], c=data[_YS['alloc']], cmap='copper')
    fig.colorbar(sc, ax=ax)

    fig.suptitle(f"Source and Destination Degree Heatmap for {s}")

    if save:
        plt.savefig(os.path.join(path, name))
        plt.close()
    else:
        plt.show()


def get_cover_diffs_as_df(graphs, s1='GMAImproved', s2s=['sqos_ot'], ratios=[0.1], threshs=['0.001']):

    df = pd.DataFrame(columns=['Strategies', f"Cover Difference", "Cover Threshold"])
    for g in graphs:
        for t in threshs:
            c1 = dh.get_cover(g, s1, t)
            for s2 in s2s:
                if s2 in ['sqos_ot','sqos_ob']:
                    for r in ratios:
                        c2 = dh.get_cover(g, s2, t, r)
                        diff = cover_difference_list(c1, c2)
                        df_small = pd.DataFrame()
                        df_small["Cover Difference"] = diff
                        df_small['Strategies'] = f"{_STOL[s1]} vs. {r*100}% {_STOL[s2]}"
                        df_small["Cover Threshold"] = t

                        df = pd.concat([df, df_small], axis=0)
                else:
                    c2 = dh.get_cover(g, s2, t)
                    diff = cover_difference_list(c1, c2)
                    df_small = pd.DataFrame()
                    df_small["Cover Difference"] = diff
                    df_small['Strategies'] = f"{_STOL[s1]} vs. {r * 100}% {_STOL[s2]}"
                    df_small["Cover Threshold"] = t

                    df = pd.concat([df, df_small], axis=0)


    return df


def get_covers_as_df(graphs, strategies, ratios=[0.1], threshs=['0.001']):

    df = pd.DataFrame(columns=[_YS['cover'], _XS['degree'], _XS['size'], _XS['diameter'], "Strategy", "Ratio"])

    for g in graphs:

        deg = dh.get_degrees(g)
        degrees = [float(x) for x in deg['degrees']]
        size = float(sfname(g))
        diameter = float(dh.get_diameter(g))

        for s in strategies:
            for t in threshs:
                if s in ['sqos_ot', 'sqos_ob']:
                    for r in ratios:
                        cover = dh.get_cover(g, s, t, r)
                        c = list(cover.values())
                        df_small = pd.DataFrame()
                        df_small[_YS['cover']] = c
                        df_small[_XS['degree']] = degrees
                        df_small[_XS['size']] = size
                        df_small[_XS['diameter']] = diameter
                        df_small['Strategy'] = _STOL[s]
                        df_small["Ratio"] = r

                        df = pd.concat([df, df_small])
                else:
                    cover = dh.get_cover(g, s, t, None)
                    c = list(cover.values())
                    df_small = pd.DataFrame()
                    df_small[_YS['cover']] = c
                    df_small[_XS['degree']] = degrees
                    df_small[_XS['size']] = size
                    df_small[_XS['diameter']] = diameter
                    df_small['Strategy'] = _STOL[s]
                    df_small["Ratio"] = 1

                    df = pd.concat([df, df_small])

    return df


def get_alloc_diffs_as_df(graphs, s1='GMAImproved', s2s=['sqos_ot'], ratios=[0.1]):
    df = pd.DataFrame(columns=['Strategies', f"Allocation Difference [{UNIT}]"])
    for g in graphs:
        a1 = dh.get_allocations(g, s1)

        for s2 in s2s:
            if s2 in ['sqos_ot', 'sqos_ob']:
                for r in ratios:
                    a2 = dh.get_allocations(g, s2, r)
                    diff = alloc_difference_list(a1, a2)
                    df_small = pd.DataFrame()
                    df_small[f"Allocation Difference [{UNIT}]"] = diff
                    df_small['Strategies'] = f"{_STOL[s1]} vs. {_STOL[s2]}"

                    df = pd.concat([df, df_small], axis=0)
            else:
                a2 = dh.get_allocations(g, s2)
                diff = alloc_difference_list(a1, a2)
                df_small = pd.DataFrame()
                df_small[f"Allocation Difference [{UNIT}]"] = diff
                df_small['Strategies'] = f"{_STOL[s1]} vs. {_STOL[s2]}"

                df = pd.concat([df, df_small], axis=0)
    return df


def get_allocs_as_df(graphs, strategies, ratios=[0.1]):

    df = pd.DataFrame(columns=[_YS['alloc'], "Path Length", "Source Degree", "Destination Degree", "Strategy", "Ratio"])

    for g in graphs:

        als = []
        pls = []
        src_degs = []
        dst_degs = []

        path_lengths = dh.get_pl(g)
        degrees = dh.get_degrees(g)

        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                for r in ratios:
                    alloc = dh.get_allocations(g, s, r)
                    for src, dests in alloc.items():
                        src_deg = degrees['degrees'][degrees['nodes'].index(src)]
                        for dst in dests.keys():
                            als.append(alloc[src][dst][0])
                            pls.append(path_lengths[src][dst])
                            src_degs.append(src_deg)
                            dst_degs.append(degrees['degrees'][degrees['nodes'].index(dst)])

                    df_small = pd.DataFrame()
                    df_small[_YS['alloc']] = als
                    df_small["Path Length"] = pls
                    df_small["Source Degree"] = src_degs
                    df_small["Destination Degree"] = dst_degs
                    df_small['Strategy'] = _STOL[s]
                    df_small["Ratio"] = r

                    df = pd.concat([df, df_small])

            else:
                alloc = dh.get_allocations(g, s)

                for src, dests in alloc.items():
                    src_deg = degrees['degrees'][degrees['nodes'].index(src)]
                    for dst in dests.keys():
                        als.append(alloc[src][dst][0])
                        pls.append(path_lengths[src][dst])
                        src_degs.append(src_deg)
                        dst_degs.append(degrees['degrees'][degrees['nodes'].index(dst)])

                df_small = pd.DataFrame()
                df_small[_YS['alloc']] = als
                df_small["Path Length"] = pls
                df_small["Source Degree"] = src_degs
                df_small["Destination Degree"] = dst_degs
                df_small['Strategy'] = _STOL[s]
                df_small["Ratio"] = 1

                df = pd.concat([df, df_small])

    return df


#################################################
'''
data = get_allocs_as_df(['Colt(153)'], _STRATEGIES, [0.1])
for s in _STRATEGIES:
    make_heat_scatter(data[data["Strategy"] == _STOL[s]], dh.get_graph_figure_path('Colt(153)'), s)
'''

data = get_allocs_as_df(['Core(10000)'], _STRATEGIES, [0.1])
# Allocation plots per strategy
for s in _STRATEGIES:
    dat = data[data["Strategy"] == _STOL[s]]
    print(f'shape data: {dat.shape}')
    for t in ['scatter', 'box']:
        make_fig_single(_XS['pl'], _YS['alloc'], dat, f"{_STOL[s]} Allocations by Path Length in {'Core(10000)'}", p_type=t, save=True, path=dh.get_graph_figure_path(g), strat=s, logy=True)

    # TODO: Heatmap src-dst alloc
# Allocation plots all strategies
for t in ['scatter', 'box']:
    make_fig_single(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in {'Core(10000)'}", p_type=t, save=True, path=dh.get_graph_figure_path(g), logy=True)
    make_fig_split(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in {'Core(10000)'}", _STRATEGIES, p_type=t, save=True, path=dh.get_graph_figure_path(g), logy=True)

# Cover plots per strategy
data = get_covers_as_df(['Core(10000)'], _STRATEGIES)
for s in _STRATEGIES:
    dat = data[data["Strategy"] == s]
    for t in ['scatter']:#, 'box']:
        for xm in ['degree', 'size', 'diameter']:
           make_fig_single(_XS[xm], _YS['cover'], dat, f"{_STOL[s]} Cover by {_XS[xm]} in {'Core(10000)'}", p_type=t, save=True, path=dh.get_graph_figure_path(g))

# Cover plots all strategies
for t in ['scatter', 'box']:
    for xm in ['degree', 'size', 'diameter']:
        make_fig_single(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in {'Core(10000)'}", p_type=t, save=True,
                        path=dh.get_graph_figure_path('Core(10000)'))
        make_fig_split(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in {'Core(10000)'}", _STRATEGIES, p_type=t, save=True,
                        path=dh.get_graph_figure_path('Core(10000)'))

# CDF plots

# Cover CDF
data = get_cover_diffs_as_df(['Core(10000)'], _STRATEGIES[0], _STRATEGIES[1:])
make_cover_cdf(data, f"CDF of Covers in {'Core(10000)'}", save=True, path=dh.get_graph_figure_path('Core(10000)'))

# Alloc CDF
data = get_alloc_diffs_as_df(['Core(10000)'], _STRATEGIES[0], _STRATEGIES[1:])
make_alloc_cdf(data, f"CDF of Allocations in {'Core(10000)'}", save=True, path=dh.get_graph_figure_path('Core(10000)'))

'''

# Create all necessary dirs if not there yet
gen_path = os.path.join(FIGURE_PATH, 'general/')
graph_path = os.path.join(FIGURE_PATH, 'graph/')

if not os.path.exists(gen_path):
    os.mkdir(gen_path)
if not os.path.exists(graph_path):
    os.mkdir(graph_path)
for g in _ALL_GRAPHS:
    if not os.path.exists(dh.get_graph_figure_path(g)):
        print(f'making dir {dh.get_graph_figure_path((g))}')
        os.mkdir(dh.get_graph_figure_path(g))
    else:
        print(f"Dir exists: {dh.get_graph_figure_path(g)}")


# Per Topology figures
_SELECTED_GRAPHS = _RAND_GRAPHS + ['Kdl(754)', 'Colt(153)']
all_g = len(_SELECTED_GRAPHS)
i = 0

print('Starting per topology plots')
for g in _SELECTED_GRAPHS:
    data = get_allocs_as_df([g], _STRATEGIES, [0.1])
    # Allocation plots per strategy
    for s in _STRATEGIES:
        dat = data[data["Strategy"] == _STOL[s]]
        print(f'shape data: {dat.shape}')
        for t in ['scatter', 'box']:
            make_fig_single(_XS['pl'], _YS['alloc'], dat, f"{_STOL[s]} Allocations by Path Length in {g}", p_type=t, save=True, path=dh.get_graph_figure_path(g), strat=s, logy=True)

        # TODO: Heatmap src-dst alloc
    # Allocation plots all strategies
    for t in ['scatter', 'box']:
        make_fig_single(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in {g}", p_type=t, save=True, path=dh.get_graph_figure_path(g), logy=True)
        make_fig_split(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in {g}", _STRATEGIES, p_type=t, save=True, path=dh.get_graph_figure_path(g), logy=True)

    # Cover plots per strategy
    data = get_covers_as_df([g], _STRATEGIES)
    for s in _STRATEGIES:
        dat = data[data["Strategy"] == s]
        for t in ['scatter']:#, 'box']:
            for xm in ['degree', 'size', 'diameter']:
                make_fig_single(_XS[xm], _YS['cover'], dat, f"{_STOL[s]} Cover by {_XS[xm]} in {g}", p_type=t, save=True, path=dh.get_graph_figure_path(g))

    # Cover plots all strategies
    for t in ['scatter', 'box']:
        for xm in ['degree', 'size', 'diameter']:
            make_fig_single(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in {g}", p_type=t, save=True,
                            path=dh.get_graph_figure_path(g))
            make_fig_split(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in {g}", _STRATEGIES, p_type=t, save=True,
                            path=dh.get_graph_figure_path(g))

    # CDF plots

    # Cover CDF
    data = get_cover_diffs_as_df([g], _STRATEGIES[0], _STRATEGIES[1:])
    make_cover_cdf(data, f"CDF of Covers in {g}", save=True, path=dh.get_graph_figure_path(g))

    # Alloc CDF
    data = get_alloc_diffs_as_df([g], _STRATEGIES[0], _STRATEGIES[1:])
    make_alloc_cdf(data, f"CDF of Allocations in {g}", save=True, path=dh.get_graph_figure_path(g))

    # Cover cdf by ratios TODO
    # cdf_cover_gma_vs_sqos_ot_ratios(g)

    i = i+1
    print(f"{i*100/all_g}% of topologies plotted ")


print('Starting Group plots: All Graphs')

# Group figures all graphs
print('Starting Allocation plots')
data = get_allocs_as_df(_ALL_GRAPHS, _STRATEGIES, [0.1])
# Allocation plots per strategy
for s in _STRATEGIES:
    dat = data[data["Strategy"] == _STOL[s]]
    for t in ['scatter', 'box']:
        make_fig_single(_XS['pl'], _YS['alloc'], dat, f"{_STOL[s]} Allocations by Path Length in all Graphs", p_type=t,
                        save=True, path=dh.get_general_figure_path(), strat=s)

    # TODO: Heatmap src-dst alloc
# Allocation plots all strategies
for t in ['scatter', 'box']:
    make_fig_single(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in all Graphs", p_type=t, save=True,
                    path=dh.get_graph_figure_path(g))
    make_fig_split(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in all Graphs", _STRATEGIES, p_type=t,
                   save=True, path=dh.get_general_figure_path())

# Cover plots per strategy
print('Starting Cover Plots')
data = get_covers_as_df([g], _STRATEGIES)
for s in _STRATEGIES:
    dat = data[data["Strategy"] == s]
    for t in ['scatter', 'box']:
        for xm in ['degree', 'size', 'diameter']:
            make_fig_single(_XS[xm], _YS['cover'], dat, f"{_STOL[s]} Cover by {_XS[xm]} in all Graphs", p_type=t, save=True,
                            path=dh.get_general_figure_path(), strat=s)

# Cover plots all strategies
for t in ['scatter', 'box']:
    for xm in ['degree', 'size', 'diameter']:
        make_fig_single(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in all Graphs", p_type=t, save=True,
                        path=dh.get_general_figure_path())
        make_fig_split(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in all Graphs", _STRATEGIES, p_type=t, save=True,
                       path=dh.get_general_figure_path())

# CDF plots

# Cover CDF
print('Starting Cover CDF')
data = get_cover_diffs_as_df(_ALL_GRAPHS, _STRATEGIES[0], _STRATEGIES[1:])
make_cover_cdf(data, f"CDF of Covers in all Graphs", save=True, path=dh.get_general_figure_path())

# Alloc CDF
print('Starting Alloc CDF')
data = get_alloc_diffs_as_df(_ALL_GRAPHS, _STRATEGIES[0], _STRATEGIES[1:])
make_alloc_cdf(data, f"CDF of Allocations in all Graphs", save=True, path=dh.get_general_figure_path())


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



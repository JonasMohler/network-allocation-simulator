from viz import *
_PALETTE = "Accent"
sns.set_theme(style="ticks", palette=_PALETTE)

_FORMAT = 'pdf'
_STRATEGIES = [
    'GMAImproved',
    'sqos_ot',
    'sqos_pt',
    'sqos_ob',
    'sqos_pb'
]

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

_LABELS = [
    'GMA',
    'Optimistic M-Approach w/ Time Division',
    'Pessimistic M-Approach w/ Time Division',
    'Optimistic M-Approach w/ Bandwidth Division',
    'Pessimistic M-Approach w/ Bandwidth Division'
]

_C_MAP = dict(zip(_LABELS, sns.color_palette(_PALETTE, 5)))
_STOL = dict(zip(_STRATEGIES, _LABELS))


_ALL_GRAPHS = os.listdir('dat/topologies/')
_ALL_GRAPHS = [x for x in _ALL_GRAPHS if not x == 'Core(10000)']
_THRESH = '0.01'
_RAND_GRAPHS = [x for x in os.listdir('dat/topologies') if x.startswith('Barabasi') or x.startswith('Erdos')]
_ZOO_GRAPHS = [x for x in _ALL_GRAPHS if x not in _RAND_GRAPHS and not x == 'Core(10000)']
_BOX_ALPHA = .3

print('Starting Group plots: All Graphs')

# Group figures all graphs
print('Starting Allocation plots')
data = get_allocs_as_df(_ALL_GRAPHS, _STRATEGIES, [0.1])
# Allocation plots per strategy
for s in _STRATEGIES:
    dat = data[data["Strategy"] == _STOL[s]]
    for t in ['scatter', 'box']:
        make_fig_single(_XS['pl'], _YS['alloc'], dat, f"{_STOL[s]} Allocations by Path Length in all Graphs", p_type=t,
                        save=True, path=dh.get_general_figure_path(), strat=s, logy=True)

    # TODO: Heatmap src-dst alloc
# Allocation plots all strategies
for t in ['scatter', 'box']:
    make_fig_single(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in all Graphs", p_type=t, save=True,
                    path=dh.get_general_figure_path(), logy=True)
    make_fig_split(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in all Graphs", _STRATEGIES, p_type=t,
                   save=True, path=dh.get_general_figure_path(), logy=True)

# Cover plots per strategy
print('Starting Cover Plots')
data = get_covers_as_df(_ALL_GRAPHS, _STRATEGIES)
print(data)
for s in _STRATEGIES:
    dat = data[data["Strategy"] == s]
    for t in ['scatter', 'box']:
        for xm in ['degree', 'size', 'diameter']:
            try:
                make_fig_single(_XS[xm], _YS['cover'], dat, f"{_STOL[s]} Cover by {_XS[xm]} in all Graphs", p_type=t, save=True,
                            path=dh.get_general_figure_path(), strat=s)
            except ValueError as e:
                print(f'Error in plotting for \nstrat: {s}\ntype: {t}\n x metric: {xm}\n Error: {e}')

# Cover plots all strategies
for t in ['scatter', 'box']:
    for xm in ['degree', 'size', 'diameter']:
        make_fig_single(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in all Graphs", p_type=t, save=True,
                        path=dh.get_general_figure_path())
        make_fig_split(_XS[xm], _YS['cover'], data, f"Cover by {_XS[xm]} in all Graphs", _STRATEGIES, p_type=t, save=True,
                       path=dh.get_general_figure_path())

# CDF plots
make_cover_cdf_abs(data, f"CDF of Covers in all Graphs", save=True, path = dh.get_general_figure_path())
# Cover CDF
print('Starting Cover CDF')
data = get_cover_diffs_as_df(_ALL_GRAPHS, _STRATEGIES[0], _STRATEGIES[1:])
make_cover_cdf(data, f"CDF of Cover Differences in all Graphs", save=True, path=dh.get_general_figure_path())


# Alloc CDF
print('Starting Alloc CDF')
data = get_alloc_diffs_as_df(_ALL_GRAPHS, _STRATEGIES[0], _STRATEGIES[1:])
print('Creating Alloc CDF Plot')
make_alloc_cdf(data, f"CDF of Allocations in all Graphs", save=True, path=dh.get_general_figure_path())

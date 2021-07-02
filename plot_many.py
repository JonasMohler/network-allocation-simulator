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
    # TODO: Heatmap src-dst alloc
# Allocation plots all strategies
make_fig_single(_XS['pl'], _YS['alloc'], data[data["Strategy"] == _STOL['GMAImproved']], f"Allocations by Path Length in all graphs", p_type='box', save=True, path=dh.get_general_figure_path(), logy=True, strat='GMA')
make_fig_split(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in all graphs", _STRATEGIES[1:], p_type='box', save=True, path=dh.get_general_figure_path(), logy=True)


# Cover plots per strategy
print('Starting Cover Plots')
data = get_covers_as_df(_ALL_GRAPHS, _STRATEGIES)

# Cover plots all strategies
make_fig_single(_XS['degree'], _YS['cover'], data[data["Strategy"] == _STOL["GMAImproved"]], f"Cover by {_XS['degree']} in all graphs", p_type='scatter', save=True,
                        path=dh.get_general_figure_path())

make_fig_split(_XS['degree'], _YS['cover'], data, f"Cover by {_XS['degree']} in all graphs", _STRATEGIES[1:], p_type='scatter', save=True,
                        path=dh.get_general_figure_path())

# CDF plots
make_cover_cdf_abs(data, f"CDF of Covers in all graphs", save=True, path=dh.get_general_figure_path())
# Cover CDF
data = get_cover_diffs_as_df(_ALL_GRAPHS, _STRATEGIES[0], _STRATEGIES[1:])
make_cover_cdf(data, f"CDF of Cover Differences in all graphs", save=True, path=dh.get_general_figure_path())

# Alloc CDF
data = get_alloc_diffs_as_df(_ALL_GRAPHS, _STRATEGIES[0], _STRATEGIES[1:])
make_alloc_cdf(data, f"CDF of Allocation Differences in all graphs", save=True, path=dh.get_general_figure_path())

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
    'alloc': 'Allocations GBpS'
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

_THRESH = '0.01'


data = get_allocs_as_df(['Core(10000)'], _STRATEGIES, [0.1])
    # TODO: Heatmap src-dst alloc
# Allocation plots all strategies
make_fig_single(_XS['pl'], _YS['alloc'], data[data["Strategy"] == _STOL['GMAImproved']], f"Allocations by Path Length in Core", p_type='box', save=True, path=dh.get_graph_figure_path(g), logy=True, strat='GMA')
make_fig_split(_XS['pl'], _YS['alloc'], data, f"Allocations by Path Length in {'Core(10000)'}", _STRATEGIES[1:], p_type='box', save=True, path=dh.get_graph_figure_path(g), logy=True)

# Cover plots per strategy
data = get_covers_as_df(['Core(10000)'], _STRATEGIES)

# Cover plots all strategies
make_fig_single(_XS['degree'], _YS['cover'], data[data["Strategy"] == _STOL["GMAImproved"]], f"Cover by {_XS['degree']} in Core", p_type='scatter', save=True,
                        path=dh.get_graph_figure_path('Core(10000)'))

make_fig_split(_XS['degree'], _YS['cover'], data, f"Cover by {_XS['degree']} in {'Core(10000)'}", _STRATEGIES[1:], p_type='scatter', save=True,
                        path=dh.get_graph_figure_path('Core(10000)'))

# CDF plots
make_cover_cdf_abs(data, f"CDF of Covers in Core", save=True, path=dh.get_graph_figure_path('Core(10000)'))
# Cover CDF
data = get_cover_diffs_as_df(['Core(10000)'], _STRATEGIES[0], _STRATEGIES[1:])
make_cover_cdf(data, f"CDF of Cover Differences in Core", save=True, path=dh.get_graph_figure_path('Core(10000)'))

# Alloc CDF
data = get_alloc_diffs_as_df(['Core(10000)'], _STRATEGIES[0], _STRATEGIES[1:])
make_alloc_cdf(data, f"CDF of Allocation Differences in Core", save=True, path=dh.get_graph_figure_path('Core(10000)'))

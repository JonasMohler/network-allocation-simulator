from argparse import ArgumentParser
from src.multiprocessing.topology.PerTopologyOperations import *

from src.util.utility import *
from src.util.const import *
import src.util.data_handler as dh
import src.util.plot_handler as ph


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
        "--multi",
        action='store_true',
        help="Create Plots with data from all specified Graphs"
    )
    parser.add_argument(
        "--single",
        action='store_true',
        help="Create Plots for all specified Graphs individually"
    )

    args = parser.parse_args()
    return args


def main(args):
    graphs = args.dirs

    if args.multi:

        print('Starting Group plots: All Graphs')

        # Allocation plots
        print('Starting Allocation plots')
        data = dh.get_allocs_as_df(graphs, STRATEGIES, [0.1])
        ph.make_fig_single(PLOT_X_LABEL['pl'], PLOT_Y_LABEL['alloc'], data[data["Strategy"] == STRATEGY_LABEL['GMAImproved']],
                    f"Allocations by Path Length in all graphs", p_type='box', save=True,
                    path=dh.get_general_figure_path(), logy=True, strat='GMA')
        ph.make_fig_split(PLOT_X_LABEL['pl'], PLOT_Y_LABEL['alloc'], data, f"Allocations by Path Length in all graphs", STRATEGIES[1:],
                   p_type='box', save=True, path=dh.get_general_figure_path(), logy=True)

        # TODO: Heatmap src-dst alloc

        # Cover plots
        print('Starting Cover Plots')
        data = dh.get_covers_as_df(graphs, STRATEGIES)

        ph.make_fig_single(PLOT_X_LABEL['degree'], PLOT_Y_LABEL['cover'], data[data["Strategy"] == STRATEGY_LABEL["GMAImproved"]],
                        f"Cover by {PLOT_X_LABEL['degree']} in all graphs", p_type='scatter', save=True,
                        path=dh.get_general_figure_path())

        ph.make_fig_split(PLOT_X_LABEL['degree'], PLOT_Y_LABEL['cover'], data, f"Cover by {PLOT_X_LABEL['degree']} in all graphs", STRATEGIES[1:],
                       p_type='scatter', save=True,
                       path=dh.get_general_figure_path())

        print('Starting CDFs')

        # CDF plots
        ph.make_fig_single('', '', data, f"CDF of Covers in all {g}", save=True, path=dh.get_general_figure_path(), p_type='cdf_c')
        # Cover CDF
        data = dh.get_cover_diffs_as_df(graphs, STRATEGIES[0], STRATEGIES[1:])
        ph.make_fig_single('', '', data, f"CDF of Cover Differences in {g}", save=True, path=dh.get_general_figure_path(), p_type='cdf_cd')

        # Alloc CDF
        data = dh.get_alloc_diffs_as_df(graphs, STRATEGIES[0], STRATEGIES[1:])
        ph.make_fig_single('', '', data, f"CDF of Allocation Differences in {g}", save=True, path=dh.get_general_figure_path(), p_type='cdf_ad')

    if args.single:

        for g in graphs:
            # Allocation plots
            print('Starting Allocation plots')
            data = dh.get_allocs_as_df([g], STRATEGIES, [0.1])

            ph.make_fig_single(PLOT_X_LABEL['pl'], PLOT_Y_LABEL['alloc'],
                               data[data["Strategy"] == STRATEGY_LABEL['GMAImproved']],
                               f"Allocations by Path Length in {g}", p_type='box', save=True,
                               path=dh.get_graph_figure_path(g), logy=True, strat='GMA')

            ph.make_fig_split(PLOT_X_LABEL['pl'], PLOT_Y_LABEL['alloc'], data,
                              f"Allocations by Path Length in {g}", STRATEGIES[1:],
                              p_type='box', save=True, path=dh.get_graph_figure_path(g), logy=True)

            # TODO: Heatmap src-dst alloc

            # Cover plots
            print('Starting Cover Plots')
            data = dh.get_covers_as_df([g], STRATEGIES)

            ph.make_fig_single(PLOT_X_LABEL['degree'], PLOT_Y_LABEL['cover'],
                               data[data["Strategy"] == STRATEGY_LABEL["GMAImproved"]],
                               f"Cover by {PLOT_X_LABEL['degree']} in {g}", p_type='scatter', save=True,
                               path=dh.get_graph_figure_path(g))

            ph.make_fig_split(PLOT_X_LABEL['degree'], PLOT_Y_LABEL['cover'], data,
                              f"Cover by {PLOT_X_LABEL['degree']} in {g}", STRATEGIES[1:],
                              p_type='scatter', save=True,
                              path=dh.get_graph_figure_path(g))

            print('Starting CDFs')

            # CDF plots
            ph.make_fig_single('', '', data, f"CDF of Covers in {g}", save=True, path=dh.get_graph_figure_path(g),
                               p_type='cdf_c')
            # Cover CDF
            data = dh.get_cover_diffs_as_df([g], STRATEGIES[0], STRATEGIES[1:])
            ph.make_fig_single('', '', data, f"CDF of Cover Differences in {g}", save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_cd')

            data = dh.get_covers_as_df([g], [STRATEGIES[1]], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            ph.make_fig_single('', '', data, f"CDF of Covers for different Opt. SQoS w/ T. Div. Ratios in {g}", save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_cr')

            # Alloc CDF
            data = dh.get_allocs_as_df([g], STRATEGIES)
            ph.make_fig_single('', '', data, f"CDF of Allocations in {g}", save=True, path=dh.get_graph_figure_path(g),
                               p_type='cdf_a', logx=True)
            data = dh.get_alloc_diffs_as_df([g], STRATEGIES[0], STRATEGIES[1:])
            ph.make_fig_single('', '', data, f"CDF of Allocation Differences in {g}", save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_ad')

            data = dh.get_allocs_as_df([g], [STRATEGIES[1]], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            ph.make_fig_single('', '', data, f"CDF of Allocations for different Opt. SQoS w/ T. Div. Ratios in {g}",
                               save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_ar', logx=True)


if __name__ == "__main__":
    args = parse_args()
    main(args)

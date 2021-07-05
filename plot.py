from argparse import ArgumentParser

import matplotlib.pyplot as plt

from src.multiprocessing.topology.PerTopologyOperations import *

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
        "--multi",
        action='store_true',
        help="Create Plots with data from all specified Graphs"
    )
    parser.add_argument(
        "--single",
        action='store_true',
        help="Create Plots for all specified Graphs individually"
    )
    parser.add_argument(
        "--r",
        nargs="+",
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Sampling ratios",
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

            degrees = dh.get_degrees(g)
            ph.make_fig_single('', '', degrees, f"CDF of Node Degrees in {g}", p_type='cdf_d', save=True, path=dh.get_graph_figure_path(g))

            # Load Allocation data
            print('Loading Allocation data ...')
            ''''''
            print(args.r)
            data = dh.get_allocs_as_df([g], STRATEGIES, args.r)
            #print(data)
            dbg = data[data["Strategy"] == STRATEGY_LABEL['GMAImproved']].compute()
            #print(dbg)
            dbs = data[(data["Ratio"] == '0.1') | (data["Ratio"] == 1)].compute()
            print(dbs)
            dbr = data[data["Strategy"] == STRATEGY_LABEL['sqos_ot']].compute()
            #print(dbr)
            '''
            sqos_ot_01 = data[(data["Strategy"] == STRATEGY_LABEL['sqos_ot']) & (data["Ratio"] == 'u0.1')].compute()
            sqos_ot_05 = data[(data["Strategy"] == STRATEGY_LABEL['sqos_ot']) & (data["Ratio"] == 'u0.5')].compute()
            sqos_ob_01 = data[(data["Strategy"] == STRATEGY_LABEL['sqos_ob']) & (data["Ratio"] == 'u0.1')].compute()
            sqos_ob_05 = data[(data["Strategy"] == STRATEGY_LABEL['sqos_ob']) & (data["Ratio"] == 'u0.5')].compute()
            sqos_pt = data[(data["Strategy"] == STRATEGY_LABEL['sqos_pt'])].compute()
            sqos_pb = data[(data["Strategy"] == STRATEGY_LABEL['sqos_pb'])].compute()

            data = dh.get_alloc_diffs_as_df([g], 'sqos_pt', ratios=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

            ax = plt.subplot()
            d_min = data['Allocation Difference [Gbps]'].min()
            d_max = data['Allocation Difference [Gbps]'].max()

            sns.ecdfplot(data=data, x=PLOT_Y_LABEL['alloc_d'], hue="Ratio", ax=ax)

            ax.set_xlim(d_min, d_max)
            plt.show()
            ph.make_fig_single('','', data, f"alloc diff over ratios sqos_pt - sqos_ot", p_type='cdf_ad', path=dh.get_graph_figure_path(g))

            print(sqos_ot_01.shape())
            ax = plt.subplot()
            ax.hist(sqos_ot_01['Allocations Gbps'], bins = 5000)
            plt.xscale('log')
            plt.show()

            print(sqos_ot_05.shape())
            ax = plt.subplot()
            ax.hist(sqos_ot_05['Allocations Gbps'], bins=5000)
            plt.xscale('log')
            plt.show()


            print(sqos_ot_01.median())
            print(sqos_ot_05.median())
            print(sqos_ob_01.median())
            print(sqos_ob_05.median())
            print(sqos_pt.median())
            print(sqos_pb.median())
'''
            # Allocation Plots
            print('Generating Allocation plots ...')

            # Allocations Over Path Length Box - GMA
            ph.make_fig_single(PLOT_X_LABEL['pl'], PLOT_Y_LABEL['alloc'], dbg
                               ,
                               f"Allocations by Path Length in {g}", p_type='box', save=True,
                               path=dh.get_graph_figure_path(g), logy=True, strat='GMA')

            # Allocations Over Path Length Box - SQOS
            ph.make_fig_split(PLOT_X_LABEL['pl'], 'Allocations Gbps', dbs,
                              f"Allocations by Path Length in {g}", STRATEGIES[1:],
                              p_type='box', save=True, path=dh.get_graph_figure_path(g), logy=True)

            # Alloc CDF - All Strategies
            ph.make_fig_single('', '', dbs, f"CDF of Allocations in {g}", save=True, path=dh.get_graph_figure_path(g),
                               p_type='cdf_a', logx=True)

            # Alloc CDF - Different sampling ratios SQOS OT
            ph.make_fig_single('', '', dbr, f"CDF of Allocations for different Opt. SQoS w/ T. Div. Ratios in {g}",
                               save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_ar', logx=True)


            # Alloc CDF - Differences between GMA vs others
            data = dh.get_alloc_diffs_as_df([g], STRATEGIES[0], [STRATEGIES[1]])
            ph.make_fig_single('', '', data, f"CDF of Allocation Differences in {g}", save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_ad')

            # Alloc Ratios CDF - GMA vs others (DEF TAKE IN)
            data = dh.get_alloc_quots_as_df([g], STRATEGIES[0], STRATEGIES[1:])
            ph.make_fig_single('', '', data, f"CDF of Allocation Ratios in {g}", save=True, path=dh.get_graph_figure_path(g), p_type='cdf_adr', logx=True)

            # TODO: Heatmap src-dst alloc

            # Cover plots
            print('Loading Cover Data')
            data = dh.get_covers_as_df([g], STRATEGIES, args.r, [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

            print('Generating Cover Plots')
            # Cover Scatter - GMA
            ph.make_fig_single(PLOT_X_LABEL['degree'], PLOT_Y_LABEL['cover'],
                               data[
                                   (data["Strategy"] == STRATEGY_LABEL["GMAImproved"]) &
                                   (data["Cover Threshold"] == 1e-6)
                               ].compute(),
                               f"Cover by {PLOT_X_LABEL['degree']} in {g}", p_type='scatter', save=True,
                               path=dh.get_graph_figure_path(g))

            # Cover Scatter - SQOS
            ph.make_fig_split(PLOT_X_LABEL['degree'], PLOT_Y_LABEL['cover'],
                              data[
                                  (data["Cover Threshold"] == 1e-6) &
                                  ((data["Ratio"] == '0.1') | (data["Ratio"] == 1))
                              ].compute(),
                              f"Cover by {PLOT_X_LABEL['degree']} in {g}", STRATEGIES[1:],
                              p_type='scatter', save=True,
                              path=dh.get_graph_figure_path(g))

            # Cover CDF - All Strategies
            ph.make_fig_single('', '',
                               data[
                                   (data["Cover Threshold"] == 1e-6) &
                                   ((data["Ratio"] == '0.1') | (data["Ratio"] == 1))
                               ].compute(),
                               f"CDF of Covers in {g}", save=True, path=dh.get_graph_figure_path(g),
                               p_type='cdf_c')

            # Covers CDF - Different sampling ratios SQOS OT
            ph.make_fig_single('', '',
                               data[
                                   (data["Strategy"] == STRATEGY_LABEL['sqos_ot']) &
                                   (data["Cover Threshold"] == 1e-6)
                               ].compute(),
                               f"CDF of Covers for different Opt. SQoS w/ T. Div. Ratios in {g}", save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_cr')

            # Covers for different Cover Thresholds - GMA
            ph.make_fig_single('', '',
                               data[
                                   (data["Strategy"] == STRATEGY_LABEL['GMAImproved'])
                               ].compute(), f"CDF of Cover for different Cover Thresholds in {g} - GMA", save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_ct')

            ph.make_fig_split('', '',
                              data[
                                  (data["Ratio"] == '0.1') |
                                  (data["Ratio"] == 1)
                              ].compute(),
                              f"CDF of Cover for different Cover Thresholds in {g} M-Approach flavours", STRATEGIES[1:], save =True,
                              path=dh.get_graph_figure_path(g), p_type='cdf_ct')

            # Cover CDF - Differences between GMA vs others
            data = dh.get_cover_diffs_as_df([g], STRATEGIES[0], STRATEGIES[1:])
            ph.make_fig_single('', '', data, f"CDF of Cover Differences in {g}", save=True,
                               path=dh.get_graph_figure_path(g), p_type='cdf_cd')




if __name__ == "__main__":
    args = parse_args()
    main(args)

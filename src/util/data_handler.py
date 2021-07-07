"""
Loaders for the intermediate data used by GMA computation algorithms.
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
import fnss
import dask
import dask.dataframe as dd

from src.util.utility import *
from src.util.naming import *
from src.util.const import *


def _load(graph, data_type, strategy=None, ratio=None, thresh=None):

    full_path = get_full_path(graph, data_type, strategy, ratio, thresh)

    # Load file

    if FILE_TYPE[data_type] == "json":
        with open(full_path, "r") as infile:
            data = json.load(infile)

    elif FILE_TYPE[data_type] == "txt":
        with open(full_path, "r") as infile:
            data = infile.read()

    elif FILE_TYPE[data_type] == "pkl":
        with open(full_path, "rb") as infile:
            data = pickle.load(infile)

    elif FILE_TYPE[data_type] == "csv":
        with open(full_path) as outfile:
            np.genfromtxtoutfile, delimiter=','
        #data = pd.read_csv(full_path)

    elif FILE_TYPE[data_type] == "xml":
        data = fnss.read_topology(full_path)

    return data


def _store(data, graph, data_type, strategy=None, ratio=None, thresh=None):

    # Build parent directory and make if they don't yet exist and is allowed
    topology_path = get_topo_path(graph)

    if not os.path.exists(topology_path):
        if data_type == TOPOLOGY:
            os.mkdir(topology_path)
        else:
            raise ValueError(F"Graph not found: {topology_path}")

    strategy_path = get_strategies_path(graph)
    graph_path = get_graph_path(graph)

    if not os.path.exists(strategy_path):
        os.mkdir(strategy_path)

    if strategy is not None:
        concrete_strategy_path = os.path.join(strategy_path, strategy)
        if not os.path.exists(concrete_strategy_path):
            os.mkdir(concrete_strategy_path)

    if not os.path.exists(graph_path):
        os.mkdir(graph_path)

    # Get file name
    full_path = get_full_path(graph, data_type, strategy, ratio, thresh)

    # Store

    if FILE_TYPE[data_type] == "json":
        with open(full_path, "w+") as outfile:
            json.dump(data, outfile)

    elif FILE_TYPE[data_type] == "txt":
        with open(full_path, "w+") as outfile:
            outfile.write(str(data))

    elif FILE_TYPE[data_type] == "pkl":
        with open(full_path, "wb+") as outfile:
            pickle.dump(data, outfile)

    elif FILE_TYPE[data_type] == "csv":
        with open(full_path, "w+") as outfile:
            np.savetxt(outfile, data, delimiter=',')
        #print(f"Writing csv to: {full_path}")
        #data.to_csv(full_path, index=False)
    elif FILE_TYPE[data_type] == "xml":
        fnss.write_topology(data, full_path)


def get_graph(graph):
    data = _load(graph, TOPOLOGY)
    return data


def set_graph(data, graph):
    _store(data, graph, TOPOLOGY)


def get_allocations(graph, strategy, ratio=None):
    data = _load(graph, ALLOCATION, strategy=strategy, ratio=ratio)
    return data


def set_allocations(data, graph, strategy, ratio=None):
    _store(data, graph, ALLOCATION, strategy=strategy, ratio=ratio)


def get_diameter(graph):
    data = _load(graph, DIAMETER)
    return data


def set_diameter(data, graph):
    _store(data, graph, DIAMETER)


def get_degrees(graph):
    data = _load(graph, DEGREE)
    return data


def set_degrees(data, graph):
    _store(data, graph, DEGREE)


def get_cover(graph, strategy, thresh, ratio=None):
    data = _load(graph, COVER, strategy=strategy, ratio=ratio, thresh=thresh)
    return data


def set_cover(data, graph, strategy, thresh, ratio=None):
    _store(data, graph, COVER, strategy=strategy, ratio=ratio, thresh=thresh)


def get_tm(graph):
    data = _load(graph, ALLOCATION_MATRIX)
    return data


def set_tm(data, graph):
    _store(data, graph, ALLOCATION_MATRIX)


def get_pl(graph):
    data = _load(graph, PATH_LENGTHS)
    return data


def set_pl(data, graph):
    _store(data, graph, PATH_LENGTHS)


def get_pc(graph, ratio=None):
    data = _load(graph, PATH_COUNTS, ratio=ratio)
    return data


def set_pc(data, graph, ratio=None):
    print(f"Writing path counts for {graph} with ratio {ratio}")
    _store(data, graph, PATH_COUNTS, ratio=ratio)


def get_shortest_paths(graph, ratio=None):
    data = _load(graph, SHORTEST_PATH, ratio=ratio)
    return data


def set_shortest_paths(data, graph, ratio=None):
    _store(data, graph, SHORTEST_PATH, ratio=ratio)


def get_alloc_diffs_as_df(graphs, s1='GMAImproved', s2s=['sqos_ot'], ratios=[0.1]):

    small_dfs = []

    print(f"Fetching Allocation differences for {len(graphs)} graphs")

    all_g = len(graphs)
    j = 0
    for g in graphs:
        print(f"Fetching Allocation differences for {g}")
        a1 = get_allocations(g, s1)

        all_s = len(s2s)
        i = 0
        print(f"Fetching Allocation differences for {all_s} Strategies")
        for s2 in s2s:
            if s2 in ['sqos_ot', 'sqos_ob']:
                for r in ratios:
                    a2 = get_allocations(g, s2, r)
                    diff = alloc_difference_list(a1, a2)
                    df_small = pd.DataFrame()
                    df_small[f"Allocation Difference [{UNIT}]"] = diff
                    df_small['Strategies'] = f"{STRATEGY_LABEL[s1]} vs. {STRATEGY_LABEL[s2]}"
                    df_small["Ratio"] = r

                    #df = pd.concat([df, df_small], axis=0)
                    small_dfs.append(df_small)
            else:
                a2 = get_allocations(g, s2)
                diff = alloc_difference_list(a1, a2)
                df_small = pd.DataFrame()
                df_small[f"Allocation Difference [{UNIT}]"] = diff
                df_small['Strategies'] = f"{STRATEGY_LABEL[s1]} vs. {STRATEGY_LABEL[s2]}"
                df_small["Ratio"] = 1

                #df = pd.concat([df, df_small], axis=0)
                small_dfs.append(df_small)

            i = i + 1
            print(f'Done with {i * 100 / all_s}% of strategies for current graph')

        j = j + 1
        print(f"Donw with {j * 100 / all_g}% of graphs")

    print("Concatenating single results")

    df = pd.concat(small_dfs)
    return df


def get_alloc_quots_as_df(graphs, s1='GMAImproved', s2s=['sqos_ot'], ratios=[0.1]):

    small_dfs = []

    print(f"Fetching Allocation ratios for {len(graphs)} graphs")

    all_g = len(graphs)
    j = 0
    for g in graphs:
        print(f"Fetching Allocation ratios for {g}")
        if s1 in ['sqos_ot', 'sqos_ob']:
            a1 = get_allocations(g, s1, 0.1)
        else:
            a1 = get_allocations(g, s1)

        all_s = len(s2s)
        i = 0
        print(f"Fetching Allocation ratios for {all_s} Strategies")
        for s2 in s2s:
            if s2 in ['sqos_ot', 'sqos_ob']:
                for r in ratios:
                    a2 = get_allocations(g, s2, r)
                    print('quots1')
                    quots = alloc_quotients_list(a1, a2)
                    df_small = pd.DataFrame()
                    df_small[f"Allocation Ratio"] = quots
                    df_small['Strategies'] = f"{STRATEGY_LABEL[s1]} vs. {STRATEGY_LABEL[s2]}"
                    df_small["Ratio"] = r

                    #df = pd.concat([df, df_small], axis=0)
                    small_dfs.append(df_small)
            else:
                a2 = get_allocations(g, s2)
                print('quots2')
                quots = alloc_quotients_list(a1, a2)
                df_small = pd.DataFrame()
                df_small[f"Allocation Ratio"] = quots
                df_small['Strategies'] = f"{STRATEGY_LABEL[s1]} vs. {STRATEGY_LABEL[s2]}"
                df_small["Ratio"] = 1

                #df = pd.concat([df, df_small], axis=0)
                small_dfs.append(df_small)

            i = i + 1
            print(f'Done with {i * 100 / all_s}% of strategies for current graph')

        j = j + 1
        print(f"Donw with {j * 100 / all_g}% of graphs")

    print("Concatenating single results")

    df = pd.concat(small_dfs)
    return df


def get_allocs_with_deg_as_df(graphs, strategies, ratios=[0.1]):

    df = pd.DataFrame(columns=[PLOT_Y_LABEL['alloc'], "Path Length", "Source Degree", "Destination Degree", "Strategy", "Ratio"])

    print(f"Fetching Allocations for {len(graphs)} graphs")

    all_g = len(graphs)
    j = 0
    small_dfs = []
    for g in graphs:
        print(f"Fetching Allocations for {g}")
        als = []
        pls = []
        src_degs = []
        dst_degs = []

        path_lengths = get_pl(g)
        degrees = get_degrees(g)

        all_s = len(strategies)
        i=0
        print(f"Fetching Allocations for {all_s} Strategies")
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                for r in ratios:
                    alloc = get_allocations(g, s, r)
                    for src, dests in alloc.items():
                        src_deg = degrees['degrees'][degrees['nodes'].index(src)]
                        for dst in dests.keys():
                            als.append(alloc[src][dst][0])
                            pls.append(path_lengths[src][dst])
                            src_degs.append(src_deg)
                            dst_degs.append(degrees['degrees'][degrees['nodes'].index(dst)])

                    df_small = pd.DataFrame()
                    df_small[PLOT_Y_LABEL['alloc']] = als
                    df_small["Path Length"] = pls
                    df_small["Source Degree"] = src_degs
                    df_small["Destination Degree"] = dst_degs
                    df_small['Strategy'] = STRATEGY_LABEL[s]
                    df_small["Ratio"] = r

                    #df = pd.concat([df, df_small])
                    small_dfs.append(df_small)

            else:
                alloc = get_allocations(g, s)

                for src, dests in alloc.items():
                    src_deg = degrees['degrees'][degrees['nodes'].index(src)]
                    for dst in dests.keys():
                        als.append(alloc[src][dst][0])
                        pls.append(path_lengths[src][dst])
                        src_degs.append(src_deg)
                        dst_degs.append(degrees['degrees'][degrees['nodes'].index(dst)])

                df_small = pd.DataFrame()
                df_small[PLOT_Y_LABEL['alloc']] = als
                df_small["Path Length"] = pls
                df_small["Source Degree"] = src_degs
                df_small["Destination Degree"] = dst_degs
                df_small['Strategy'] = STRATEGY_LABEL[s]
                df_small["Ratio"] = 1

                #df = pd.concat([df, df_small])
                small_dfs.append(df_small)

            i = i+1
            print(f'Done with {i*100/all_s}% of strategies for current graph')

        j = j+1
        print(f"Donw with {j*100/all_g}% of graphs")
    print("Concatenating single results")
    df = pd.concat(small_dfs)
    return df


def get_cover_diffs_as_df(graphs, s1='GMAImproved', s2s=['sqos_ot'], ratios=[0.1], threshs=['0.001']):

    all_g = len(graphs)
    print(f"Fetching Cover differences for {all_g} graphs")
    small_dfs = []

    j = 0
    for g in graphs:
        print(f"Fetching Cover differences for {g}")
        for t in threshs:
            c1 = get_cover(g, s1, t)
            all_s = len(s2s)
            i = 0
            print(f"Fetching Covers for {all_s} Strategies")
            for s2 in s2s:
                if s2 in ['sqos_ot','sqos_ob']:
                    for r in ratios:
                        c2 = get_cover(g, s2, t, r)
                        diff = cover_difference_list(c1, c2)
                        df_small = pd.DataFrame()
                        df_small["Cover Difference"] = diff
                        df_small['Strategies'] = f"{STRATEGY_LABEL[s1]} vs. {r*100}% {STRATEGY_LABEL[s2]}"
                        df_small["Cover Threshold"] = t

                        #df = pd.concat([df, df_small], axis=0)
                        small_dfs.append(df_small)
                else:
                    c2 = get_cover(g, s2, t)
                    diff = cover_difference_list(c1, c2)
                    df_small = pd.DataFrame()
                    df_small["Cover Difference"] = diff
                    df_small['Strategies'] = f"{STRATEGY_LABEL[s1]} vs. {r * 100}% {STRATEGY_LABEL[s2]}"
                    df_small["Cover Threshold"] = t

                    #df = pd.concat([df, df_small], axis=0)
                    small_dfs.append((df_small))
                i = i + 1
                print(f'Done with {i * 100 / all_s}% of strategies for current graph')

        j = j + 1
        print(f"Done with {j * 100 / all_g}% of graphs")

    print("Concatenating results")

    df = pd.concat(small_dfs)

    return df


def get_covers_as_df(graphs, strategies, ratios=[0.1], threshs=['0.001']):
    df = pd.DataFrame(columns=[PLOT_Y_LABEL['cover'], PLOT_X_LABEL['degree'], PLOT_X_LABEL['size'], PLOT_X_LABEL['diameter'], "Strategy", "Ratio", "Cover Threshold"])
    dask_df_l = dd.from_pandas(df, chunksize=4000)
    all_g = len(graphs)
    print(f"Fetching Covers for {all_g} graphs")

    small_dfs = []

    j = 0
    for g in graphs:
        print(f"Fetching Covers for {g}")

        deg = get_degrees(g)
        degrees = [float(x) for x in deg['degrees']]
        size = float(sfname(g))
        diameter = float(get_diameter(g))

        all_s = len(strategies)
        i = 0
        print(f"Fetching Covers for {all_s} Strategies")
        print(f"Strategies: {strategies}")
        for s in strategies:
            for t in threshs:
                if s in ['sqos_ot', 'sqos_ob']:
                    for r in ratios:
                        try:
                            if os.path.exists(get_full_path(g, COVER, s, r, t)):
                                cover = get_cover(g, s, t, r)
                                c = list(cover.values())
                                df_small = pd.DataFrame()
                                df_small[PLOT_Y_LABEL['cover']] = c
                                df_small[PLOT_X_LABEL['degree']] = degrees
                                df_small[PLOT_X_LABEL['size']] = size
                                df_small[PLOT_X_LABEL['diameter']] = diameter
                                df_small['Strategy'] = STRATEGY_LABEL[s]
                                df_small["Ratio"] = r
                                df_small["Cover Threshold"] = t

                                #df = pd.concat([df, df_small])
                                dask_df = dd.from_pandas(df_small, chunksize=4000)
                                dask_df_l = dd.concat([dask_df_l, dask_df])
                                #small_dfs.append(df_small)
                                #print(f'appended df_small: {df_small} for ratio r: {r}')
                        except Exception as e:
                            print(f"Error in cover assembling: {e}")
                            break
                else:
                    try:
                        if os.path.exists(get_full_path(g, COVER, s, thresh=t)):
                            cover = get_cover(g, s, t, None)
                            c = list(cover.values())
                            df_small = pd.DataFrame()
                            df_small[PLOT_Y_LABEL['cover']] = c
                            df_small[PLOT_X_LABEL['degree']] = degrees
                            df_small[PLOT_X_LABEL['size']] = size
                            df_small[PLOT_X_LABEL['diameter']] = diameter
                            df_small['Strategy'] = STRATEGY_LABEL[s]
                            df_small["Ratio"] = 1
                            df_small["Cover Threshold"] = t

                            #df = pd.concat([df, df_small])
                            dask_df = dd.from_pandas(df_small, chunksize=4000)
                            dask_df_l = dd.concat([dask_df_l, dask_df])
                            #small_dfs.append(df_small)
                    except Exception as e:
                        print(f"Error in cover assembling: {e}")
                        break

            i = i + 1
            print(f'Done with {i * 100 / all_s}% of strategies for current graph')

        j = j + 1
        print(f"Done with {j * 100 / all_g}% of graphs")
    print("Concatenating results")
    #df = pd.concat(small_dfs)
    df = dask_df_l
    return df


def get_allocs_as_df(graphs, strategies, ratios=[0.1]):
    print(ratios)
    df = pd.DataFrame(columns=[PLOT_Y_LABEL['alloc'], "Path Length", "Strategy", "Ratio"])
    dask_df_l = dd.from_pandas(df, chunksize=4000)

    all_g = len(graphs)
    print(f"Fetching Allocations for {all_g} graphs")
    small_dfs = []

    j = 0
    for g in graphs:
        print(f"Fetching Allocations for {g}")

        path_lengths = get_pl(g)

        all_s = len(strategies)
        i=0
        print(f"Fetching Allocations for {all_s} Strategies")
        for s in strategies:
            if s in ['sqos_ot', 'sqos_ob']:
                for r in ratios:
                    #print(r)
                    #print(f'fetching: {get_full_path(g, ALLOCATION, s, r)}')
                    if os.path.exists(get_full_path(g, ALLOCATION, s, r)):
                        alloc = get_allocations(g, s, r)

                        als = []
                        pls = []
                        srcs = []
                        dsts = []
                        for src, dests in alloc.items():
                            for dst in dests.keys():
                                als.append(alloc[src][dst][0])
                                pls.append(path_lengths[src][dst])
                                srcs.append(src)
                                dsts.append(dst)

                        df_small = pd.DataFrame()
                        df_small[PLOT_Y_LABEL['alloc']] = als
                        df_small["Path Length"] = pls
                        df_small['Strategy'] = STRATEGY_LABEL[s]
                        df_small["Ratio"] = r
                        df_small["Source"] = srcs
                        df_small["Dest"] = dsts

                        #df = pd.concat([df, df_small])
                        dask_df = dd.from_pandas(df_small, chunksize = 4000)
                        dask_df_l = dd.concat([dask_df_l, dask_df])
                        #small_dfs.append(df_small)

            else:

                if os.path.exists(get_full_path(g, ALLOCATION, s)):
                    alloc = get_allocations(g, s)

                    als = []
                    pls = []
                    srcs = []
                    dsts = []
                    for src, dests in alloc.items():
                        for dst in dests.keys():
                            als.append(alloc[src][dst][0])
                            pls.append(path_lengths[src][dst])
                            srcs.append(src)
                            dsts.append(dst)

                    df_small = pd.DataFrame()
                    df_small[PLOT_Y_LABEL['alloc']] = als
                    df_small["Path Length"] = pls
                    df_small['Strategy'] = STRATEGY_LABEL[s]
                    df_small["Ratio"] = 1
                    df_small["Source"] = srcs
                    df_small["Dest"] = dsts

                    #df = pd.concat([df, df_small])
                    dask_df = dd.from_pandas(df_small, chunksize=4000)
                    dask_df_l = dd.concat([dask_df_l, dask_df])
                    #small_dfs.append(df_small)

            i = i+1
            print(f'Done with {i*100/all_s}% of strategies for current graph')

        j = j+1
        print(f"Done with {j*100/all_g}% of graphs")
    print("Concatenating single results")
    #df = pd.concat(small_dfs)
    df = dask_df_l
    return df


def sfname(graph):
    size = int(graph.split('(')[1].split(')')[0])
    return size


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


def alloc_quotients_list(a1,a2):
    quots = []
    print(f"A1: {a1}")
    print(f"A2: {a2}")
    for src, dests in a1.items():
        for dst in dests.keys():
            if src in a2 .keys() and dst in a2[src].keys():
                print(f"A1 alloc at {src} to {dst}: {a1[src][dst][0]}")
                print(f"A2 alloc at {src} to {dst}: {a2[src][dst][0]}")
                quots.append(a1[src][dst][0] / a2[src][dst][0])
    return quots


def cover_difference_list(c1, c2):
    diffs = []
    for node, cover in c1.items():
        if node in c2.keys():
            diffs.append(c1[node]-c2[node])
    return diffs

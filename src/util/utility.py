import networkx as nx
import numpy as np

import numpy as np
import fnss
from src.util.utility import *
import src.util.data_handler as dh
from fnss.topologies.simplemodels import ring_topology, star_topology
from fnss.topologies.randmodels import barabasi_albert_topology, erdos_renyi_topology
import xml.etree.ElementTree as ET


from src.util.const import *
#from gma.gma_const import *
from src.types import PCList
from collections import defaultdict
from src.util.const import PRECISION
#from gma.gma_const import PRECISION


###
#
# ALLOCATION MATRIX
#
###

# int : {(int,int):{int:int}}
def dict_from_list(lst):
    dic = {}
    for el in lst:
        k = list(el.keys())[0]
        dic[k] = el[k]
    return dic


def all_allocation_matrices(G, use_bgp=False):
    allocation_matrices = {}
    i = 0
    alln = len(G.nodes())
    for cur in G.nodes():
        # print("-")
        # print(cur, G.degree(cur))
        allocation_matrices[cur] = allocation_matrix(G, cur, use_bgp=use_bgp)

        i=i+1
        if i == alln:
            print(f"{round(100 * i / alln, 4)}%")
        else:
            print(f"{round(100 * i / alln, 4)}%", end="\r")
    return allocation_matrices


def allocation_matrix(G: nx.Graph, node: int, use_bgp: bool = False,) -> np.ndarray:
    """Create the allocation matrix for a node in the graph."""

    mat = get_matrix_entries(G, node, use_bgp)
    mat, _ = normalize_matrix(G, node, mat, MAX_ROUNDS)
    if np.any(np.isnan(mat)):
        raise ValueError("Elements in the matrix are NaN.")
    return mat


def get_matrix_entries(
    G: nx.Graph, node: int, use_bgp: bool = False,
):
    """Load the entries in the allocation matrix."""
    # Neighbors sorted by index
    neigh = sorted(list(G[node]))
    n_neigh = len(neigh)
    # print(n_neigh)
    mat = np.zeros((n_neigh + 1, n_neigh + 1))  # +1 to accommodate the internal
    for idx_src, src in enumerate(neigh):
        for idx_dst, dst in enumerate(neigh):
            if src != dst:
                cap = G[src][node][BW]
                if use_bgp:
                    allow = _get_bgp_bw(G, node, src, dst, cap)
                else:
                    allow = cap
                mat[idx_dst, idx_src] = allow
    # Fill in the internal interface
    internal_bw = G.nodes[node][INT_BW]
    for idx_dst, dst in enumerate(neigh):
        # Incoming allocation
        allow = G[dst][node][BW]
        mat[n_neigh, idx_dst] = allow
        # Outgoing allocation proportional to internal interface
        mat[idx_dst, n_neigh] = internal_bw
    return mat


def _get_relationship(G, src, dst):
    """Get the BGP relationship of two adjacent nodes."""
    return G[src][dst][TYPE]


def _get_bgp_bw(G, node, src, dst, cap):
    rel_in = _get_relationship(G, src, node)
    rel_out = _get_relationship(G, node, dst)
    if rel_in == CUST or rel_out == CUST:
        return cap
    return 0


def normalize_matrix(G, node, mat, max_rounds):
    """Normalize the allocation matrix of a node.

    The normalization iteratively reduces the intf-intf allocations to the
    the point that all convergents and divergents are lower than capacity.
    """
    # TODO: check this code and test
    neigh = sorted(list(G[node]))
    cap_out = [G[node][x][BW] for x in neigh]
    cap_in = [G[x][node][BW] for x in neigh]
    # Add the internal interface
    internal_bw = G.nodes[node][INT_BW]
    cap_out.append(internal_bw)
    cap_in.append(internal_bw)
    cap_out = np.asarray(cap_out)
    cap_in = np.asarray(cap_in)
    # First normalization of the convergence always occurs
    conv = np.sum(mat, axis=1)
    mat = mat * np.minimum(cap_out[:, None] / conv[:, None], 1)
    # Check if this is enough
    i = 0
    while (
        not (
            np.all(np.less_equal(np.sum(mat, axis=0), cap_in))
            and np.all(np.less_equal(np.sum(mat, axis=1), cap_out))
        )
        and i < max_rounds
    ):
        # Normalize rows (convergence)
        convs = np.sum(mat, axis=1)
        for idx in range(convs.shape[0]):
            if convs[idx] > cap_out[idx]:
                mat[idx, :] = _normalize_array(mat[idx, :], cap_out[idx])
        # Normalize columns (divergence)
        divs = np.sum(mat, axis=0)
        for idx in range(divs.shape[0]):
            if divs[idx] > cap_in[idx]:
                mat[:, idx] = _normalize_array(mat[:, idx], cap_in[idx])
        i += 1
    return mat, i


def _normalize_array(array, target):
    """Normalizes array to target value."""
    normed = array / np.sum(array) * target
    return normed


def set_internal_cap_const(graph, capacity):
    for node in graph.nodes:
        graph.nodes[node][INT_BW] = capacity
    return graph


def set_internal_cap_max_link(graph):
    """Set the internal capacity as the maximum of the link capacities incoming."""
    for node in graph.nodes:
        max_cap = 0
        for neigh in graph.adj[node]:
            cur_cap = graph.adj[node][neigh]["capacity"]
            max_cap = max(cur_cap, max_cap)
        graph.nodes[node][INT_BW] = max_cap
    return graph


def set_internal_cap_fraction_out(graph, fraction=0.1, min_cap=0):
    """Set the internal link capacity to be a fraction of all the capacity on the outside."""
    for node in graph.nodes:
        sum_cap = 0
        for neigh in graph.adj[node]:
            cur_cap = graph.adj[node][neigh]["capacity"]
            sum_cap += cur_cap
        graph.nodes[node][INT_BW] = max(sum_cap * fraction, min_cap)
    return graph

###
#
# SIMULATION UTIL
#
###


def numpy_choice(num_samples, sample_size, elements, probabilities):
    return np.asarray([np.random.choice(elements, sample_size, p=probabilities, replace=True) for _ in range(num_samples)])


def single_shortest_path(graph, node):
    sp = nx.single_source_shortest_path(graph, node)
    return sp


def single_shortest_path_length(graph, node):
    pl = nx.single_source_shortest_path_length(graph, node)
    return pl


def all_degrees(graph):
    """Get all the node degrees of a graph."""
    deg = graph.degree()
    nodes, degrees = list(zip(*deg))
    return nodes, degrees


def all_shortest_paths(graph):
    sps = nx.all_pairs_dijkstra_path(graph)
    return dict(sps)


def all_k_shortest_paths(graph, num_sp):
    nodes = list(graph.nodes())
    ksps = defaultdict(lambda: defaultdict(list))
    for idx, src in enumerate(nodes):
        for dst in nodes[(idx + 1) :]:
            sps_gen = nx.shortest_simple_paths(graph, src, dst)
            for _ in range(num_sp):
                try:
                    next_sp = next(sps_gen)
                    ksps[src][dst].append(next_sp)
                    ksps[dst][src].append(next_sp)
                except StopIteration:
                    break
    return ksps


def alloc_to_cover(alloc, thresh, sps, is_mp):
    """Compute the _cover_ of a graph, starting from allocations."""
    cover = {}
    i = 0
    alln = len(alloc.items())

    for src, src_paths in alloc.items():
        num_covered = 0
        num_destinations = len(sps[src])
        for dst, data in src_paths.items():
            if dst in sps[src]:
                #print(data)
                if is_mp:
                    pall = data#[0]
                else:
                    pall = data[0]
                #print('THERE')
                if pall > thresh:
                    # The allocation is larger than the threshold.
                    # Update the cover.
                    num_covered += 1
        cover[src] = num_covered / num_destinations if num_destinations != 0 else 1
        i=i+1
        if i == alln:
            print(f"{round(100 * i / alln, 4)}%")
        else:
            print(f"{round(100 * i / alln, 4)}%", end="\r")
    return cover


def per_node_alloc_to_cover(src_alloc, thresh, src_sps):

    n_dest = 0
    n_cover = 0
    for dst, path in src_sps.items():
        if len(path)>1:
            n_dest = n_dest+1
            if src_alloc[dst][0] > thresh:
                n_cover = n_cover+1
    cover = n_cover/n_dest
    return cover


# Fraction of pairs reachable by each other in some graph
def alloc_to_pair_cover(alloc, thresh):

    num_pairs = 0
    num_covered = 0
    for src, src_paths in alloc.items():

        for dst, data in src_paths.items():
            pall = data
            num_pairs += 1
            if pall > thresh:
                # The allocation is larger than the threshold.
                # We can reach this node, and thus this pair should count towards the total count

                num_covered += 1
    cover = num_covered / num_pairs
    return cover


def alloc_to_reach(shortest, thresh):
    """DEPRECATED. Compute the _reach_ of shortest paths."""
    reaches = {}
    for src, src_paths in shortest.items():
        max_reach = 0
        for dst, data in src_paths.items():
            plen, pall = data
            if pall > thresh:
                # The allocation is larger than the threshold
                #   update the reach
                max_reach = max(max_reach, plen)
        reaches[src] = max_reach
    return reaches


def compute_allocations_simple(sp, nodes, deg):
    """Simplified computation of allocations.

    In the case all capacities are equal to 1.
    Works both in the case of 1-shortest-path o k-shortest-paths.
    """

    def _compute_alloc_path(path, nodes, deg):
        alw = 1
        for node in path:
            di = deg[nodes.index(node)]
            alw = alw / di
        return alw

    allocations = defaultdict(dict)
    for src, src_paths in sp.items():
        for dst, data in src_paths.items():
            if src != dst:
                # K-shortest-path case
                allocs = []
                for path in data:
                    alloc = _compute_alloc_path(path, nodes, deg)
                    alloc = np.round(alloc, decimals=PRECISION)
                    allocs.append(alloc)
                allocations[src][dst] = allocs
    return allocations


def k_alloc_to_sum(allocations):
    """Sum the k-shortest path allocations for source-destination pairs.

    This function computes the end-to-end allocation over multiple paths.
    """
    allocs_sum = defaultdict(dict)
    for src, sa in allocations.items():
        for dst, allocs in sa.items():
            asum = 0
            for pall in allocs:
                asum += pall
            allocs_sum[src][dst] = asum
    return allocs_sum


def shortest_to_diameter(shortest):
    """Compute the diameter of the network starting from shortest paths.

    This is preferred to the implementation in `networkx` as the shortest paths have
    already been computed.
    """
    i=0
    alln = len(shortest.items())
    max_l = 0
    for src, dests in shortest.items():
        for dst, path in dests.items():
            if len(path)>max_l:
                max_l = len(path)

        i=i+1
        if i == alln:
            print(f"{round(100 * i / alln, 4)}%")
        else:
            print(f"{round(100 * i / alln, 4)}%", end="\r")
    return max_l


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


def make_barabasi_albert(n_nodes, add_links, init_links, const=False):
    # Create random (barabasi albert) topology
    if const:
        name = f"c_Barabasi_Albert_{add_links}_{init_links}_({n_nodes})"
    else:
        name = f"Barabasi_Albert_{add_links}_{init_links}_({n_nodes})"
    graph = barabasi_albert_topology(n_nodes, add_links, init_links)
    fnss.set_capacities_constant(graph, 1)
    graph = set_internal_cap_const(graph, 1)
    dh.set_graph(graph, name)

    # change id types to strings
    path = dh.get_full_path(name, TOPOLOGY)
    tree = ET.parse(path)
    root = tree.getroot()

    for child in root:
        if child.tag == 'node':
            child.attrib['id.type'] = "string"
        if child.tag == "link":
            for grandchild in child:
                if grandchild.tag == "from" or grandchild.tag == "to":
                    grandchild.attrib['type'] = 'string'

    tree.write(path)

    return name


def make_erdos_reniy(n_nodes, prob):
    name = f"Erdos_Renyi_{prob}_({n_nodes})"
    graph = erdos_renyi_topology(n_nodes, prob)
    fnss.set_capacities_constant(graph, 1)
    graph = set_internal_cap_const(graph, 1)
    dh.set_graph(graph, name)

    # change id types to strings
    path = dh.get_full_path(name, TOPOLOGY)
    tree = ET.parse(path)
    root = tree.getroot()

    for child in root:
        if child.tag == 'node':
            child.attrib['id.type'] = "string"
        if child.tag == "link":
            for grandchild in child:
                if grandchild.tag == "from" or grandchild.tag == "to":
                    grandchild.attrib['type'] = 'string'

    tree.write(path)

    return name

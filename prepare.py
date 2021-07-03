"""Main runner for topology preparation."""
from argparse import ArgumentParser

import fnss

from src.multiprocessing.topology.PerTopologyOperations import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--zoo",
        action='store_true',
        help="Prepare Topology Zoo"
    )
    parser.add_argument(
        "--core",
        action='store_true',
        help="Prepare Internet Core from CAIDA"
    )
    parser.add_argument(
        "--rand",
        action='store_true',
        help="Prepare a selection of random graphs"
    )
    parser.add_argument(
        "--cw",
        action='store_true',
        help="Prepare with constant (1) link capacities (default: degree-weighted)"
    )

    args = parser.parse_args()
    return args


def main(args):

    if args.zoo:
        topologies = []
        for filename in os.listdir(TOPOLOGYZOO):
            if filename.endswith("graphml"):
                print("---")
                # Load the graph
                graph = nx.read_graphml(os.path.join(TOPOLOGYZOO, filename))
                # Check if it is fully connected
                suffix = ""
                if not nx.is_connected(graph):
                    # Keep only the largest connected component
                    print(f"Original graph size: {len(graph.nodes())}")
                    connected_nodes = max(nx.connected_components(graph), key=len)
                    graph = graph.subgraph(connected_nodes).copy()
                    suffix = "-GC"
                    print(
                        f"KEEPING only largest connected component of size: {len(graph.nodes())}"
                    )
                # Save at xml
                name = filename.rsplit(".", maxsplit=1)[0]
                name = f"{name}{suffix}({len(graph.nodes())})"
                print(name)

                topo = os.path.join(DATA_PATH, name)
                if not os.path.exists(topo):
                    os.mkdir(topo)

                graph_path = os.path.join(topo, 'graph/')
                if not os.path.exists(graph_path):
                    os.mkdir(graph_path)

                graph = fnss.Topology(graph)
                fnss.write_topology(graph, os.path.join(graph_path, "topology.xml"))
                topologies.append(name)

        if args.cw:
            proc = AddConstantCapacity(topologies, 1, 1)
            proc.run()
        else:
            proc = AddDegreeGravityCapacity(topologies, CAPACITY_INTERVALS, 1)
            proc.run()


    if args.core:

        print('starting bidirectional caida')

        topo_bidir = fnss.parse_caida_as_relationships(CAIDA)

        # Create the core
        print(f">>> Creating core <<<")

        print(f"In the topology, there are {len(topo_bidir.nodes)} ASes with {len(topo_bidir.edges)} links")
        degrees = [topo_bidir.degree(x) for x in topo_bidir.nodes]
        print(f"The average degree of a node is {np.average(degrees):.3f}")

        print('Remove Stubs')

        # Remove all the stubs from the topology recursively
        # Until no "just customer nodes" remain
        n_nodes = np.inf
        while n_nodes != len(topo_bidir.nodes):
            n_nodes = len(topo_bidir.nodes)
            for node in list(topo_bidir.nodes):
                if topo_bidir[node] == {}:
                    topo_bidir.remove_node(node)

        print('With stubs removed and sampling')
        print(f"In the topology, there are {len(topo_bidir.nodes)} ASes with {len(topo_bidir.edges)} links")
        degrees = [topo_bidir.degree(x) for x in topo_bidir.nodes]
        print(f"The average degree of a node is {np.average(degrees):.3f}")

        print("--- core creating completed ---")

        # Rename the nodes to have them in order [0, # nodes)
        nodes = list(topo_bidir.nodes())
        relabeling = np.arange(len(nodes))
        labelmap = dict(zip(nodes, relabeling))
        topo_bidir = nx.relabel_nodes(topo_bidir, labelmap)

        print('Adding capacities')
        # Add capactiy to the links and nodes
        if args.cw:
            fnss.set_capacities_constant(topo_bidir, 1, capacity_unit=UNIT)
            topo_bidir = set_internal_cap_const(topo_bidir, 1)
        else:
            fnss.set_capacities_degree_gravity(topo_bidir, CAPACITY_INTERVALS, capacity_unit=UNIT)
            topo_bidir = set_internal_cap_max_link(topo_bidir)

        # Add reverse edges with inverted business relation
        for s, e in topo_bidir.edges():
            rel = topo_bidir[s][e]["type"]
            cap = topo_bidir[s][e][BW]
            if rel == "peer":
                topo_bidir.add_edge(e, s, capacity=cap, type="peer")
            elif rel == "customer":
                topo_bidir.add_edge(e, s, capacity=cap, type="provider")
            elif rel == "provider":
                topo_bidir.add_edge(e, s, capacity=cap, type="customer")
            else:
                raise ValueError()

        # Save like other topos
        dh.set_graph(topo_bidir, 'Core')

    if args.rand:
        rand_ts = []
        for adl in RAND_ADD_LINKS:
            for init_n in RAND_INIT_NODES:
                name = make_barabasi_albert(RAND_N_NODES, adl, init_n)
                rand_ts.append(name)

        for p in RAND_P_LINK_CREATE:
            name = make_erdos_reniy(RAND_N_NODES, p)
            rand_ts.append(name)

        name = make_barabasi_albert(2500, 20, 50)
        rand_ts.append(name)

        if args.cw:
            proc = AddConstantCapacity(rand_ts, 1, 1)
            proc.run()
        else:
            proc = AddDegreeGravityCapacity(rand_ts, CAPACITY_INTERVALS, 1)
            proc.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)
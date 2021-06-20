import os
from argparse import ArgumentParser

import fnss
import networkx as nx
from src.util.const import CAPACITY_INTERVALS
import src.util.data_handler as dh
from src.multiprocessing.topology.PerTopologyOperations import AddConstantCapacity, AddDegreeGravityCapacity

DATA = "dat/topologyzoo"
OUT = "dat/topologies"

for filename in os.listdir(DATA):
    if filename.endswith("graphml"):
        print("---")
        # Load the graph
        graph = nx.read_graphml(os.path.join(DATA, filename))
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
        topo = os.path.join(OUT, name)
        if not os.path.exists(topo):
            os.mkdir(topo)

        graph_path = os.path.join(topo, 'graph/')
        if not os.path.exists(graph_path):
            os.mkdir(graph_path)

        graph = fnss.Topology(graph)
        fnss.write_topology(graph, os.path.join(graph_path, "topology.xml"))

        proc = AddDegreeGravityCapacity(name, CAPACITY_INTERVALS, 1)
        proc.run()

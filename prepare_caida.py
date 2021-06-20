import numpy as np
import fnss
from src.util.const import UNIT, CAIDA
from src.util.utility import *
import src.util.data_handler as dh

CAPACITY_INTERVALS = np.linspace(400, 1000, 100)

# Load CAIDA topology


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

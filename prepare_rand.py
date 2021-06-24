import numpy as np
import fnss
from src.util.const import UNIT, CAIDA
from src.util.utility import *
import src.util.data_handler as dh
from fnss.topologies.simplemodels import ring_topology, star_topology
from fnss.topologies.randmodels import barabasi_albert_topology, erdos_renyi_topology
import xml.etree.ElementTree as ET
from src.multiprocessing.topology.PerTopologyOperations import AddDegreeGravityCapacity

_ADD_LINKS = [
    1,
    2,
    5,
    10,
    15
]
_INIT_NODES = [
    20,
    25,
    50,
    75,
    100
]
P_LINK_CREATE = [
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]

_N_NODES = 500

def make_barabasi_albert(n_nodes, add_links, init_links):
    # Create random (barabasi albert) topology
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

    proc = AddDegreeGravityCapacity([name], CAPACITY_INTERVALS, 1)
    proc.run()


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

    proc = AddDegreeGravityCapacity([name], CAPACITY_INTERVALS, 1)
    proc.run()
'''
for adl in _ADD_LINKS:
    for init_n in _INIT_NODES:
        make_barabasi_albert(_N_NODES, adl, init_n)

for p in P_LINK_CREATE:
    make_erdos_reniy(_N_NODES, p)
'''
make_barabasi_albert(2500, 20, 50)
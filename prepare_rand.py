import numpy as np
import fnss
from src.util.const import UNIT, CAIDA
from src.util.utility import *
import src.util.data_handler as dh
from fnss.topologies.simplemodels import ring_topology, star_topology
from fnss.topologies.randmodels import barabasi_albert_topology, erdos_renyi_topology
import xml.etree.ElementTree as ET
from src.multiprocessing.topology.PerTopologyOperations import AddDegreeGravityCapacity


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

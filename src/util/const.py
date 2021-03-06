"""
Constants
"""
import os

###################################################
# GENERAL
###################################################
CAIDA = "dat/CAIDA/20210601.as-rel.txt"
COVER_THRESHOLD = 1e-6
UNIT = "Gbps"

VERBOSITY = 2
###################################################
###################################################


###################################################
# Data
###################################################
DATA_PATH_REMOTE = '/data2/jmohler/dat/topologies/'
DATA_PATH = os.path.join(os.getcwd(), 'dat/topologies/')
FIGURE_PATH = os.path.join(os.getcwd(), 'dat/figures/')
TOPOLOGYZOO = "dat/topologyzoo"

# DATA TYPE CONSTANTS
ALLOCATION = 0
SHORTEST_PATH = 1
ALLOCATION_MATRIX = 2
PATH_LENGTHS = 3
PATH_COUNTS = 4
TOPOLOGY = 5
DIAMETER = 6
COMPARISON = 7
COVER = 8
DEGREE = 9
FIGURE = 10
COVER_IMPROVEMENT = 11

# DATA TYPE - FILE TYPE MAP
FILE_TYPE = [
    'json',
    'json',
    'pkl',
    'json',
    'pkl',
    'xml',
    'txt',
    'json',
    'json',
    'json',
    'json',
    'json'
]
###################################################
###################################################


###################################################
# SIMULATION
###################################################

# AS business relationships
PEER = "peer"
CUST = "customer"
PROV = "provider"
TYPE = "type"

# Capacity
BW = "capacity"
INT_BW = "internal_capacity"  # Internal bandwidth of a node
MAX_ROUNDS = 10

# Precision with which to compute the allocations
PRECISION = 12

CAPACITY_INTERVALS = [10, 40, 100]
###################################################
###################################################


###################################################
# Plotting
###################################################
# PLOT_FORMAT = 'pdf'
PLOT_FORMAT = 'png'

MATPLOTLIB_BACKEND = 'Agg'
# MATPLOTLIB_BACKEND = 'Qt5Cairo'

STRATEGIES = [
    'GMAImproved',
    'sqos_ot',
    'sqos_pt',
    'sqos_ob',
    'sqos_pb'
]
THRESH_LABEL = {
    0.01 : '10Mbps',
    0.001 : '1Mbps',
    0.0001 : '100Kbps',
    0.00001 : '10Kbps',
    0.000001 : '1Kbps'
}

PLOT_X_LABEL = {
    'degree': 'Node Degree',
    'size': 'Graph Size',
    'diameter': 'Graph Diameter',
    'pl': 'Path Length'
}

PLOT_Y_LABEL = {
    'cover': 'Cover [%]',
    'cover_d': 'Cover Difference',
    'alloc': f'Allocations {UNIT}',
    'alloc_d': f'Allocation Difference [Gbps]'
}

STRATEGY_LABEL = {
    'GMAImproved': 'GMA',
    'sqos_ot': 'OMA\nMax',
    'sqos_pt': 'PMA\nMax',
    'sqos_ob': 'OMA\nConcurrent',
    'sqos_pb': 'PMA\nConcurrent'
}
STRAT_ORDER = ['GMA', 'OMA\nMax', 'PMA\nMax', 'OMA\nConcurrent', 'PMA\nConcurrent']
PALETTE = "Set1"

PLOT_THRESH = 1e-3
###################################################
###################################################

###################################################
# RANDOM GRAPH
###################################################
RAND_ADD_LINKS = [
    1,
    2,
    5,
    10,
    15
]
RAND_INIT_NODES = [
    20,
    25,
    50,
    75,
    100
]
RAND_P_LINK_CREATE = [
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

RAND_N_NODES = [
    150,
    200,
    250,
    500,
    750,
    1000,
    1250,
    1500,
    2000
]
###################################################
###################################################

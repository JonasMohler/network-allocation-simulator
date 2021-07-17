import os
from src.multiprocessing.topology.PerTopologyOperations import AddConstantCapacity
from subprocess import call

def cp_dir(source, target):
    call(['cp', '-a', source, target])

TOPO = '/data2/jmohler/dat/topologies'

for dir in os.listdir(TOPO):
    if dir.startswith('Barabasi'):
        cp_dir(dir, f"c_{dir}")

tops = []
for dir in os.listdir(TOPO):
    if dir.startswith('c_Barabasi'):
        tops.append(dir)

proc = AddConstantCapacity(tops, 70, 15)
proc.run()


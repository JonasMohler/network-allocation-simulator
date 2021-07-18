import os
# TOPO = 'dat/topologies'
# TOPO = 'dat/t_temp'
TOPO = '/data2/jmohler/dat/topologies/'

cwd = os.getcwd()
for dir in os.listdir(TOPO):
    for i in ['a0.1', 'a0.2', 'a0.3', 'a0.4', 'a0.5', 'a0.6', 'a0.7', 'a0.8', 'a0.9']:
        samp = os.path.join(TOPO ,os.path.join(dir,f'graph/{i}_shortest_paths.json'))
        count = os.path.join(TOPO ,os.path.join(dir, f'graph/{i}_path_counts.pkl'))
        os.remove(samp)
        os.remove(count)

        '''
    if os.path.exists(pl):
        pln = os.path.join(cwd,os.path.join(TOPO ,os.path.join(dir, 'graph/path_lengths.json')))
        os.rename(pl, pln)
        '''
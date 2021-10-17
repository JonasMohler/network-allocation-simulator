#Running a simulation

##Preparation

Prepare the wanted topologies using the prepare.py file.

Use --zoo for topologyzoo graphs, --rand for a selection of random graphs, --core for the internet core model. Default link capacities are degree weighted. Use --cw to use constant link weights.

##Simulation
After preparation run a simulation using the simulate.py file.

Pass a list of topologies to run the simulation on using -d (i.e. simulate.py -d Core Aanet(15) Eenet(13)).

Use --all to run all steps of the simulation, or run individual steps using respective flags (i.e. simulate.py -d Eenet(13) --all will run the full simulation, including covers for the Eenet topology, simulate.py -d Eenet(13) --sp will only calculate shortest paths for Eenet)
Use -h to see all options

Specify the out directory using -o, sampling ration using -r, --thrs for the cover threshold, ...
Use -h to get more configuration details.

##Aggregation

Use the aggregates.py file to calculate aggregates over simulations over multiple topologies.

Specify the topologies to consider using -d, shortest paths to consider using --num_sp, --r for ratios, --out for specifying the out directory.

##Plotting

Use plot.py to create a number of predefined plots. Make sure the DATA_PATH variable in const.py points to the correct data location for the simulations to consider.

Use plot_aggs.py to create a number of predefined plots over previously calculated aggregates.
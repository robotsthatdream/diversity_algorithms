#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    18/03/2020
""" 

import numpy as np

import sys, os

from diversity_algorithms.experiments.exp_utils import RunParam, analyze_params
from diversity_algorithms.analysis.population_analysis import get_coverage, build_grid, update_grid, coverage, exploration_reachable_uniformity



#from scoop import futures

params_commandline = {
    "rundir": RunParam("d", "", "Directory of the run"),
    "maxgen": RunParam("g", -1, "max generation to look up to (default will stop when nothing is found)"),
    "archive_pattern" : RunParam("", "archive_small_novelty_fit_bd_id_parent_id_gen%d.npz", "Archive data files pattern"),
    "outfile": RunParam("","archive_stats.npz", "Output file"),
    "force_grid_recompute": RunParam("f",0, "If this is true, the grid coverage will always be recomputed. If this is false and the archive is already a grid, it will just use the archive grid directly. (Default false, only useful to set to true if your MAP-elites grid and your analysis grid are different.)"),
}

analyze_params(params_commandline, sys.argv)


max_gen = params_commandline["maxgen"].get_value()

if(max_gen < 1):
    max_gen = np.inf

rundir = params_commandline["rundir"].get_value()
archive_pattern = params_commandline["archive_pattern"].get_value()
outfile = params_commandline["outfile"].get_value()
force_grid_recompute=bool(params_commandline["force_grid_recompute"].get_value())

if(rundir==""):
    print("Please specify rundir")
    sys.exit(1)

print("Reading environment parameters")

try:
    params_xp = dict(np.load(os.path.join(rundir, "params.npz"), allow_pickle=True))
except FileNotFoundError:
    print("Can't find param file -_-")
    sys.exit(1)

nb_bin=int(params_xp["nb_bin"])
min_x=params_xp["min_bd"]
max_x=params_xp["max_bd"]
env_name=str(params_xp["env_name"])
archive_type=str(params_xp["archive_type"])


def get_points_from_npzarchive(archive):
    points = list()
    size = int(archive['size'])
    for idx in range(size):
        points.append(tuple(archive["bd_%d" % idx]))
    return points


def grid_from_archive(points,  min_x, max_x, nb_bin):
    grid = build_grid(min_x, max_x, nb_bin)
    update_grid(grid,min_x, max_x, points)
    cov = coverage(grid)
    return (grid, cov)



def stat_grid_coverage(archive, force=force_grid_recompute):
    if((archive_type=='grid') and not force): # Just use the MAP elites grid
        n_bins_total = np.power(nb_bin, len(min_x))
        print("b")
        return archive["size"]/float(n_bins_total)
    else:
        points = get_points_from_npzarchive(archive)
        grid, cov = grid_from_archive(points,  min_x, max_x, nb_bin)
        return cov

def stat_mean_fitness(archive):
    return np.mean([archive["fit_%d" % i] for i in range(archive["size"])])

def stat_std_fitness(archive):
    return np.std([archive["fit_%d" % i] for i in range(archive["size"])])

def stat_mean_nov(archive):
    return np.mean([archive["novelty_%d" % i] for i in range(archive["size"])])

def stat_std_nov(archive):
    return np.std([archive["novelty_%d" % i] for i in range(archive["size"])])



statistics_per_archive = {
    "coverage":stat_grid_coverage,
    "mean_fit":stat_mean_fitness,
    "std_fit":stat_std_fitness,
    "mean_nov":stat_mean_nov,
    "std_nov":stat_std_nov
    }




def get_stats(rundir, max_gen):
    output_data = dict()
    gen = 1 # start at gen 1
    for k in statistics_per_archive.keys():
        output_data[k] = list()
    while gen < max_gen:
        try:
            archive = np.load(os.path.join(rundir, archive_pattern % gen), allow_pickle=True)
        except FileNotFoundError:
            print("Could not find population file for gen %d, terminating." % gen)
            break
        print("Processing for gen %d..." % gen)
        for (key,func) in statistics_per_archive.items():
            output_data[key].append(func(archive))
        gen += 1
    return output_data
        
        
        
        
if(__name__=='__main__'):
    outdata = get_stats(rundir, max_gen)
    outfile_path = os.path.join(rundir,outfile) 
    np.savez(outfile_path, **outdata)
    print("Coverage and rerachable uniformity data saved to %s" % outfile)






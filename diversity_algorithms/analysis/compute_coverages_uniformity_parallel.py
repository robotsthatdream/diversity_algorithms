#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    20/01/2020
""" 

import numpy as np
import sys, os

from diversity_algorithms.experiments.exp_utils import RunParam, analyze_params
import diversity_algorithms.analysis.data_utils as du
from diversity_algorithms.analysis.population_analysis import get_coverage, build_grid, update_grid, coverage, exploration_reachable_uniformity




from scoop import futures

params_commandline = {
    "rundir": RunParam("d", "", "Directory of the run"),
    "gen": RunParam("g", 0, "Generation to look up to"),
    "evofile_pattern" : RunParam("", "evolvability_ind%d_bd_gen%d.npz", "Evolvabilitty data files pattern"),
    "n_indivs": RunParam("", -1, "Number of indivs for which evolvability was generated (default = pop size)"),
    "outfile_pattern": RunParam("","coverage_uniformity_gen%d.npz", "Output file pattern"),
    "verbosity": RunParam("v",0, "Verbosity level"),
    "allow_partial" : RunParam("p",0, "If not enough evofiles are found, should we compute with the present files ? (default no)")
}

analyze_params(params_commandline, sys.argv)


gen = params_commandline["gen"].get_value()
rundir = params_commandline["rundir"].get_value()
evofile_pattern = params_commandline["evofile_pattern"].get_value()
n_indivs = params_commandline["n_indivs"].get_value()
outfile_pattern = params_commandline["outfile_pattern"].get_value()
verbose=bool(params_commandline["verbosity"].get_value())
allow_partial=bool(params_commandline["allow_partial"].get_value())

if(gen==0 or rundir==""):
	print("Please specify generation and rundir")
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

if(n_indivs < 0):
	n_indivs = int(params_xp["pop_size"])


print("Checking evolvability files")
evofiles = list()

for i in range(n_indivs):
	filepath = os.path.join(rundir,evofile_pattern % (i, gen))
	if(os.path.isfile(filepath)):
		evofiles.append(filepath)
	else:
		print("WARNING: could not find '%s'" % filepath)

if not evofiles:
	print("No evofiles found. Exiting.")
	sys.exit(1)


def grid_from_file(file,  min_x, max_x, nb_bin, verbose=False):
	if(verbose):
		print("Computing grid for '%s'..." % file, end='')
	points=du.get_points_from_genfile(file)
	grid = build_grid(min_x, max_x, nb_bin)
	update_grid(grid,min_x, max_x, points)
	cov = coverage(grid)
	unif = exploration_reachable_uniformity(grid)
	print(".", end='', flush=True)
	return (grid, cov, unif)


if(__name__=='__main__'):
	if not evofiles:
		print("No files found. Exiting.")
		sys.exit(0)
	if((not allow_partial) and (len(evofiles)<n_indivs)):
		print("Not enough files found. Exiting.")
		sys.exit(0)
	print("%d evofiles to process" % len(evofiles))
	print("Getting grids and individual coverages and uniformities...")
	indiv_data = list(futures.map(lambda f: grid_from_file(f, min_x, max_x, nb_bin, verbose), evofiles))
	indiv_grids, indiv_coverages, indiv_unifs = zip(*indiv_data)
	print("Reducing...")
	sum_grid = np.sum(indiv_grids, axis=0)
	print("Computing global coverage and uniformity...")
	gcov = coverage(sum_grid)
	gunif = exploration_reachable_uniformity(sum_grid)
	print("Done ! Global coverage is %f and global reachable uniformity is %f" % (gcov, gunif))
	print("Saving data...")
	outfile = os.path.join(rundir,outfile_pattern % gen) 
	np.savez(outfile, global_coverage=gcov, global_uniformity=gunif, indiv_coverages=np.array(indiv_coverages), indiv_uniformities=np.array(indiv_unifs))
	print("Coverage and rerachable uniformity data saved to %s" % outfile)

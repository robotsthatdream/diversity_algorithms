
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

import gym, gym_fastsim

from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 

from deap import creator, base

import dill
import pickle
import math

import sys

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.quality_diversity import set_creator
set_creator(creator)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax)
#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.quality_diversity import QDEa
from diversity_algorithms.algorithms.utils import *

from diversity_algorithms.experiments.exp_utils import *

# =====


with_scoop=True

if with_scoop:
	from scoop import futures



with_scoop=True

if with_scoop:
	from scoop import futures


# declaration of params: RunParam(short_name (single letter for call from command line), default_value, doc)
params={
	"run_dir_name": RunParam("R", "", "name of the dir in which to put the dir with the run files"),
	"verbosity": RunParam("v", "none", "verbosity level (all, none or module specific values"),
	"pop_size": RunParam("p", 100, "population size (number of indiv sampled)"),
	"n_add": RunParam("n", -1, "number of children to try to add to the archive (equivalent to lambdaNov for NS), default =pop_size"),
	"initial_seed_size": RunParam("i", 100, "number of random individuals to generate and try to add to the archive to seed it at the beginning"),
	"variant": RunParam("a", "QD", "variant of the QD algorithm"),
	"archive_type" : RunParam("A", "grid", "Archive type (grid or unstructured)"),
	"grid_n_bin" : RunParam("", -1, "Number of bins per dimension for grid archive (default auto = environment default)"),
	"unstructured_neighborhood_radius" : RunParam("", -1., "Replace radius for unstructured archive (default = half default grid size)"),
	"replace_strategy": RunParam("r", "fitness", "strategy for archive replacement (always, never, random, fitness, novelty or disttoparent)"),
	"sample_strategy": RunParam("s", "random", "strategy for sampling the archive (random or novelty)"),
	"kdtree_update": RunParam("K", "default", "Is the KD-tree updated every time an indiv has been added (immediate) or only once per gen (delayed) ? Default is delayed for structured archives and immediate for unstructured."),
	"env_name": RunParam("e", "Fastsim-LS2011", "Environment name"),
	"nb_gen":   RunParam("g", 100, "number of generations"),
	"dump_period_evolvability": RunParam("V", 0, "period of evolvability estimation"),
	"extra_evolvability_gens": RunParam("x", "", "coma-separated lists of extra periods when to estimate evolvability"),
	"dump_period_offspring": RunParam("o", 1, "period of behavior descriptor dump"),
	"dump_period_population": RunParam("d", 1, "period of population dump"),
	"dump_period_archive_full": RunParam("D", 100, "period of a full archive dump (with full genotype)"),
	"dump_period_archive_small": RunParam("Q", 1, "period of minor archive dump (just BDs, ID and novelty)"),
	"cxpb": RunParam("", 0., "cross-over rate"), # No crossover
	"mutpb": RunParam("",1., "mutation rate"),  # All offspring are mutated...
	"indpb": RunParam("",0.1, "indiv probability"), # ...but only 10% of parameters are mutated
	"eta_m": RunParam("", 15.0, "Eta parameter for polynomial mutation"),
	"min": RunParam("", -5., "Min value of the genotype"), # WARNING, some variants do not use it at all. -5 seems reasonable for NN weights
	"max": RunParam("", 5., "Min value of the genotype"), # WARNING, some variants do not use it at all. 5 seems reasonable for NN weights
	"k_nov": RunParam("", 15, "Number of neighbors to take into account for novelty computation"),
	"geno_type": RunParam("G", "realarray", "type of genotype (either realarray or dnn)"),
	"eval_budget": RunParam("B", -1, "evaluation budget (ignored if -1). "),
	}

analyze_params(params, sys.argv)




# Controller definition :
# Parameters of the neural net
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
# Create a dict with all the properties of the controller
controller_params = {"controller_type":SimpleNeuralController,"controller_params":nnparams}

# Get environment
eval_func = create_functor(params, controller_params)

# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_func(g)


# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
	# Get env and controller

	sparams, pool=preparing_run(eval_func, params, with_scoop)
	

	archive, logbook, nb_eval = QDEa(eval_with_functor, sparams, pool)

	terminating_run(sparams, None, archive, logbook, nb_eval)


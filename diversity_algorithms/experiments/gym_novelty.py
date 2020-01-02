
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


from diversity_algorithms.algorithms.novelty_search import set_creator


#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.novelty_search import novelty_ea
from diversity_algorithms.algorithms.utils import *

from diversity_algorithms.experiments.exp_utils import *

# =====


with_scoop=True

if with_scoop:
	from scoop import futures


# declaration of params: RunParam(short_name (single letter for call from command line), default_value, doc)
params={
	"run_dir_name": RunParam("R", "", "name of the dir in which to put the dir with the run files"),
	"verbosity": RunParam("v", "none", "verbosity level (all, none or module specific values"),
	"pop_size": RunParam("p", 100, "population size (mu)"),
	"lambda": RunParam("l", 2., "Number of offspring generated (coeff on pop_size)"),
	"env_name": RunParam("e", "Fastsim-LS2011", "Environment name"),
	"nb_gen":   RunParam("g", 100, "number of generations"),
	"dump_period_evolvability": RunParam("V", 100, "period of evolvability estimation"),
	"dump_period_bd": RunParam("b", 1, "period of behavior descriptor dump"),
	"dump_period_population": RunParam("d", 1, "period of population dump"),
	"dump_period_archive": RunParam("D", 1, "period of archive dump"),
	"variant": RunParam("a", "NS", "variant of the Novelty Search algorithm"),
	"cxpb": RunParam("", 0, "cross-over rate"), # No crossover
	"mutpb": RunParam("",1., "mutation rate"),  # All offspring are mutated...
	"indpb": RunParam("",0.1, "indiv probability"), # ...but only 10% of parameters are mutated
	"eta_m": RunParam("", 15.0, "Eta parameter for polynomial mutation"),
	"min": RunParam("", -5., "Min value of the genotype"), # WARNING, some variants do not use it at all. -5 seems reasonable for NN weights
	"max": RunParam("", 5., "Min value of the genotype"), # WARNING, some variants do not use it at all. 5 seems reasonable for NN weights
	"k": RunParam("", 15, "Number of neighbors to take into account for novelty computation"),
	"add_strategy": RunParam("s", "random", "strategy for archive inclusion (random or novel)"),
	"lambda_nov": RunParam("", 6, "number of indiv added to the archive at each gen"),
	"geno_type": RunParam("G", "realarray", "type of genotype (either realarray or dnn)"),
	"eval_budget": RunParam("B", -1, "evaluation budget (ignored if -1). "),
	"alphas": RunParam("A", 1.1, "alpha for the computation of alpha shapes, to build explored areas and compute distance to them"),
	}

analyze_params(params, sys.argv)


# Controller definition :
# Parameters of the neural net
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
# Create a dict with all the properties of the controller
controller_params = {"controller_type":SimpleNeuralController,"controller_params":nnparams}

# Get environment
eval_func = create_functor(params, controller_params)

nbobj=params["variant"].get_value().count("+")+1
creator.create("FitnessMax", base.Fitness, weights=(1.0,)*nbobj)
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax)
set_creator(creator)


# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_func(g)

# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
	# Get env and controller

	sparams, pool=preparing_run(eval_func, params, with_scoop)

	

	pop, archive, logbook, nb_eval = novelty_ea(eval_with_functor, sparams, pool)

	terminating_run(sparams, pop, archive, logbook, nb_eval)


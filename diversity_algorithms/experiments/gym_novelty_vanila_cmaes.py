
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

import gym, gym_fastsim

from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 
from diversity_algorithms.algorithms.cmaes import cmaes, with_scoop

from deap import creator, base

import dill
import pickle
import math



import cma
from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.novelty_management import *
from diversity_algorithms.experiments.exp_utils import *






# declaration of params: RunParam(short_name (single letter for call from command line), default_value, doc)
params={
	"verbosity": RunParam("v", "none", "verbosity level (all, none or module specific values"),
	"env_name": RunParam("e", "Fastsim-LS2011", "Environment name"),
	"dump_period_evolvability": RunParam("V", 100, "period of evolvability estimation"),
	"dump_period_bd": RunParam("b", 1, "period of behavior descriptor dump"),
	"dump_period_population": RunParam("d", 1, "period of population dump"),
	"dump_period_archive": RunParam("D", 1, "period of archive dump"),
	"variant": RunParam("a", "CMAES_NS", "variant of the CMAES algorithm"),
	"min": RunParam("", -5., "Min value of the genotype"), # WARNING, some variants do not use it at all. -5 seems reasonable for NN weights
	"max": RunParam("", 5., "Min value of the genotype"), # WARNING, some variants do not use it at all. 5 seems reasonable for NN weights
	"k": RunParam("", 15, "Number of neighbors to take into account for novelty computation"),
	"add_strategy": RunParam("s", "random", "strategy for archive inclusion (random or novel)"),
	"lambda_nov": RunParam("", 6, "number of indiv added to the archive at each gen"),
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
	sparams, pool=preparing_run(eval_func, params, with_scoop, deap=False)
	
	if (sparams["variant"] not in ["CMAES_NS", "CMAES_DM", "CMAES_NS_mu1"]):
		print("Invalid variant: "+variant)
	esresult, archive, nb_eval = cmaes(eval_with_functor, sparams, pool)
	terminating_run(sparams, None, archive, None, nb_eval)


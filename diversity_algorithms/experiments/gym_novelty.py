
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

import gym, gym_fastsim

from diversity_algorithms.environments import EvaluationFunctor
from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 

from diversity_algorithms.algorithms import grid_features

from deap import creator, base

import dill
import pickle
import math

import sys

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.novelty_search import set_creator
set_creator(creator)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax)
#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.novelty_search import novelty_ea
from diversity_algorithms.algorithms.utils import *

from diversity_algorithms.experiments.exp_utils import *

# =====


with_scoop=True

if with_scoop:
	from scoop import futures


env_name=None
# Each worker gets a functor
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
#env, controller = generate_gym_env_and_controller(params=nnparams)
eval_gym = EvaluationFunctor(controller_type=SimpleNeuralController,controller_params=nnparams,get_behavior_descriptor='auto')

# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_gym(g)



# declaration of params: RunParam(short_name (single letter for call from command line), default_value, doc)
params={
	"verbosity": RunParam("v", "none", "verbosity level (all, none or module specific values"),
	"pop_size": RunParam("p", 100, "population size (mu)"),
	"lambda": RunParam("l", 2., "Number of offspring generated (coeff on pop_size)"),
	"env_name": RunParam("e", "FastsimSimpleNavigation-v0", "gym environment name"),
	"nb_gen":   RunParam("g", 100, "number of generations"),
	"evolvability_period": RunParam("V", 100, "period of evolvability estimation"),
	"dump_period_bd": RunParam("b", 1, "period of behavior descriptor dump"),
	"dump_period_pop": RunParam("d", 1, "period of population dump"),
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
	"geno_type": RunParam("G", "realarray", "type of genotype (either realarray or dnn)")
	}

analyze_params(params, sys.argv)
	
eval_gym.set_env(None,params["env_name"].get_value(), with_bd=True)


# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
	# Get env and controller

	# Dumping how the run has been launched
	run_name=generate_exp_name(params["env_name"].get_value()+"_"+params["variant"].get_value())
	print("Saving logs in "+run_name)
	dump_exp_details(sys.argv,run_name, params)

	# Completing the parameters (and putting them in a simple dict for future use)
	sparams=get_simple_params_dict(params)

	if (sparams["env_name"] in grid_features.keys()):
	        min_bd=grid_features[sparams["env_name"]]["min_x"]
	        max_bd=grid_features[sparams["env_name"]]["max_x"]
	        nb_bin_bd=grid_features[sparams["env_name"]]["nb_bin"]

	        grid=build_grid(min_bd, max_bd, nb_bin_bd)
	        grid_offspring=build_grid(min_bd, max_bd, nb_bin_bd)
	        stats=None
	        stats_offspring=None
	        nbc=nb_bin_bd**2
	        nbs=nbc*2 # min 2 samples per bin
	        evolvability_nb_samples=nbs
	else:
                grid=None
                grid_offspring=None
                min_bd=None
                max_bd=None
                nb_bin_bd=None
                evolvability_nb_samples=0
                nbs=0

	sparams["ind_size"]=eval_gym.controller.n_weights
	
	sparams["evolvability_nb_samples"]=evolvability_nb_samples
	sparams["min_bd"]=min_bd # not used by NS. It is just to keep track of it in the saved param file
	sparams["max_bd"]=max_bd # not used by NS. It is just to keep track of it in the saved param file

	# We use a different window size to compute statistics in order to have the same number of points for population and offspring statistics
	window_population=nbs/sparams["pop_size"]
	window_offspring=nbs/(sparams["lambda"]*sparams["pop_size"])
	
	if (sparams["evolvability_period"]>0) and (evolvability_nb_samples>0):
		stats=get_stat_fit_nov_cov(grid,prefix="population_",indiv=True,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_population)
		stats_offspring=get_stat_fit_nov_cov(grid_offspring,prefix="offspring_",indiv=True,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_offspring)
	else:
		stats=get_stat_fit_nov_cov(grid,prefix="population_",indiv=False,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_population)
		stats_offspring=get_stat_fit_nov_cov(grid_offspring,prefix="offspring_", indiv=False,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_offspring)

	sparams["stats"] = stats # Statistics
	sparams["stats_offspring"] = stats_offspring # Statistics on offspring
	sparams["window_population"]=window_population
	sparams["window_offspring"]=window_offspring
	sparams["run_name"]=run_name
	
	print("Launching Novelty Search with the following parameter values:")
	for k in sparams.keys():
		print("\t"+k+": "+str(sparams[k]))
	if (grid is None):
                print("WARNING: grid features have not been defined for env "+sparams["env_name"]+". This will have no impact on the run, except that the coverage statistic has been turned off")
	if (sparams["evolvability_period"]>0) and (evolvability_nb_samples>0):
		print("WARNING, evolvability_nb_samples>0. The run will last much longer...")

	if with_scoop:
		pool=futures
	else:
		pool=None
		
	dump_params(sparams,run_name)

	pop, archive, logbook = novelty_ea(eval_with_functor, sparams, pool)

	dump_pop(pop,sparams["nb_gen"],sparams["run_name"])
	dump_logbook(logbook,sparams["nb_gen"],sparams["run_name"])
	dump_archive(archive,sparams["nb_gen"],sparams["run_name"])
		
	dump_end_of_exp(run_name)
	
	print("The population, log, archives, etc have been dumped in: "+run_name)
	


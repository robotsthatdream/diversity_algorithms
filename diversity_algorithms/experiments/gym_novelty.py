
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

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.novelty_search import set_creator
set_creator(creator)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.novelty_search import NovES
from diversity_algorithms.algorithms.utils import *
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



def launch_nov(env_name, pop_size, nb_gen, evolvability_period=0, dump_period_pop=10, dump_period_bd=1):
	"""Launch a novelty search run on the maze
	
	Launch a novelty search run on the maze:
	:param pop_size: population size
	:param nb_gen: number of generations to compute
	:param evolvability_nb_samples: number of samples to estimate the evolvability of each individual in the population
	:param evolvability_period: period of the evolvability estimation
	:param dump_period_pop: period of populatin dump
	:param dump_period_bd: period of behavior descriptors dump	

	WARNING: the evolvability requires to generate and evaluate pop_size*evolvability_nb_samples just for statistics purposes, it will significantly slow down the process.
	"""
#	if (env_name not in grid_features.keys()):
#                print("You need to define the features of the grid to be used to track behavior descriptor coverage in algorithms/__init__.py")
#                return None, None

	if (env_name in grid_features.keys()):
	        min_x=grid_features[env_name]["min_x"]
	        max_x=grid_features[env_name]["max_x"]
	        nb_bin=grid_features[env_name]["nb_bin"]

	        grid=build_grid(min_x, max_x, nb_bin)
	        grid_offspring=build_grid(min_x, max_x, nb_bin)
	        stats=None
	        stats_offspring=None
	        nbc=nb_bin**2
	        nbs=nbc*2 # min 2 samples per bin
	        evolvability_nb_samples=nbs
	else:
                grid=None
                grid_offspring=None
                min_x=None
                max_x=None
                nb_bin=None
                evolvability_nb_samples=0
                nbs=0
                
	params={"IND_SIZE":eval_gym.controller.n_weights, 
		"CXPB":0, # No crossover
		"MUTPB":1., # All offspring are mutated...
		"INDPB":0.1, # ...but only 10% of parameters are mutated
		"ETA_M": 15.0, # Eta parameter for polynomial mutation
		"NGEN":nb_gen, # Number of generations
		"MIN": -5, # Seems reasonable for NN weights
		"MAX": 5, # Seems reasonable for NN weights
		"MU": pop_size,
		"LAMBDA": pop_size*2,
		"K":15,
		"ADD_STRATEGY":"random",
		"LAMBDANOV":6,
		"EVOLVABILITY_NB_SAMPLES": evolvability_nb_samples,
		"EVOLVABILITY_PERIOD":evolvability_period,
		"DUMP_PERIOD_POP": dump_period_pop,
		"DUMP_PERIOD_BD": dump_period_bd,
		"MIN_X": min_x, # not used by NS. It is just to keep track of it in the saved param file
		"MAX_X": max_x, # not used by NS. It is just to keep track of it in the saved param file
		"NB_BIN":nb_bin # not used by NS. It is just to keep track of it in the saved param file
	}


	# We use a different window size to compute statistics in order to have the same number of points for population and offspring statistics
	window_population=nbs/params["MU"]
	window_offspring=nbs/params["LAMBDA"]
	
	if (evolvability_period>0) and (evolvability_nb_samples>0):
		stats=get_stat_fit_nov_cov(grid,prefix="population_",indiv=True,min_x=min_x,max_x=max_x,nb_bin=nb_bin, gen_window_global=window_population)
		stats_offspring=get_stat_fit_nov_cov(grid_offspring,prefix="offspring_",indiv=True,min_x=min_x,max_x=max_x,nb_bin=nb_bin, gen_window_global=window_offspring)
	else:
		stats=get_stat_fit_nov_cov(grid,prefix="population_",indiv=False,min_x=min_x,max_x=max_x,nb_bin=nb_bin, gen_window_global=window_population)
		stats_offspring=get_stat_fit_nov_cov(grid_offspring,prefix="offspring_", indiv=False,min_x=min_x,max_x=max_x,nb_bin=nb_bin, gen_window_global=window_offspring)

	params["STATS"] = stats # Statistics
	params["STATS_OFFSPRING"] = stats_offspring # Statistics on offspring
	params["WINDOW_POPULATION"]=window_population
	params["WINDOW_OFFSPRING"]=window_offspring
	
	
	print("Launching Novelty Search with pop_size=%d, nb_gen=%d and evolvability_nb_samples=%d"%(pop_size, nb_gen, evolvability_nb_samples))
	if (grid is None):
                print("WARNING: grid features have not been defined for env "+env_name+". This will have no impact on the run, except that the coverage statistic has been turned off")
	if (evolvability_period>0) and (evolvability_nb_samples>0):
		print("WARNING, evolvability_nb_samples>0. The run will last much longer...")

	if with_scoop:
		pool=futures
	else:
		pool=None
		
	dump_params(params,run_name)
	pop, archive, logbook = NovES(eval_with_functor, params, pool, run_name, geno_type="realarray")
	dump_pop(pop,nb_gen,run_name)
	dump_logbook(logbook,nb_gen,run_name)
	dump_archive(archive,nb_gen,run_name)
	
	return pop, logbook




pop_size=100
nb_gen=1000
evolvability_period=0
dump_period_pop=10
dump_period_bd=1

try:
	opts, args = getopt.getopt(sys.argv[1:],"he:p:g:v:b:d:",["env_name=","pop_size=","nb_gen=","evolvability_period=","dump_period_bd=","dump_period_pop="])
except getopt.GetoptError:
	print(sys.argv[0]+" -e <env_name> [-p <population size> -g <number of generations> -v <eVolvability computation period> -b <BD dump period> -d <generation dump period>]")
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print(sys.argv[0]+" -e <env_name> [-p <population size> -g <number of generations> -v <eVolvability computation period> -b <BD dump period> -d <generation dump period>]")
		sys.exit()
	elif opt in ("-e", "--env_name"):
		env_name = arg
	elif opt in ("-p", "--pop_size"):
		pop_size = int(arg)
	elif opt in ("-g", "--nb_gen"):
		nb_gen = int(arg)
	elif opt in ("-v", "--evolvability_period"):
		evolvability_period = int(arg)
	elif opt in ("-b", "--dump_period_bd"):
		dump_period_bd = int(arg)
	elif opt in ("-d", "--dump_period_pop"):
		dump_period_pop = int(arg)
		
if(env_name is None):
	print("You must provide the environment name (as it ias been registered in gym)")
	print(sys.argv[0]+" -e <env_name> [-p <population size> -g <number of generations> -v <eVolvability computation period> -b <BD dump period> -d <generation dump period>]")
	sys.exit()
	
	
eval_gym.set_env(None,env_name, with_bd=True)


# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
	# Get env and controller

			
	run_name=generate_exp_name(env_name)
	print("Saving logs in "+run_name)
	dump_exp_details(sys.argv,run_name)

	pop, logbook = launch_nov(env_name, pop_size, nb_gen, evolvability_period, dump_period_pop, dump_period_bd)

	
	dump_end_of_exp(run_name)
	
	print("The population, log, archives, etc have been dumped in: "+run_name)
	


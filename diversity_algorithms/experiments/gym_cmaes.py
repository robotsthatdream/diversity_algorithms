
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

import gym, gym_fastsim

from diversity_algorithms.environments import EvaluationFunctor
from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 

from deap import creator, base

import dill
import pickle
import math

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.evolutionary_algorithms import set_creator
set_creator(creator)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.evolutionary_algorithms import CMA_ES
from diversity_algorithms.algorithms.utils import *
# =====


with_scoop=True

if with_scoop:
	from scoop import futures


env_name=None
# Each worker gets a functor
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
#env, controller = generate_gym_env_and_controller(params=nnparams)
eval_gym = EvaluationFunctor(controller_type=SimpleNeuralController,controller_params=nnparams,get_behavior_descriptor=None)

# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_gym(g)


def launch_cmaes(pop_size, nb_gen):
	"""Launch a cmaes search run on a gym environment
        
	Launch a cmaes search run on a gym environment:
	:param pop_size: population size
	:param nb_gen: number of generations to compute
        """

	stats=get_stats_fitness(prefix="population_")
        
	params={"IND_SIZE":eval_gym.controller.n_weights, 
		"NGEN":nb_gen, # Number of generations
		"MU": pop_size,
                "STATS": stats
	}


        
	if (pop_size is None):
                pop_size="<default>"
	print("Launching CMA-ES with pop_size="+str(pop_size)+", nb_gen=%d"%( nb_gen))

	if with_scoop:
		pool=futures
	else:
                pool=None
                
	dump_params(params,run_name)
	pop,logbook, hof = CMA_ES(eval_with_functor, params, pool, run_name)
	dump_pop(pop,nb_gen,run_name)
	dump_logbook(logbook,nb_gen,run_name)
        
	return pop, logbook

pop_size=100
nb_gen=1000

        
try:
        opts, args = getopt.getopt(sys.argv[1:],"he:p:g:",["env_name=","pop_size=","nb_gen="])
except getopt.GetoptError:
        print(sys.argv[0]+" -e <env_name> [-p <population size> -g <number of generations>]")
        sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
                print(sys.argv[0]+" -e <env_name> [-p <population size> -g <number of generations>]")
                sys.exit()
	elif opt in ("-e", "--env_name"):
                env_name = arg
	elif opt in ("-p", "--pop_size"):
                pop_size = int(arg)
	elif opt in ("-g", "--nb_gen"):
                nb_gen = int(arg)
                
if(env_name is None):
        print("You must provide the environment name (as it ias been registered in gym)")
        print(sys.argv[0]+" -e <env_name> [-p <population size> -g <number of generations>]")
        sys.exit()
        
        
eval_gym.set_env(None,env_name)


# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
        # Get env and controller

                        
	run_name=generate_exp_name(env_name)
	print("Saving logs in "+run_name)
	dump_exp_details(sys.argv,run_name)

	pop, logbook = launch_cmaes(pop_size, nb_gen)

        
	dump_end_of_exp(run_name)
	
	print("The population, log, archives, etc have been dumped in: "+run_name)
        


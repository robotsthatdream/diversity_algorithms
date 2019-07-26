
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

from diversity_algorithms.environments.maze_fastsim import *
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.algorithms.stats import *

from deap import creator, base

import dill
import pickle

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.novelty_search import set_creator
set_creator(creator)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.novelty_search import NovES
from diversity_algorithms.algorithms.utils import *

# =====

#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
#creator.create("Strategy", list, typecode="d")



with_scoop=True

if with_scoop:
	from scoop import futures


# Each worker gets a functor
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
env, controller = generate_gym_env_and_controller(params=nnparams)
eval_dist_maze = EvaluationFunctor(env, controller,with_behavior_descriptor=True)

# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_dist_maze(g)


def launch_nov(pop_size, nb_gen, evolvability_nb_samples, evolvability_period=100, dump_period_pop=10, dump_period_bd=1):
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
	min_x=[0,0]
	max_x=[600,600]
	nb_bin=20
	grid=build_grid(min_x, max_x, nb_bin)
	if (evolvability_nb_samples>0):
		stats=get_stat_coverage(grid,indiv=True,min_x=min_x,max_x=max_x,nb_bin=nb_bin)
	else:
		stats=get_stat_coverage(grid,indiv=False,min_x=min_x,max_x=max_x,nb_bin=nb_bin)

	params={"IND_SIZE":controller.n_weights, 
		"CXPB":0,
		"MUTPB":0.5,
		"NGEN":nb_gen,
		"STATS":stats,
		"MIN": -10,
		"MAX": 10,
		"MU": pop_size,
		"LAMBDA": pop_size*2,
		"K":15,
		"ADD_STRATEGY":"random",
		"LAMBDANOV":6,
                "EVOLVABILITY_NB_SAMPLES": evolvability_nb_samples,
                "EVOLVABILITY_PERIOD":evolvability_period,
                "DUMP_PERIOD_POP": dump_period_pop,
                "DUMP_PERIOD_BD": dump_period_bd
	}
	
	print("Launching Novelty Search with pop_size=%d, nb_gen=%d and evolvability_nb_samples=%d"%(pop_size, nb_gen, evolvability_nb_samples))
	if (evolvability_nb_samples>0):
                print("WARNING, evolvability_nb_samples>0. The run will last much longer...")

	if with_scoop:
		pool=futures

	dump_params(params,run_name)
	pop, logbook = NovES(eval_with_functor, params, pool, run_name)
	dump_pop(pop,nb_gen,run_name)
	dump_logbook(logbook,nb_gen,run_name)
	dump_archive(archive,nb_gen,run_name)
        
	return pop, logbook

# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
        # Get env and controller

	pop_size=100
	nb_gen=1000
	evolvability_nb_samples=0
	evolvability_period=-1

	run_name=generate_exp_name("")

        
	try:
                opts, args = getopt.getopt(sys.argv[1:],"hp:g:e:P:",["pop_size=","nb_gen=", "evolvability_nb_samples=","evolvability_period="])
	except getopt.GetoptError:
                print(sys.argv[0]+" -p <population size> -g <number of generations> -e <evolvability_nb_samples> -P <evolvability_period>")
                sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
                        print(sys.argv[0]+" -p <population size> -g <number of generations>")
                        sys.exit()
		elif opt in ("-p", "--pop_size"):
                                  pop_size = int(arg)
		elif opt in ("-g", "--nb_gen"):
                                  nb_gen = int(arg)
		elif opt in ("-e", "--evolvability_nb_samples"):
                                  evolvability_nb_samples = int(arg)
		elif opt in ("-P", "--evolvability_period"):
                                  evolvability_period = int(arg)

	pop, logbook = launch_nov(pop_size, nb_gen, evolvability_nb_samples, evolvability_period)

	
	print("The final population, logbook and run_name have been dumped by pickle in: "+run_name+"_results")
        


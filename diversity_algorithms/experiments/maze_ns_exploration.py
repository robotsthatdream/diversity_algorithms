
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

from diversity_algorithms.environments.maze_fastsim import *

from deap import creator, base


# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.novelty_search import set_creator
set_creator(creator)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.novelty_search import NovES
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


def launch_nov(pop_size, nb_gen):
	params={"IND_SIZE":controller.n_weights, 
		"CXPB":0.5,
		"MUTPB":0.5,
		"NGEN":nb_gen,
		"STATS":None,
		"MIN": -10,
		"MAX": 10,
		"MU": pop_size,
		"LAMBDA": pop_size*2,
		"K":15,
		"ADD_STRATEGY":"random",
		"LAMBDANOV":6
	}
	
	#try:
	#	del creator.FitnessMin
	#except AttributeError:
	#	pass
	#
	#try:
	#	del creator.Individual
	#except AttributeError:
	#	pass
	
	#try:
	#	del creator.Strategy
	#except AttributeError:
	#	pass
	
	#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	#creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
	#creator.create("Strategy", list, typecode="d")
	

	if with_scoop:
		pool=futures

	pop, logbook, hof, run_name = NovES(eval_with_functor, params, pool)

	return pop, logbook, hof, run_name

# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
        # Get env and controller

	pop_size=100
	nb_gen=1000
        
	try:
                opts, args = getopt.getopt(sys.argv[1:],"hp:g:",["pop_size=","nb_gen="])
	except getopt.GetoptError:
                print(sys.argv[0]+" -p <population size> -g <number of generations>")
                sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
                        print(sys.argv[0]+" -p <population size> -g <number of generations>")
                        sys.exit()
		elif opt in ("-p", "--pop_size"):
                                  pop_size = int(arg)
		elif opt in ("-g", "--nb_gen"):
                                  nb_gen = int(arg)

	pop, logbook, hof, run_name = launch_nov(pop_size, nb_gen)                                  
	
	print("Result of the mu+lambda ES on the maze with distance to obj. function: best="+str(hof[0]))

	#params["VARIANT"]=","
	#pop, logbook, hof = ES(eval_dist_maze,params)
	#print("Result of the mu,lambda ES on the maze with distance to obj. function: best="+str(hof[0])+", fitness="+str(benchmarks.sphere(hof[0])[0]))


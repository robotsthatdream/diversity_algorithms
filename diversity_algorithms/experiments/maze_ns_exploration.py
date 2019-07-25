
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

from diversity_algorithms.environments import EvaluationFunctor
from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 

from deap import creator, base

import pickle

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.novelty_search import set_creator
set_creator(creator)


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms import NovES
# =====

#creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax, strategy=None)
#creator.create("Strategy", list, typecode="d")



with_scoop=True

if with_scoop:
	from scoop import futures


# Each worker gets a functor
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
#env, controller = generate_gym_env_and_controller(params=nnparams)
eval_dist_maze = EvaluationFunctor(controller_type=SimpleNeuralController,controller_params=nnparams,with_behavior_descriptor=True)

# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_dist_maze(g)


def launch_nov(pop_size, nb_gen, evolvability_nb_samples):

        
	if (evolvability_nb_samples>0):
                min_x=[0,0]
                max_x=[600,600]
                nb_bin=20
                grid=build_grid(min_x, max_x, nb_bin)
                stats=get_stat_coverage(grid,indiv=True,min_x=min_x,max_x=max_x,nb_bin=nb_bin)
                # SD comment: indiv=True does not work for the moment, working on it...
	else:
                stats=None
	params={"IND_SIZE":eval_dist_maze.controller.n_weights, 
		"CXPB":0, # No crossover
		"MUTPB":1., # All offspring are mutated...
		"INDPB":0.1, # ...but only 10% of parameters are mutated
		"ETA_M": 15.0, # Eta parameter for polynomial mutation
		"NGEN":nb_gen, # Number of generations
		"STATS":stats, # Statistics
		"MIN": -10, # Seems reasonable for NN weights
		"MAX": 10, # Seems reasonable for NN weights
		"MU": pop_size,
		"LAMBDA": pop_size*2,
		"K":15,
		"ADD_STRATEGY":"random",
		"LAMBDANOV":6,
		"EVOLVABILITY_NB_SAMPLES": evolvability_nb_samples
	}
	
	print("Launching Novelty Search with pop_size=%d, nb_gen=%d and evolvability_nb_samples=%d"%(pop_size, nb_gen, evolvability_nb_samples))
	if (evolvability_nb_samples>0):
                print("WARNING, evolvability_nb_samples>0. The run will last much longer...")

	if with_scoop:
		pool=futures

	pop, logbook, run_name = NovES(eval_with_functor, params, pool)

	return pop, logbook, run_name

# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
        # Get env and controller

	pop_size=100
	nb_gen=1000
	evolvability_nb_samples=0
        
	try:
                opts, args = getopt.getopt(sys.argv[1:],"hp:g:e:",["pop_size=","nb_gen=", "evolvability_nb_samples="])
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
		elif opt in ("-e", "--evolvability_nb_samples"):
                                  evolvability_nb_samples = int(arg)

	pop, logbook, run_name = launch_nov(pop_size, nb_gen, evolvability_nb_samples)


	exp_res={}
	exp_res["pop"]=pop
	exp_res["logbook"]=logbook
	exp_res["run_name"]=run_name

	f=open(run_name+"_results","wb")
	pickle.dump(exp_res,f)
	f.close()
	
	print("The final population, logbook and run_name have been dumped by pickle in: "+run_name+"_results")
        


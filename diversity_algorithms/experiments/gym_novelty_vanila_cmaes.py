
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



import cma
from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.novelty_search import *


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


def generate_evolvability_samples_cmaes(run_name, es, evolvability_nb_samples, evolvability_period, gen):
    if (evolvability_nb_samples>0) and (evolvability_period>0) and (gen % evolvability_period==0):
        print("\nWARNING: evolvability_nb_samples>0. We generate %d individuals for each indiv in the population for statistical purposes"%(evolvability_nb_samples))
        print("sampling for evolvability... ",end='', flush=True)
        evolvability_samples=es.ask(number=evolvability_nb_samples)
        fit_bd = futures.map(eval_with_functor,evolvability_samples) #[eval_with_functor(g) for g in solutions]
        dump_bd_evol=open(run_name+"/bd_evol_model_gen%04d.log"%(gen),"w")
        for fbd in fit_bd:
                dump_bd_evol.write(" ".join(map(str,fbd[1]))+"\n")
        dump_bd_evol.close()
        print("done")

               
def launch_cmaes(nb_samples, run_name, dump_period_pop, dump_period_bd, variant="NS"):
        """Launch a cmaes search run on a gym environment

        Launch a cmaes search run on a gym environment.
        :param variant: the variant to launch, can be "NS" or "DM" (fitness = - distance to current model)
        """

        if (variant not in ["NS", "DM", "NS_mu1"]):
                print("Invalid variant: "+variant)

        
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

        print("Launching vanila CMA-ES. variant="+variant+". Nb_samples="+str(nb_samples)+" dump_pop="+str(dump_period_pop), " dump_bd="+str(dump_period_bd))
        k=15
        add_strategy="random"
        lambdaNov=6

        center=[0]*eval_gym.controller.n_weights
        sigma=5./3. #stdev, min-max of -5;5, which suggests a value of 5/3 for the stdev (see http://cma.gforge.inria.fr/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html) 
        
        if (variant == "NS_mu1"):
                opts = cma.CMAOptions()
                opts.set('CMA_mu', 1)		

        es = cma.CMAEvolutionStrategy(center, sigma)
        i=0
        j=0
        archive=None
        gen=0
        while (not es.stop()) and (i<nb_samples):
                j+=1
                gen+=1
                solutions = es.ask()
                i+=len(solutions)
                fit_bd = futures.map(eval_with_functor,solutions) #[eval_with_functor(g) for g in solutions]
                pop=[]
                nov=[]
                fit=[]
                for g,fbd in zip(solutions,fit_bd):
                        ind=Indiv(g,fbd[0],fbd[1])
                        pop.append(ind)
                        fit.append(fbd[0][0])

                if (variant == "DM"):
                        model=es.mean
                        fit_bd_model = list(futures.map(eval_with_functor,[model])) #[eval_with_functor(g) for g in solutions]
                        model_bd=fit_bd_model[0][1]
                        dm_fit=[]
                        for ind in pop:
                                dm_fit.append(-np.linalg.norm(np.array(model_bd)-np.array(ind.bd)))
                        es.tell(solutions,dm_fit)
                        print("Gen=%d, min dist_to_model=%f, max dist_to_model=%f, min fit=%f, max fit=%f (evals remaining=%d)"%(gen,min(dm_fit),max(dm_fit), min(fit), max(fit), nb_samples-i))
                        
                        
                if (variant in ["NS", "NS_mu1"]):
                        
                        if ((archive is not None) and (archive.ready())):
                                update_model=True
                        else:
                                update_model=False

                        archive=updateNovelty(pop,pop,archive,k,add_strategy,lambdaNov)

                        for ind in pop:
                                nov.append(ind.novelty)

                        if(update_model):
                                es.tell(solutions, [-ind.novelty for ind in pop])
                        else:
                                print("No model update, the archive still needs to grow to estimate novelty...")

                        print("Gen=%d, min novelty=%f, max novelty=%f, min fit=%f, max fit=%f (evals remaining=%d)"%(gen,min(nov),max(nov), min(fit), max(fit), nb_samples-i))

                generate_dumps(run_name, dump_period_bd, dump_period_pop, pop, None, gen, pop1label="population", archive=None, logbook=None)
                generate_evolvability_samples_cmaes(run_name, es, evolvability_nb_samples, evolvability_period, gen)

                es.disp()
        #es.result_pretty()
        return es.result
                
nb_samples=100000
dump_period_pop=10
dump_period_bd=1
evolvability_period=0
variant="NS"


try:
        opts, args = getopt.getopt(sys.argv[1:],"he:s:b:d:v:a:",["env_name=","nb_samples=","dump_period_bd=","dump_period_pop=", "evolvability_period=", "variant="])
except getopt.GetoptError:
        print(sys.argv[0]+" -e <env_name> [-s <number of samples> -b <BD dump period> -d <generation dump period> -v <eVolvability computation period>]")
        sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
                print(sys.argv[0]+" -e <env_name> [-s <number of samples> -b <BD dump period> -d <generation dump period> -v <eVolvability computation period>]")
                sys.exit()
	elif opt in ("-e", "--env_name"):
                env_name = arg
	elif opt in ("-s", "--nb_samples"):
                nb_samples = int(arg)
	elif opt in ("-b", "--dump_period_bd"):
		dump_period_bd = int(arg)
	elif opt in ("-d", "--dump_period_pop"):
		dump_period_pop = int(arg)
	elif opt in ("-v", "--evolvability_period"):
		evolvability_period = int(arg)
	elif opt in ("-a", "--variant"):
		variant = arg
                
if(env_name is None):
        print("You must provide the environment name (as it ias been registered in gym)")
        print(sys.argv[0]+" -e <env_name> [-s <number of samples>]")
        sys.exit()
        
        
eval_gym.set_env(None,env_name, with_bd=True)


# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
        # Get env and controller

                        
        run_name=generate_exp_name(env_name+"_CMAES_"+variant)
        print("Saving logs in "+run_name)
        dump_exp_details(sys.argv,run_name)

        res = launch_cmaes(nb_samples, run_name, dump_period_pop, dump_period_bd, variant=variant)
        np.savez(run_name+"/cma_result.npz",res)
        
        dump_end_of_exp(run_name)
	
        print("The population, log, archives, etc have been dumped in: "+run_name)
        


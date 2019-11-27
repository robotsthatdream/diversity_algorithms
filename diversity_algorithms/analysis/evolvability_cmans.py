from deap import tools, base, algorithms, creator
import diversity_algorithms.algorithms.utils as utils

import getopt

import gym, gym_fastsim

from diversity_algorithms.environments import EvaluationFunctor
from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 
from diversity_algorithms.algorithms.utils import * 

from diversity_algorithms.algorithms import grid_features
from diversity_algorithms.algorithms.cma_ns import generate_CMANS, CMANS_Strategy_C_rank_one, Indiv_CMANS


from os import path

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.cma_ns import set_creator_cmans
set_creator_smans(creator)

nbfitobj=2 #len(pop[0].fitness.values)

creator.create("MyFitness", base.Fitness, weights=(1.0,)*nbfitobj)
creator.create("Individual", Indiv_CMANS, typecode="d", fitness=creator.MyFitness, strategy=None)
 

with_scoop=True

if with_scoop:
    from scoop import futures
    pool=futures
else:
    pool=None


dir_name=None
env_name=None
# Each worker gets a functor
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
#env, controller = generate_gym_env_and_controller(params=nnparams)
eval_gym = EvaluationFunctor(controller_type=SimpleNeuralController,controller_params=nnparams,get_behavior_descriptor='auto')
variant="CMANS"


# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_gym(g)

def generate_evolvability_pop_cmans(pop_dir, gen):


    print("Generating evolvability for a CMANS population ")
    popfile=pop_dir+"/pop_gen%d.npz"%(gen)

        
    params=dict(np.load(pop_dir+"/params.npz", allow_pickle=True))

    if ("SIGMA" not in params.keys()):
        params["SIGMA"]=1
    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda : random.uniform(params["MIN"], params["MAX"]))
    toolbox.register("individual", generate_CMANS, creator.Individual, CMANS_Strategy_C_rank_one, size=params["IND_SIZE"], xmin=params["MIN"], xmax=params["MAX"], sigma=params["SIGMA"], w=[1]*params["MU"], lambda_ = params["LAMBDA"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_with_functor)

    if(pool):
        toolbox.register("map", pool.map)


    pop=utils.load_pop_toolbox(popfile, toolbox)
    print("Read a population of size: %d"%(len(pop)))

    generate_evolvability_samples(pop_dir,pop,params["EVOLVABILITY_NB_SAMPLES"], 1, gen, toolbox)
    return pop


try:
	opts, args = getopt.getopt(sys.argv[1:],"he:d:g:a:",["env_name=", "exp_dir=", "gen_num=", "variant="])
except getopt.GetoptError:
	print(sys.argv[0]+" -e <env_name> -d <dir_name> -g <generation_number>")
	sys.exit(2)
for opt, arg in opts:
	if opt == '-h':
		print(sys.argv[0]+" -e <env_name> -d <dir_name>")
		sys.exit()
	elif opt in ("-e", "--env_name"):
		env_name = arg
	elif opt in ("-d", "--dir_name"):
		dir_name = arg
	elif opt in ("-g", "--gen_num"):
		gen_num = int(arg)
		
if(env_name is None):
	print("You must provide the environment name (as it has been registered in gym)")
	print(sys.argv[0]+" -e <env_name>")
	sys.exit()
if(dir_name is None):
	print("You must provide the environment name (as it has been registered in gym)")
	print(sys.argv[0]+" -e <env_name>")
	sys.exit()
	
	
eval_gym.set_env(None,env_name, with_bd=True)


# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
    # Get env and controller

    generate_evolvability_pop_cmans(dir_name, gen_num)
    


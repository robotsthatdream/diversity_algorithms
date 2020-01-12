from deap import tools, base, algorithms, creator
import diversity_algorithms.algorithms.utils as utils

import getopt

import gym, gym_fastsim

from diversity_algorithms.environments import EvaluationFunctor
from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 
from diversity_algorithms.algorithms.utils import * 

import re

from diversity_algorithms.algorithms.novelty_search import build_toolbox_ns

from os import path

from diversity_algorithms.experiments.exp_utils import *

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57

from diversity_algorithms.algorithms.novelty_search import set_creator
set_creator(creator)

nbfitobj=2 #len(pop[0].fitness.values)

creator.create("MyFitness", base.Fitness, weights=(1.0,)*nbfitobj)
creator.create("Individual", list, typecode="d", fitness=creator.MyFitness, strategy=None)

with_scoop=True

if with_scoop:
    from scoop import futures
    pool=futures
else:
    pool=None


# declaration of params: RunParam(short_name (single letter for call from command line), default_value, doc)
params={
	"pop_file": RunParam("p", "", "name of population file to load"),
	"verbosity": RunParam("v", "none", "verbosity level (all, none or module specific values"),
	}

analyze_params(params, sys.argv)

# Each worker gets a functor
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
# Create a dict with all the properties of the controller
controller_params = {"controller_type":SimpleNeuralController,"controller_params":nnparams}

eval_func=None

# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_func(g)

def generate_evolvability_pop(pop_file):
    global eval_func

    pop_dir=os.path.dirname(pop_file)
    if (pop_dir==""):
        pop_dir="."

    m=re.search('.*_gen([0-9]+).npz', pop_file)
    if (m is None):
        print("Error: the population file: "+pop_file+" does not respect the template: .*_genXXX.npz")
        return None

    gen=int(m.group(1))

    params=dict(np.load(pop_dir+"/params.npz", allow_pickle=True))

    params["run_name"]=pop_dir

    # Get environment
    eval_func = create_functor(params, controller_params)


    toolbox = build_toolbox_ns(eval_func, params, pool)

    population=utils.load_pop_toolbox(pop_file, toolbox)
    print("Read a population at gen %d and of size: %d"%(gen, len(population)))

    generate_evolvability_samples(params, population, gen, toolbox, force=True)
    return population

# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
    # Get env and controller

    # TODO: 
    # - load the params
    # - create the right toolbox

    generate_evolvability_pop(params["pop_file"].get_value())
    


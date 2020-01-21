
# coding: utf-8

## EA & pyMaze expriments - vanilla python version for SCOOP parallelism
import sys,getopt
import numpy as np

import gym, gym_fastsim

from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 
import diversity_algorithms.algorithms.utils as utils

from deap import creator, base

import dill
import pickle
import math

import sys
import re

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57


from diversity_algorithms.algorithms.novelty_search import set_creator


#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.novelty_search import build_toolbox_ns
from diversity_algorithms.algorithms.utils import *

from diversity_algorithms.experiments.exp_utils import *

# =====


with_scoop=True

# Extract pop file
params_commandline = {
    "pop_file": RunParam("p", "", "name of population file to load"),
    "verbosity": RunParam("v", "none", "verbosity level (all, none or module specific values"),
}

analyze_params(params_commandline, sys.argv)


pop_file = params_commandline["pop_file"].get_value()
verbosity = params_commandline["verbosity"].get_value()


# Get run params
pop_dir=os.path.dirname(pop_file)
if (pop_dir==""):
    pop_dir="."

m=re.search('.*_gen([0-9]+).npz', pop_file)
if (m is None):
    print("Error: the population file: "+pop_file+" does not respect the template: .*_genXXX.npz")
    sys.exit(1)

gen=int(m.group(1))

params_xp=dict(np.load(pop_dir+"/params.npz", allow_pickle=True))

# Set run dir correctly
params_xp["run_name"]=pop_dir





# declaration of params: RunParam(short_name (single letter for call from command line), default_value, doc)



# Controller definition :
# Parameters of the neural net
nnparams={"n_hidden_layers": 2, "n_neurons_per_hidden": 10}
# Create a dict with all the properties of the controller
controller_params = {"controller_type":SimpleNeuralController,"controller_params":nnparams}

# Get environment
eval_func = create_functor(params_xp, controller_params)

nbobj=str(params_xp["variant"]).count("+")+1
creator.create("FitnessMax", base.Fitness, weights=(1.0,)*nbobj)
creator.create("Individual", list, typecode="d", fitness=creator.FitnessMax)
set_creator(creator)


# DO NOT pass the functor directly to futures.map -- this creates memory leaks
# Wrapper that evals with the local functor
def eval_with_functor(g):
	return eval_func(g)


def generate_evolvability_pop(pop_file):
    if with_scoop:
        from scoop import futures
        pool=futures
    else:
        pool=None
    toolbox = build_toolbox_ns(eval_with_functor, params_xp, pool)

    population=utils.load_pop_toolbox(pop_file, toolbox)
    print("Read a population at gen %d and of size: %d"%(gen, len(population)))

    generate_evolvability_samples(params_xp, population, gen, toolbox, force=True)
    return population



# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
    # Get env and controller
    generate_evolvability_pop(pop_file)


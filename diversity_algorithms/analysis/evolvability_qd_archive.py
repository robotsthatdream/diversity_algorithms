
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

import sys, os
import re

# =====
# Yes, this is ugly. This is DEAP's fault.
# See https://github.com/DEAP/deap/issues/57


from diversity_algorithms.algorithms.quality_diversity import set_creator


#creator.create("Strategy", list, typecode="d")

from diversity_algorithms.algorithms.quality_diversity import build_toolbox_qd
from diversity_algorithms.algorithms.utils import *

from diversity_algorithms.experiments.exp_utils import *

# =====


with_scoop=True

# Extract pop file
params_commandline = {
    "archive_file": RunParam("a", "", "name of archive file to load"),
    "n_sample" : RunParam("n", 100, "number of indivs to sample (-1 for all archive)"),
    "check_existing" : RunParam("", "yes", "Check if evolvability files already exist and get other indiv if it does"),
    "verbosity": RunParam("v", "none", "verbosity level (all, none or module specific values")
}

analyze_params(params_commandline, sys.argv)


archive_file = params_commandline["archive_file"].get_value()
verbosity = params_commandline["verbosity"].get_value()
check_existing = params_commandline["check_existing"].get_value()
if(check_existing in ["yes", "Yes", True, "True", "true"]):
    check_existing = True
else:
    check_existing = False
n_sample = int(params_commandline["n_sample"].get_value())


# Get run params
archive_dir=os.path.dirname(archive_file)
if (archive_dir==""):
    archive_dir="."

m=re.search('.*_gen([0-9]+).npz', archive_file)
if (m is None):
    print("Error: the archive file: "+archive_file+" does not respect the template: .*_genXXX.npz")
    sys.exit(1)

gen=int(m.group(1))

params_xp=dict(np.load(archive_dir+"/params.npz", allow_pickle=True))

# Set run dir correctly
params_xp["run_name"]=archive_dir





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


def generate_evolvability_archive(archive_file, archive_dir, n_to_sample, check_exists=True):
    if with_scoop:
        from scoop import futures
        pool=futures
    else:
        pool=None
    toolbox = build_toolbox_qd(eval_with_functor, params_xp, pool)

    archive_inds=utils.load_pop_toolbox(archive_file, toolbox) # Should also work OOTB with an archive
    n_archive = len(archive_inds)
    print("Read an archive of size %d at gen %d and "%(n_archive, gen))
    pop_sample = {}
    if((n_archive <= n_to_sample) or (n_to_sample == -1)):
        print("Asked to sample %d individuals out of %d, will just sample the full archive"%(n_to_sample, n_archive))
        pop_sample = archive_inds # List of all indivs
    else:
        already_done = []
        if(check_exists):
            pattern_find_existing = re.compile("evolvability_ind([0-9]+).*gen%d.npz" % gen)
            for filename in os.listdir(archive_dir):
                m = pattern_find_existing.match(filename)
                if(m):
                    already_done.append(int(m.groups()[0]))
        if(already_done):
            print("Found %d already computed evolvabilities")
                
        random_indices = list(range(n_to_sample))
        np.random.shuffle(random_indices)
        random_indices_ok = [i for i in random_indices if i not in already_done] # Remove the already done
        random_indices_ok_sample = random_indices_ok[:n_to_sample] # Only keep n_to_sample at most
        pop_sample = {i:archive_inds[i] for i in random_indices_ok_sample}
        


    generate_evolvability_samples(params_xp, pop_sample, gen, toolbox, force=True)
    return population



# THIS IS IMPORTANT or the code will be executed in all workers
if(__name__=='__main__'):
    # Get env and controller
    generate_evolvability_archive(archive_file, archive_dir, n_sample, check_existing)


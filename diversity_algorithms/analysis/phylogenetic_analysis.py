#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    23/01/2020
""" 


""" 
Usage : python3 phylogenetic_analysis -d <run dir>

Will create <run dir>/phylogenetic_tree.p.gz, which can later be loaded by :

---------
import dill, gzip

with gzip.open("phylogenetic_tree.p.gz") as f: 
    indivs_by_uuid, indivs_by_generation = dill.load(f) 
---------

indivs_by_uuid is a dict with all indivs in all populations by UUID
indivs_by_generation is a dict returning a list of all individuals
in the population of a given generation (indivs_by_generation[500] to get
the indivs in generation 500.

The Individuals can directly be printed; attributes are rather self-explanatory.

"""


import numpy as np
import sys, os
from diversity_algorithms.experiments.exp_utils import RunParam, analyze_params
import dill as pk
import gzip

sys.setrecursionlimit(10000)


def behavioral_dist(bd1, bd2):
	return np.sqrt(np.sum((bd1 - bd2)**2))




class Individual(object):
	def __init__(self, uuid, bd, gob, fitness=None, parent=None):
		self.id = uuid
		self.bd = np.array(bd)
		self.gob = gob # generation of birth
		self.lastgen = gob # last generation where the indiv is seen
		self.offspring = list()
		self.fit = fitness
		if(parent):
			self.parent = parent
			self.dist_to_parent = behavioral_dist(self.bd, parent.bd)
			self.evolutionary_path_length = parent.evolutionary_path_length + self.dist_to_parent
			self.parent.offspring.append(self)
		else: # Randomly generated ancestor
			self.parent = None
			self.dist_to_parent = 0.
			self.evolutionary_path_length = 0.
	def __str__(self):
		out = "-----------------\n"
		out += ("UUID: %s" % self.id) + "\n"
		out += ("Behavior: %s" % str(self.bd)) + "\n"
		if(self.fit):
			out += ("Fitness: %s" % str(self.bd)) + "\n"
		out += ("Generation of birth: %d" % self.gob) + "\n"
		out += ("Last seen on generation: %d (age %d gens)" % (self.lastgen, (1+self.lastgen-self.gob))) + "\n"
		out += ("Parent: %s" % str(self.parent.id)) + "\n"
		out += ("Distance to parent: %f" % self.dist_to_parent) + "\n"
		out += ("Total evolutionary path length: %f" % self.evolutionary_path_length) + "\n"
		out += ("Offspring : %d" % len(self.offspring)) + "\n"
		for o in self.offspring:
			out += (" * %s (born gen. %d)" % (o.id, o.gob)) + "\n"
		out += ("-----------------")
		return out


# The unknown parent, in case of errors
unknown_parent = Individual("<unknown>", [-np.inf, -np.inf], -np.inf)
unknown_parent.evolutionary_path_length = np.inf
unknown_parent.dist_to_parent = np.inf



# Arguments handling
params_commandline = {
    "rundir": RunParam("d", "", "Directory of the run"),
    "popfile_pattern": RunParam("p", "population_all_dist_to_explored_area_dist_to_parent_rank_novelty_gen%d.npz", "Pattern of the population files name"),
    "outfile": RunParam("o", "phylogenetic_tree.p.gz", "Output file"),
    "max_gen": RunParam("g", -1, "Max generation (will stop when it doesn't find pop file otherwise")
}
analyze_params(params_commandline, sys.argv)

rundir = params_commandline["rundir"].get_value()
popfile_pattern = params_commandline["popfile_pattern"].get_value()
outfile = params_commandline["outfile"].get_value()
max_gen = params_commandline["max_gen"].get_value()


if(rundir==""):
	print("Please specify a rundir with -d")
	sys.exit(1)

if(max_gen < 1):
	max_gen = np.inf



def build_phylogenetic_tree(rundir, maxgen):
	indivs_by_uuid = dict()
	indivs_by_generation = dict()
	gen = 1 # start at gen 1
	while gen < maxgen:
		try:
			pop = np.load(os.path.join(rundir, popfile_pattern % gen), allow_pickle=True)
		except FileNotFoundError:
			print("Could not find population file for gen %d, terminating." % gen)
			break
		popsize = int(pop["size"])
		print("Processing %d indivs from gen %d..." % (popsize, gen))
		indivs_gen = list()
		for i in range(popsize): # Iterate on all indivs
			uuid = str(pop["id_%d" % i])
			if uuid in indivs_by_uuid: # This is an old individual - just add it to the current gen
				indivs_gen.append(indivs_by_uuid[uuid])
				indivs_by_uuid[uuid].lastgen = gen # update the last generation seen attribute
			else: # New individual
				# First, find the parent
				myparent_id = pop["parent_id_%d" % i]
				if(myparent_id == None): # No parent - random ancestor
					myparent = None
				else:
					try:
						myparent = indivs_by_uuid[str(myparent_id)]
					except KeyError:
						print("WARNING: Can't find parent %s of indiv %s (indiv %d of gen %d)." % (str(myparent_id), uuid, i, gen))
						if(gen == 1):
							print("...but we're at the first generation, so we will consider there is no parent.")
							myparent = None
						else:
							print("Setting unknown parent.")
							myparent = unknown_parent # Set the unknown parent
				myindiv = Individual(uuid, np.array(pop["bd_%d" % i]), gen, parent=myparent) # Create individual
				indivs_by_uuid[uuid] = myindiv # Add ot to the big archive
				indivs_gen.append(myindiv) # Add it to the current gen
		indivs_by_generation[gen] = indivs_gen # save the generation list
		gen += 1 # Increment gen
	return (indivs_by_uuid, indivs_by_generation)




if(__name__=='__main__'):
	indivs_by_uuid, indivs_by_generation = build_phylogenetic_tree(rundir, max_gen)
	savefile = os.path.join(rundir, outfile)
	print("Saving data to %s..." % savefile)
	with gzip.open(savefile,'wb') as f:
		pk.dump((indivs_by_uuid, indivs_by_generation), f)
	print("Done")



















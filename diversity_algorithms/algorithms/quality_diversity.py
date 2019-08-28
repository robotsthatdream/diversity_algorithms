#!/usr/bin python -w

import random
from scipy.spatial import KDTree
import numpy as np
import datetime
import os, sys
import array


from diversity_algorithms.controllers import DNN, initDNN, mutDNN, mateDNNDummy

creator = None
def set_creator(cr):
	global creator
	creator = cr

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *

# ### Unstructured

#class NovArchive:
#	"""Archive used to compute novelty scores."""
#	def __init__(self, lbd, k=15):
#		self.all_bd=lbd
#		self.kdtree=KDTree(self.all_bd)
#		self.k=k
#		print("Archive constructor. size = %d"%(len(self.all_bd)))
#		
#	def update(self,new_bd):
#		oldsize=len(self.all_bd)
#		self.all_bd=self.all_bd + new_bd
#		self.kdtree=KDTree(self.all_bd)
#		print("Archive updated, old size = %d, new size = %d"%(oldsize,len(self.all_bd)))
#	def get_nov(self,bd, population=[]):
#		dpop=[]
#		for ind in population:
#			dpop.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
#		darch,ind=self.kdtree.query(np.array(bd),self.k)
#		d=dpop+list(darch)
#		d.sort()
#		if (d[0]!=0):
#			print("WARNING in novelty search: the smallest distance should be 0 (distance to itself). If you see it, you probably try to get the novelty with respect to a population your indiv is not in. The novelty value is then the sum of the distance to the k+1 nearest divided by k.")
#		return sum(d[:self.k+1])/self.k # as the indiv is in the population, the first value is necessarily a 0.

#	def size(self):
#		return len(self.all_bd)


def replace_if_better(oldind,newind,fit_index=0):
	return oldind.fitness.values[fit_index] < newind.fitness.values[fit_index]

def replace_always(oldind,newind):
	return True

def replace_never(oldind,newind):
	return False

class StructuredGrid:
	""" Structured grid for MAP-Elite like stuff
	    Also includes a KD-tree and maintain novelty scores
	"""
	def __init__(self, initial_pop, bins_per_dim, dims_ranges, replace_strategy=replace_never, compute_novelty=True, k_nov_knn=15):
		self.dim = len(dims_ranges)
		self.bins_per_dim = bins_per_dim
		self.dims_ranges = dims_ranges
		self.mins = np.array([r[0] for r in self.dims_ranges])
		self.maxs = np.array([r[1] for r in self.dims_ranges])
		self.grid = dict()
		for p in initial_pop:
			self.try_add(p)
		self.with_novelty = compute_novelty
		self.k = k_nov_knn
		if(compute_novelty):
			self.update_novelty()

	def get_size(self):
		return len(self.grid)

	def get_content_as_list(self):
		return list(self.grid.values())

	def update_novelty(self):
		if not self.with_novelty:
			print("ERROR: Requested novelty computation operation but the grid was built with compute_novelty=False")
			sys.exit(1)
		# 1) Build KD tree
		self.kdtree=KDTree([ind.bd for ind in self.grid.values()])
		# 2) Compute novelty values
		for bin_ in self.grid:
			nov = self.get_nov(self.grid[bin_].bd, in_archive=True)
			self.grid[bin_].novelty = nov

	def get_nov(self, bd, extra_indivs=None, in_archive=False):
		if not self.with_novelty:
			print("ERROR: Requested novelty computation operation but the grid was built with compute_novelty=False")
			sys.exit(1)
		dists=[]
		# Handle the extra_indivs
		for ind in extra_indivs:
			dists.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
		# Query KNN in archive
		dists_archive, _ = self.kdtree.query(np.array(bd),self.k+1)
		dists += list(dists_archive)
		dists.sort()
		if(in_archive):
			return sum(dists[1:self.k+1])/self.k # dists[0] is the distance to yourself
		else:
			return sum(dists[:self.k])/self.k # dists[0] is the distance to yourself



	def bd_to_bin(self,bd):
		normbd = (np.array(bd) - self.mins)/(self.maxs - self.mins)
		bins = np.array(normbd*self.bins_per_dim, dtype=int)
		return tuple(bins)

	def try_add(self,indiv):
		bd = indiv.bd
		indiv_bin = self.bd_to_bin(bd)
		if indiv_bin in self.grid:
			old_indiv = self.grid[indiv_bin]
			if replace_strategy(old_indiv, indiv): # Replace
				self.grid[indiv_bin] = indiv
				if self.with_novelty:
					self.update_novelty()
				return True
			else: # Do not replace
				return False
		else: # Cell empty - add indiv
			self.grid[indiv_bin] = indiv
			if self.with_novelty:
				self.update_novelty()
			return True

	def sample_archive(self, n, strategy="random"):
		allindivs = list(self.grid.values())
		if(strategy=="random"):
			indices = np.random.choice(self.get_size(), n, replace=False)
		elif(strategy="novelty"):
			if not self.with_novelty:
				print("ERROR: Requested novelty-based sampling but the grid was built with compute_novelty=False")
				sys.exit(1)
			novelties = [ind.novelty for ind in allindivs]
			indices = np.argsort(novelties)[:-(n+1):-1]
		else:
			print("ERROR: Unknown sampling strategy %s" % str(strategy)
			sys.exit(1)
		return allindivs[indices]


class UnstructuredArchive:
	""" Unstructured archive
	"""
	def __init__(self, initial_pop, r_ball_replace, replace_strategy=replace_never, k_nov_knn=15):
		self.r = r_ball_replace
		self.archive = list()
		for p in initial_pop:
			self.try_add(p)
		self.k = k_nov_knn
		self.update_novelty()

	def get_size(self):
		return len(self.archive)
	
	def get_content_as_list(self):
		return list(self.archive)

	def update_novelty(self):
		# 1) Build KD tree
		self.kdtree=KDTree(self.archive)
		# 2) Compute novelty values
		for (i,ind) in enumerate(self.archive):
			nov = self.get_nov(ind.bd, in_archive=True)
			self.archive[i].novelty = nov

	def get_nov(self, bd, extra_indivs=None, in_archive=False):
		dists=[]
		# Handle the extra_indivs
		for ind in extra_indivs:
			dists.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
		# Query KNN in archive
		dists_archive, _ = self.kdtree.query(np.array(bd),self.k+1)
		dists += list(dists_archive)
		dists.sort()
		if(in_archive):
			return sum(dists[1:self.k+1])/self.k # dists[0] is the distance to yourself
		else:
			return sum(dists[:self.k])/self.k # dists[0] is the distance to yourself
	
	
	def try_add(self,indiv):
		bd = indiv.bd
		close_neighbors = self.kdtree.query_ball_point(bd, self.r)
		if not close_neighbors: # No neighbors in ball, no problem - add indiv
			self.archive.append(indiv)
			self.update_novelty()
			return True
		else: # Neighbor(s)
			replace_ok = True
			for indiv_index in close_neighbors:
				old_indiv = self.archive[indiv_index]
				if not replace_strategy(old_indiv, indiv): # Replace
					replace_ok = False
					break
			if replace_ok:
				close_neighbors.sort(reverse=True)
				for index in close_neighbors:
					self.archive.pop(index) # Remove neighbors
				self.archive.append(indiv) # Add new indiv
				self.update_novelty() # Update novelty
				return True
			else: # Do not replace
				return False

	def sample_archive(self, n, strategy="random"):
		if(strategy=="random"):
			indices = np.random.choice(self.get_size(), n, replace=False)
		elif(strategy="novelty"):
			novelties = [ind.novelty for ind in self.archive]
			indices = np.argsort(novelties)[:-(n+1):-1]
		else:
			print("ERROR: Unknown sampling strategy %s" % str(strategy)
			sys.exit(1)
		return self.archive[indices]



## DEAP compatible algorithm
def QDEa(population, toolbox, n_parents, cxpb, mutpb, ngen, k_nov=15, archive_type=StructuredGrid, archive_kwargs={"bins_per_dim":50, "dims_ranges":([0,600],[0,600])}, replace_strategy=replace_never, sample_strategy="novelty", stats_offspring=None, halloffame=None, dump_period_bd=1, dump_period_pop=10, evolvability_period=50, evolvability_nb_samples=0, verbose=__debug__, run_name="runXXX"):
	"""QD algorithm
 
	QD algorithm. Parameters:
	:param population: the population to start from
	:param toolbox: the DEAP toolbox to use to generate new individuals and evaluate them
	:param mu: the number of parent individuals to keep from one generation to another
	:param lambda_: the number of offspring to generate (lambda_ needs to be greater than mu)
	:param cxpb: the recombination rate
	:param mutpb: the mutation rate
	:param ngen: the number of generation to compute
	:param k_nov: the number of neighbors to take into account while computing novelty
	:param add_strategy: the archive update strategy (can be "random" or "novel")
	:param stats: the statistic to use (on the population, i.e. survivors from parent+offspring)
	:param stats_offspring: the statistic to use (on the set of offspring)
	:param halloffame: the halloffame
	:param dump_period_bd: the period for dumping behavior descriptors
	:param dump_period_pop: the period for dumping the current population
	:param evolvability_period: period of the evolvability computation
	:param evolvability_nb_samples: the number of samples to generate from each individual in the population to estimate their evolvability (WARNING: it will significantly slow down a run and it is used only for statistical reasons
	"""
		
	if(halloffame!=None):
		print("WARNING: the hall of fame argument is ignored in the Novelty Search Algorithm")
	
		
	#print("	 lambda=%d, mu=%d, cxpb=%.2f, mutpb=%.2f, ngen=%d, k=%d, lambda_nov=%d"%(lambda_,mu,cxpb,mutpb,ngen,k,lambdaNov)) #TODO replace

	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals']
	if (stats is not None):
		logbook.header += stats.fields
	if (stats_offspring is not None):
		logbook.header += stats_offspring.fields

	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in population if not ind.fitness.valid]
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	# fit is a list of fitness (that is also a list) and behavior descriptor

	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit[0]
		ind.parent_bd=None
		ind.bd=listify(fit[1])

	for ind in population:
		ind.am_parent=0
	

	gen=0	

	# Do we look at the evolvability of individuals (WARNING: it will make runs much longer !)
	if (evolvability_nb_samples>0) and (evolvability_period>0):
		print("WARNING: evolvability_nb_samples>0. We generate %d individuals for each indiv in the population for statistical purposes"%(evolvability_nb_samples))
		print("sampling for evolvability: ",end='', flush=True)
		ig=0
		for ind in population:
			print(".", end='', flush=True)
			ind.evolvability_samples=sample_from_pop([ind],toolbox,evolvability_nb_samples,cxpb,mutpb)
			dump_bd_evol=open(run_name+"/bd_evol_indiv%04d_gen%04d.log"%(ig,gen),"w")
			for inde in ind.evolvability_samples:
				dump_bd_evol.write(" ".join(map(str,inde.bd))+"\n")
			dump_bd_evol.close()
			ig+=1
		print("")

	record = stats.compile(population) if stats is not None else {}
	record_offspring = stats_offspring.compile(population) if stats_offspring is not None else {}
	logbook.record(gen=0, nevals=len(invalid_ind), **record, **record_offspring)
	if verbose:
		print(logbook.stream)
	
	if dump_period_bd:
		dump_bd=open(run_name+"/bd_%04d.log"%gen,"w")
		for ind in population:
			dump_bd.write(" ".join(map(str,ind.bd))+"\n")
		dump_bd.close()
		dump_bd=open(run_name+"/bd_%04d.log"%gen,"w")
		for ind in population:
			dump_bd.write(" ".join(map(str,ind.bd))+"\n")
		dump_bd.close()
	
	if dump_period_pop:
		dump_pop(population, 0, run_name) # Dump initial pop

	for ind in population:
		ind.evolvability_samples=None # To avoid memory to inflate too much..
	
	archive_args = archive_kwargs + {"k_nov_knn": k_nov}
	archive = archive_type(population,**archive_args)
	
	
	# Begin the generational process
	for gen in range(1, ngen + 1):
		# Sample from the archive
		parents = archive.sample_archive(n_parents, strategy=sample_strategy)
		
		if(len(parents)) < n_parents:
			print("WARNING: Not enough individuals in archive to sample %d parents; will complete with random individuals" % n_parents)
			extra_random_indivs = toolbox.population(n=(n_parents-len(parents)))
			extra_fitnesses = toolbox.map(toolbox.evaluate, extra_random_indivs)
			for ind, fit in zip(extra_random_indivs, extra_fitnesses):
				ind.fitness.values = fit[0] 
				ind.parent_bd=None
				ind.bd=listify(fit[1])
				ind.am_parent=0
			parents += extra_random_indivs
		
		# Vary the population
		offspring = algorithms.varOr(parents, toolbox, n_parents, cxpb, mutpb)

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit[0] 
			ind.parent_bd=ind.bd
			ind.bd=listify(fit[1])
			ind.am_parent=0

		for ind in population:
			ind.am_parent=1
		for ind in offspring:
			ind.am_parent=0
			
		# Try to add the offspring to the archive
		n_added = 0
		for ind in offspring:
			if(archive.try_add(ind)):
				n_added += 1

		if(dump_period_bd and(gen % dump_period_bd == 0)): # Dump offspring behavior descriptors
			dump_bd=open(run_name+"/bd_%04d.log"%gen,"w")
			for ind in offspring:
				dump_bd.write(" ".join(map(str,ind.bd))+"\n")
			dump_bd.close()


		print("Gen %d - %d individuals added to the archive"%(gen, n_added))

		
		if(dump_period_pop and(gen % dump_period_pop == 0)): # Dump offspring
			dump_pop(offspring, gen,run_name,"offspring")
			dump_archive(archive, gen,run_name)
			dump_logbook(logbook, gen,run_name)


		
		# Do we look at the evolvability of individuals (WARNING: it will make runs much longer !)
		if (evolvability_nb_samples>0) and (evolvability_period>0) and (gen % evolvability_period == 0):
			print("Sampling for evolvability: ",end="", flush=True)
			ig=0
			for ind in offspring:
				print(".", end='', flush=True)
				ind.evolvability_samples=sample_from_pop([ind],toolbox,evolvability_nb_samples,cxpb,mutpb)
				dump_bd_evol=open(run_name+"/bd_evol_indiv%04d_gen%04d.log"%(ig,gen),"w")
				for inde in ind.evolvability_samples:
					dump_bd_evol.write(" ".join(map(str,inde.bd))+"\n")
				dump_bd_evol.close()
				ig+=1
			print("")
		
		# Update the statistics with the new population
		record_offspring = stats_offspring.compile(offspring) if stats_offspring is not None else {}
		logbook.record(gen=gen, nevals=len(invalid_ind), **record_offspring)
		if verbose:
			print(logbook.stream)

		for ind in offspring:
			ind.evolvability_samples=None

			
	return offspring, archive, logbook




def QD(evaluate,myparams,pool=None, run_name="runXXX", geno_type="realarray"):
	"""Novelty-based Mu plus lambda ES."""

	params={"IND_SIZE":1, 
			"CXPB":0, # crossover probility
			"MUTPB":0.5, # probability to mutate an individual
			"NGEN":1000, # number of generations
			"STATS_OFFSPRING":None, # Statistics on offspring
			"MIN": 0, # Min of genotype values
			"MAX": 1, # Max of genotype values
			"LAMBDA": 100, # Number of offspring generated at each generation
			"ALPHA": 0.1, # Alpha parameter of Blend crossover
			"ETA_M": 15.0, # Eta parameter for polynomial mutation
			"INDPB": 0.1, # probability to mutate a specific genotype parameter given that the individual is mutated. (The unconditional probability of a parameter being mutated is INDPB*MUTPB
			"K":15, # Number of neighbors to consider in the archive for novelty computation
			"ARCHIVE_TYPE"="grid",
			"ARCHIVE_ARGS"={"bins_per_dim":50, "dims_ranges":([0,600],[0,600])},
			"REPLACE_STRATEGY"="never",
			"SAMPLE_STRAGEGY"="novelty",
			"EVOLVABILITY_NB_SAMPLES":0, # How many children to generate to estimate evolvability
			"EVOLVABILITY_PERIOD": 100, # Period to estimate evolvability
			"DUMP_PERIOD_POP": 10, # Period to dump population
			"DUMP_PERIOD_BD": 1 # Period to dump behavior descriptors
			}
	
	
	for key in myparams.keys():
		params[key]=myparams[key]

		 
	toolbox = base.Toolbox()

	if(geno_type == "realarray"):
		print("** Unsing fixed structure networks (MLP) parameterized by a real array **")
		# With fixed NN
		# -------------
		toolbox.register("attr_float", lambda : random.uniform(params["MIN"], params["MAX"]))
		
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=params["IND_SIZE"])
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("mate", tools.cxBlend, alpha=params["ALPHA"])
	
		# Polynomial mutation with eta=15, and p=0.1 as for Leni
		toolbox.register("mutate", tools.mutPolynomialBounded, eta=params["ETA_M"], indpb=params["INDPB"], low=params["MIN"], up=params["MAX"])
	
	elif(geno_type == "dnn"):
		print("** Unsing dymamic structure networks (DNN) **")
		# With DNN (dynamic structure networks)
		#---------
		toolbox.register("individual", initDNN, creator.Individual, in_size=params["GENO_N_IN"],out_size=params["GENO_N_OUT"])

		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		toolbox.register("mate", mateDNNDummy, alpha=params["ALPHA"])
	
		# Polynomial mutation with eta=15, and p=0.1 as for Leni
		toolbox.register("mutate", mutDNN, mutation_rate_params_wb=params["DNN_MUT_PB_WB"], mutation_eta=params["DNN_MUT_ETA_WB"], mutation_rate_add_conn=params["DNN_MUT_PB_ADD_CONN"], mutation_rate_del_conn=params["DNN_MUT_PB_DEL_CONN"], mutation_rate_add_node=params["DNN_MUT_PB_ADD_NODE"], mutation_rate_del_node=params["DNN_MUT_PB_DEL_NODE"])
	else:
		raise RuntimeError("Unknown genotype type %s" % geno_type)
	#Common elements - selection and evaluation
	#toolbox.register("select", tools.selBest, fit_attr='novelty') # Useless in QD, selection is handled in archive code
	toolbox.register("evaluate", evaluate)
	
	# Parallelism
	if(pool):
		toolbox.register("map", pool.map)
	

	pop = toolbox.population(n=params["LAMBDA"])
	
	if(params["ARCHIVE_TYPE"] == "grid"):
		archiveType = StructuredGrid
	elif(params["ARCHIVE_TYPE"] == "archive"):
		archiveType = UnstructuredArchive
	else:
		print("ERROR: Unknown archive type %s" % str(params["ARCHIVE_TYPE"]))
		sys.exit(1)
	
	if(params["REPLACE_STRATEGY"]=="never"):
		replaceStrat = replace_never
	elif(params["REPLACE_STRATEGY"]=="always"):
		replaceStrat = replace_always
	elif(params["REPLACE_STRATEGY"]=="fitness"):
		replaceStrat = replace_if_better
	else:
		print("ERROR: Unknown replacement strategy %s" % str(params["REPLACE_STRATEGY"]))
		sys.exit(1)
		
	rpop, archive, logbook = QDEa(pop, toolbox, n_parents=params["LAMBDA"], cxpb=params["CXPB"], mutpb=params["MUTPB"], ngen=params["NGEN"], k_nov=params["K"], archive_type=archiveType, archive_kwargs=params["ARCHIVE_ARGS"], replace_strategy=replaceStrat, sample_strategy=params["SAMPLE_STRAGEGY"], stats_offspring=params["STATS_OFFSPRING"], halloffame=None, evolvability_nb_samples=params["EVOLVABILITY_NB_SAMPLES"], evolvability_period=params["EVOLVABILITY_PERIOD"], dump_period_bd=params["DUMP_PERIOD_BD"], dump_period_pop=params["DUMP_PERIOD_POP"], verbose=False, run_name=run_name)
		
	return rpop, archive, logbook
  
if (__name__=='__main__'):
	print("Test of the QD")

	#TODO

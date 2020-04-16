#!/usr/bin python -w

import random
from scipy.spatial import cKDTree as KDTree
import numpy as np
import datetime
import os, sys
import array


creator = None
def set_creator(cr):
	global creator
	creator = cr

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.stats import get_bd_dist_to_parent
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *

from diversity_algorithms.environments import registered_environments # To get grid dimensions


def criterion_fitness(ind):
	return ind.fitness.values[0]

def criterion_novelty(ind):
	return ind.novelty


def replace_if_better(oldind,newind,criterion):
	return criterion(oldind) < criterion(newind)

def replace_if_fitter(oldind,newind):
	return replace_if_better(oldind, newind, criterion=criterion_fitness)

def replace_if_newer(oldind,newind):
	return replace_if_better(oldind, newind, criterion=criterion_novelty)

def replace_if_further_from_parent(oldind,newind):
	return replace_if_better(oldind, newind, criterion=get_bd_dist_to_parent) # No parent = -1


def replace_always(oldind,newind):
	return True

def replace_never(oldind,newind):
	return False

def replace_random(oldind, newind, p=0.5):
	return (np.random.uniform() < p)


replace_strategies = {"never": replace_never,
	"always": replace_always,
	"random": replace_random,
	"fitness": replace_if_fitter, # WARNING: Only makes sense with a fitness/quality, we don't have that now
	"disttoparent": replace_if_further_from_parent,
	"novelty": replace_if_newer}


class StructuredGrid:
	""" Structured grid for MAP-Elite like stuff
	    Also includes a KD-tree and maintain novelty scores
	"""
	def __init__(self, initial_pop, bins_per_dim, dims_ranges, replace_strategy=replace_never, compute_novelty=True, k_nov_knn=15, kd_update_scheme="default"):
		self.dim = len(dims_ranges)
		self.kd_update_scheme = kd_update_scheme
		self.bins_per_dim = bins_per_dim
		self.dims_ranges = dims_ranges
		self.mins = np.array([r[0] for r in self.dims_ranges])
		self.maxs = np.array([r[1] for r in self.dims_ranges])
		self.grid = dict()
		self.with_novelty = compute_novelty
		self.k = k_nov_knn
		self.replace_strategy = replace_strategy
		for p in initial_pop:
			self.try_add(p)
		if(compute_novelty):
			self.update_novelty()

	def size(self):
		return len(self.grid)

	def get_content_as_list(self):
		return list(self.grid.values())

	def _rebuild_kdtree(self):
		self.kdtree=KDTree([ind.bd for ind in self.grid.values()])

	def update_novelty(self):
		if not self.with_novelty:
			print("ERROR: Requested novelty computation operation but the grid was built with compute_novelty=False")
			sys.exit(1)
		# 1) Build KD tree
		self._rebuild_kdtree()
		# 2) Compute novelty values
		for bin_ in self.grid:
			nov = self.get_nov(self.grid[bin_].bd, in_archive=True)
			self.grid[bin_].novelty = nov

	def get_nov(self, bd, extra_indivs=[], in_archive=False):
		if not self.with_novelty:
			print("ERROR: Requested novelty computation operation but the grid was built with compute_novelty=False")
			sys.exit(1)
		dists=[]
		# Handle the extra_indivs
		for ind in extra_indivs:
			dists.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
		# Query KNN in archive
		dists_archive, _ = self.kdtree.query(np.array(bd),self.k+1, n_jobs=-1)
		dists += list(dists_archive)
		dists.sort()
		if(in_archive):
			return sum(dists[1:self.k+1])/self.k # dists[0] is the distance to yourself
		else:
			return sum(dists[:self.k])/self.k

	def post_add_update(self):
		if(self.kd_update_scheme in ["default", "delayed"]): # Default = delayed : do nothing
			return
		elif(self.kd_update_scheme == "immediate"): # Immediate = juste rebuild tree
			self._rebuild_kdtree()
		elif(self.kd_update_scheme == "full"): # Full = rebuild all novelty scores (super costly)
			self.update_novelty()

	def bd_to_bin(self,bd):
		normbd = (np.array(bd) - self.mins)/(self.maxs - self.mins)
		bins = np.array(normbd*self.bins_per_dim, dtype=int)
		return tuple(bins)

	def try_add(self,indiv):
		bd = indiv.bd
		indiv_bin = self.bd_to_bin(bd)
		if indiv_bin in self.grid:
			old_indiv = self.grid[indiv_bin]
			if self.replace_strategy(old_indiv, indiv): # Replace
				self.grid[indiv_bin] = indiv
				if self.with_novelty:
					self.post_add_update()
				return True
			else: # Do not replace
				return False
		else: # Cell empty - add indiv
			self.grid[indiv_bin] = indiv
			if self.with_novelty:
				self.post_add_update()
			return True

	def sample_archive(self, n, strategy="random"):
		allindivs = list(self.grid.values())
		if(n >= self.size()): # If there are not enough (or just enough) indivs in the archive, return them all
			return allindivs
		elif(strategy=="random"):
			indices = np.random.choice(self.size(), n, replace=False)
		elif(strategy=="novelty"):
			if not self.with_novelty:
				print("ERROR: Requested novelty-based sampling but the grid was built with compute_novelty=False")
				sys.exit(1)
			novelties = [ind.novelty for ind in allindivs]
			indices = np.argsort(novelties)[:-(n+1):-1]
		else:
			print("ERROR: Unknown sampling strategy %s" % str(strategy))
			sys.exit(1)
		return list([allindivs[i] for i in indices])


class UnstructuredArchive:
	""" Unstructured archive
	"""
	def __init__(self, initial_pop, r_ball_replace, replace_strategy=replace_never, k_nov_knn=15, kd_update_scheme="default"):
		self.r = r_ball_replace
		self.kd_update_scheme = kd_update_scheme
		self.archive = list()
		self.replace_strategy = replace_strategy
		self.k = k_nov_knn
		self.kdtree = None
		for p in initial_pop:
			self.try_add(p)
		self.update_novelty()

	def size(self):
		return len(self.archive)
	
	def get_content_as_list(self):
		return list(self.archive)

	def _rebuild_kdtree(self):
		self.kdtree=KDTree([ind.bd for ind in self.archive])

	def update_novelty(self):
		# 1) Build KD tree
		self._rebuild_kdtree()
		# 2) Compute novelty values
		for (i,ind) in enumerate(self.archive):
			nov = self.get_nov(ind.bd, in_archive=True)
			self.archive[i].novelty = nov

	def get_nov(self, bd, extra_indivs=[], in_archive=False):
		dists=[]
		# Handle the extra_indivs
		for ind in extra_indivs:
			dists.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
		# Query KNN in archive
		dists_archive, _ = self.kdtree.query(np.array(bd),self.k+1, n_jobs=-1)
		dists += list(dists_archive)
		dists.sort()
		if(in_archive):
			return sum(dists[1:self.k+1])/self.k # dists[0] is the distance to yourself
		else:
			return sum(dists[:self.k])/self.k
	
	def post_add_update(self):
		if(self.kd_update_scheme in ["default", "immediate"]): # Default : juste rebuild tree
			self._rebuild_kdtree()
		elif(self.kd_update_scheme == "delayed"): # Do nothing
			return
		elif(self.kd_update_scheme == "full"): # Full = rebuild all novelty scores (super costly)
			self.update_novelty()
	
	
	def try_add(self,indiv):
		bd = indiv.bd
		close_neighbors = ([] if((self.kdtree is None) or (self.r == 0)) else self.kdtree.query_ball_point(bd, self.r, n_jobs=-1))
		if not close_neighbors: # No neighbors in ball, no problem - add indiv
			self.archive.append(indiv)
			self.post_add_update()
			return True
		else: # Neighbor(s)
			replace_ok = True
			for indiv_index in close_neighbors:
				old_indiv = self.archive[indiv_index]
				if not self.replace_strategy(old_indiv, indiv): # Replace
					replace_ok = False
					break
			if replace_ok:
				close_neighbors.sort(reverse=True)
				for index in close_neighbors:
					self.archive.pop(index) # Remove neighbors
				self.archive.append(indiv) # Add new indiv
				self.post_add_update()
				return True
			else: # Do not replace
				return False

	def sample_archive(self, n, strategy="random"):
		if(n >= self.size()): # If there are not enough (or just enough) indivs in the archive, return them all
			return list(self.archive) #return a copy
		elif(strategy=="random"):
			indices = np.random.choice(self.size(), n, replace=False)
		elif(strategy=="novelty"):
			novelties = [ind.novelty for ind in self.archive]
			indices = np.argsort(novelties)[:-(n+1):-1]
		else:
			print("ERROR: Unknown sampling strategy %s" % str(strategy))
			sys.exit(1)
		return list([self.archive[i] for i in indices])




def build_toolbox_qd(evaluate,params,pool=None):
         
    toolbox = base.Toolbox()

    if(params["geno_type"] == "realarray"):
        print("** Using fixed structure networks (MLP) parameterized by a real array **")
        # With fixed NN
        # -------------
        toolbox.register("attr_float", lambda : random.uniform(params["min"], params["max"]))
        
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=params["ind_size"])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        #toolbox.register("mate", tools.cxBlend, alpha=params["alpha"])
    
        # Polynomial mutation with eta=15, and p=0.1 as for Leni
        toolbox.register("mutate", tools.mutPolynomialBounded, eta=params["eta_m"], indpb=params["indpb"], low=params["min"], up=params["max"])
    else:
        raise RuntimeError("Unknown genotype type %s" % geno_type)

    #Common elements - selection and evaluation
    
    v=str(params["variant"])
    variant=v.replace(",","")
    if (variant == "NS"): 
        toolbox.register("select", tools.selBest, fit_attr='novelty')
    elif (variant == "Fit"):
        toolbox.register("select", tools.selBest, fit_attr='fitness')
    else:
        toolbox.register("select", tools.selNSGA2)
        
    toolbox.register("evaluate", evaluate)
    
    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)

    
    return toolbox



## DEAP compatible algorithm
def QDEa(evaluate, params, pool=None):
	"""QD algorithm
	"""
	toolbox=build_toolbox_qd(evaluate,params,pool)

	seed_population = toolbox.population(n=params["initial_seed_size"])
		
	#print("	 lambda=%d, mu=%d, cxpb=%.2f, mutpb=%.2f, ngen=%d, k=%d, lambda_nov=%d"%(lambda_,mu,cxpb,mutpb,ngen,k,lambdaNov)) #TODO replace

	nb_eval=0

	logbook = tools.Logbook()
	logbook.header = ['gen', 'nevals']
	if (params["stats"] is not None):
		logbook.header += params["stats"].fields

	# Evaluate the individuals with an invalid fitness
	invalid_ind = [ind for ind in seed_population if not ind.fitness.valid]
	nb_eval+=len(invalid_ind)
	fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
	# fit is a list of fitness (that is also a list) and behavior descriptor

	for ind, fit in zip(invalid_ind, fitnesses):
		ind.fitness.values = fit[0]
		ind.fit = fit[0]
		ind.parent_bd=None
		ind.bd=listify(fit[1])
		ind.id = generate_uuid()
		ind.parent_id = None
		ind.dist_parent = -1
		ind.gen_created = 0

	for ind in seed_population:
		ind.am_parent=0
	
	# Warnings
	if((params["archive_type"] == "unstructured") and (params["kdtree_update"] == "delayed")):
		print("*WARNING* : Using unstrructured archive with delayed kd-tree update. This is quicker but may cause two individuals to be added to the same neighborhood at a given iteration. This may or may not be important depending on what you're doing.")
	if((params["archive_type"] == "structured") and (params["kdtree_update"] == "immediate") and (params["replace_strategy"] != "novelty")):
		print("*WARNING* : Using structured archive with immediate kd-tree update. This is not a problem but except with novelty-based replacement this is useless, and much slower.")
	
	if(params["kdtree_update"] == "full"):
		if(params["replace_strategy"] == "novelty"):
			print("*WARNING* : Doing novelty-based replacement with 'full' archive post-add update. This is the best way to do it but it's very slow !")
		else:
			print("*WARNING* : Doing 'full' archive post-add update. This is very slow and useless except with novelty-based replacement !")

	if((params["archive_type"] == "unstructured") or (params["archive_type"] == "archive")):
		# If no ball size is given, take a diameter of average size of a dimension / nb_bin
		if(params["unstructured_neighborhood_radius"] < 0):
			#Fetch behavior space dimensions
			gridinfo = registered_environments[params["env_name"]]["grid_features"]
			avg_dim_sizes = np.mean(np.array(gridinfo["max_x"]) - np.array(gridinfo["min_x"]))
			params["unstructured_neighborhood_radius"] = avg_dim_sizes / (2*gridinfo["nb_bin"])
			print("Unstructured archive replace radius autoset to %f" % params["unstructured_neighborhood_radius"])
		archive = UnstructuredArchive(seed_population, r_ball_replace=params["unstructured_neighborhood_radius"], replace_strategy=replace_strategies[params["replace_strategy"]], k_nov_knn=params["k_nov"], kd_update_scheme=params["kdtree_update"])
	elif(params["archive_type"] == "grid"):
		#Fetch behavior space dimensions
		gridinfo = registered_environments[params["env_name"]]["grid_features"]
		dim_ranges = list(zip(gridinfo["min_x"],gridinfo["max_x"]))
		if(params["grid_n_bin"] <= 0):
			params["grid_n_bin"] = gridinfo["nb_bin"] # If no specific discretization is given, take the environment default
			print("Archive grid bin number autoset to %d" % params["grid_n_bin"])
		archive = StructuredGrid(seed_population, bins_per_dim=params["grid_n_bin"], dims_ranges=dim_ranges, replace_strategy=replace_strategies[params["replace_strategy"]], compute_novelty=True, k_nov_knn=params["k_nov"], kd_update_scheme=params["kdtree_update"])
	else:
		raise RuntimeError("Unknown archive type %s" % params["archive_type"])


	if(params["n_add"] <= 0):
		params["n_add"] = params["pop_size"]


	gen=0

	#Redefine the "initial population" as the archive content (maybe not all were added)
	seed_population = archive.get_content_as_list()
	
	generate_evolvability_samples(params, seed_population, gen, toolbox)



	record = params["stats"].compile(seed_population) if params["stats"] is not None else {}
	logbook.record(gen=0, nevals=len(invalid_ind), **record)
	if(verbosity(params)):
		print(logbook.stream)
	
	for ind in seed_population:
		ind.evolvability_samples=None # To prevent memory from inflating too much..
	
	# (probably) dump the original archive (gen0)
	dump_data(archive.get_content_as_list(), gen, params, prefix="archive_full", attrs=["all"])
	
	
	# Begin the generational process
	for gen in range(1, params["nb_gen"] + 1):
		# Sample from the archive
		population = archive.sample_archive(params["pop_size"], strategy=params["sample_strategy"])
		
#		print("Sampled pop")
#		for ind in population:
#			print ("* ind %s novelty %f bd %s" % (ind.id, ind.novelty, str(ind.bd)))
		
		parents = list(population)
		# We will select - at random - n_add parents from the sampled ones
		random.shuffle(parents)
		parents = parents[:params["n_add"]]
		
		if(len(parents)) < params["n_add"]:
			print("WARNING: Not enough individuals sampled to get %d parents; will complete with %d random individuals" % (params["n_add"], params["n_add"]-len(parents)))
			extra_random_indivs = toolbox.population(n=(params["n_add"]-len(parents)))
			nb_eval+=len(extra_random_indivs)
			extra_fitnesses = toolbox.map(toolbox.evaluate, extra_random_indivs)
			for ind, fit in zip(extra_random_indivs, extra_fitnesses):
				ind.fitness.values = fit[0]
				ind.fit = fit[0]
				ind.parent_bd=None
				ind.bd=listify(fit[1])
				ind.id = generate_uuid()
				ind.parent_id = None
				ind.am_parent=0
				ind.dist_parent = -1
				ind.gen_created = gen
			parents += extra_random_indivs
		
		# Vary the population
		offspring = algorithms.varOr(parents, toolbox, params["n_add"], params["cxpb"], params["mutpb"])

		# Evaluate the individuals with an invalid fitness
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		nb_eval+=len(invalid_ind)
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit[0] 
			ind.fit = fit[0]
			ind.parent_bd=ind.bd
			ind.bd=listify(fit[1])
			ind.parent_id = ind.id
			ind.id = generate_uuid()
			ind.am_parent=0
			ind.dist_parent = get_bd_dist_to_parent(ind)
			ind.gen_created = gen

		for ind in parents:
			ind.am_parent=1
		for ind in offspring:
			ind.am_parent=0
#		
#		# Compute novelties - useless, actually
#		for ind in offspring:
#			if(use_pop_for_nov):
#				ind.novelty = archive.get_nov(ind.bd, in_archive=True, extra_indivs=offspring) # about in_archive, ind is not in the archive, but it's in extra_indivs - that works the same
#			else:
#				ind.novelty = archive.get_nov(ind.bd, in_archive=False)
#		
		# Try to add the offspring to the archive
		n_added = 0
		for ind in offspring:
			if(archive.try_add(ind)):
				n_added += 1

		# Rebuild novelty for whole archive
		archive.update_novelty()

#		print("Added offspring :")
#		for ind in offspring:
#			print ("* ind %s novelty %f bd %s" % (ind.id, ind.novelty, str(ind.bd)))

		print("Gen %d - %d individuals added to the archive (current size %d)"%(gen, n_added, archive.size()))


		if (("eval_budget" in params.keys()) and (params["eval_budget"]!=-1) and (nb_eval>=params["eval_budget"])): 
			params["nb_gen"]=gen
			terminates=True
		else:
			terminates=False

		dump_data(population, gen, params, prefix="population", attrs=["all"], force=terminates)
		dump_data(offspring, gen, params, prefix="offspring", attrs=["all"], force=terminates)
		dump_data(archive.get_content_as_list(), gen, params, prefix="archive_full", attrs=["all"], force=terminates, attrs_in_name=False)
		dump_data(archive.get_content_as_list(), gen, params, prefix="archive_small", attrs=["novelty", "fit", "bd", "id", "parent_id", "parent_bd", "dist_parent", "gen_created"], force=terminates, attrs_in_name=False)
		
		#For evolvability, sample the params["pop_size"] most novel
		evolvability_pop = archive.sample_archive(params["pop_size"], strategy="novelty")
		generate_evolvability_samples(params, evolvability_pop, gen, toolbox)
		for ind in evolvability_pop:
			ind.evolvability_samples=None

		
		# Update the statistics with the new population
		record = params["stats"].compile(population) if params["stats"] is not None else {}
		logbook.record(gen=gen, nevals=len(invalid_ind), **record)
		if(verbosity(params)):
			print(logbook.stream)



			
	return archive, logbook, nb_eval



if (__name__=='__main__'):
	print("Test of the QD")

	#TODO

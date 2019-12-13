#!/usr/bin python -w

# Seed-ES: a new Novelty Search algorithm

import random
from scipy.spatial import KDTree
import numpy as np
import datetime
import os
import array
import sys

#from diversity_algorithms.controllers import DNN, initDNN, mutDNN, mateDNNDummy

creator = None
def set_creator_cmans(cr):
    global creator
    creator = cr

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.novelty_management import *
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *

import copy
from operator import attrgetter

class Indiv_CMANS(object):
    def __getitem__(self, i):
        if (self.strategy is not None):
            return self.strategy.centroid[i]
    def __len__(self):
        if (self.strategy is not None):
            return len(self.strategy.centroid)
        
    def get_centroid(self):
        return self.strategy.centroid
    def set_centroid(self,centroid):
        self.strategy.centroid = np.array(centroid)
    def set_C(self,C):
        self.strategy.C=np.array(C)

    def print_params(self):
        if (hasattr(self, "strategy") and self.strategy is not None): 
            return self.strategy.print_params()

    def dump_to_dict(self, out_dict, i, attrs):
        # Dumps to out_dict with attributes (attribute_%d)%(i), i being supposed to be the index of the ind in the dump
        #print("Ind attributes: "+str(self.__dict__.keys()))
        #print("Ind strategy attributes: "+str(self.strategy.__dict__.keys()))
        if ("all" in attrs):
            myattrs=attrs+["centroid", "C", "sigma", "w", "ccov", "fit", "novelty", "bd"]
        else:
            myattrs=attrs
        for k in self.strategy.__dict__.keys():
            if (k in myattrs):
                out_dict[k+"_%d" % (i)] = np.array(getattr(self.strategy,k))
        for k in self.__dict__.keys():
            if (k in myattrs):
                out_dict[k+"_%d" % (i)] = np.array(getattr(self,k))
        #if ((hasattr(ind,'evolvability_samples')) and (self.evolvability_samples is not None)):
        #    for (j,indj) in enumerate(self.evolvability_samples):
        #            out_dict["es_%d_%d" %(i,j)] = indj.bd 
        
class CMANS_Strategy_C_rank_one:
    def __init__(self, ind_init, centroid, sigma, w, lambda_, ccov=0.2):
        self.ind_init = ind_init
        self.centroid = np.array(centroid)
        self.C = np.identity(len(centroid))
        self.sigma=sigma
        self.w=[ww/sum(w) for ww in w] # weights must sum to one
        self.ccov=ccov
        self.muw=1./sum([ww**2 for ww in self.w])
        self.lambda_=lambda_
        self.mu=len(w)

        self.diagD, self.B = np.linalg.eigh(self.C)
        indx = np.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD


    def print_params(self):
        return "Centroid: "+str(self.centroid)+" C: "+str(self.C)+" sigma: "+str(self.sigma)+" w: "+str(self.w)+" ccov: "+str(self.ccov)+" lambda: "+str(self.lambda_)+" mu: "+str(self.mu)
        
    def generate_samples(self, lambda_):
        arz = [ self.centroid + self.sigma*np.dot(np.random.standard_normal(self.centroid.shape),self.BD.T) for _ in range(lambda_)]
        npop= [ self.ind_init() for _ in range(lambda_)]
        #print("Generate: ")
        for centroid,p in zip(arz,npop):
            p.strategy = copy.deepcopy(self)
            p.set_centroid(centroid)
            #print("    "+str(p.get_centroid()))
        return npop

    def generate(self):
        return self.generate_samples(self.lambda_)

    def update(self,population, variant="CMANS"):
        # in this version, the centroid is not adapted
        #print("Before update of C: min=%f max=%f"%(self.C.min(), self.C.max()))
        if (variant=="CMANS"):
            sorted_pop = sorted(population, key=attrgetter("novelty"), reverse=True)
        elif (variant=="CMANSD"):
            sorted_pop = sorted(population, key=attrgetter("dist_to_model"), reverse=True)
            
        y = [(s.get_centroid() - self.centroid)/self.sigma for s in sorted_pop[:self.mu]]
        #print("Len y=%d mu=%d muw=%f"%(len(y), self.mu, self.muw))
        yw=np.array([sum([self.w[i]*y[i] for i in range(self.mu)])])
        #print("Len yw=%d yw="%(len(yw))+str(yw))
        ywywT= np.dot(yw.T,yw)

        #print("yw.ywT="+str(ywywT))
        self.C = (1.-self.ccov)*self.C + self.ccov * self.muw * ywywT
        #print("Updated C: "+str(self.C))
        #print("After update of C: min=%f max=%f"%(self.C.min(), self.C.max()))

        self.diagD, self.B = np.linalg.eigh(self.C)
        indx = np.argsort(self.diagD)

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]

        self.diagD = self.diagD[indx] ** 0.5
        self.B = self.B[:, indx]
        self.BD = self.B * self.diagD
        

def generate_CMANS(icls, scls, size, xmin, xmax, sigma, w, lambda_=100, ccov=0.2):
    centroid = [random.uniform(xmin, xmax) for _ in range(size)]
    ind = icls(centroid)
    ind.strategy = scls(icls, centroid, sigma, w, lambda_, ccov)
    return ind

def build_toolbox_cmans(evaluate,params,pool=None):
         
    toolbox = base.Toolbox()
    
    if(params["geno_type"]!="realarray"):
        print("ERROR: CMANS supports only realarray genotypes.")
        sys.exit(1)

    print("** Using fixed structure networks (MLP) parameterized by a real array **")
    # With fixed NN
    # -------------
    toolbox.register("attr_float", lambda : random.uniform(params["min"], params["max"]))
    
    toolbox.register("individual", generate_CMANS, creator.Individual, CMANS_Strategy_C_rank_one, size=params["ind_size"], xmin=params["min"], xmax=params["max"], sigma=params["sigma"], w=[1]*params["cma_lambda"], lambda_ = params["cma_lambda"], ccov=params["ccov"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selNSGA2)
        
    toolbox.register("evaluate", evaluate)
    
    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)
    
    return toolbox


def cmans(evaluate, params, pool):
    """CMA-NS algorithm
 
    CMA-NS algorithm. Parameters:
    :param population: the population of seeds to start from
    :param toolbox: the DEAP toolbox to use to generate new individuals and evaluate them
    :param mu: the number of parent individuals to keep from one generation to another
    :param lambda_: the number of offspring to generate (lambda_ needs to be greater than mu)
    :param cxpb: the recombination rate
    :param mutpb: the mutation rate
    :param ngen: the number of generation to compute
    :param k: the number of neighbors to take into account while computing novelty
    :param add_strategy: the archive update strategy (can be "random" or "novel")
    :param lambdaNov: the number of individuals to add to the archive at a given generation
    :param stats: the statistic to use (on the population, i.e. survivors from parent+offspring)
    :param stats_offspring: the statistic to use (on the set of offspring)
    :param halloffame: the halloffame
    :param dump_period_bd: the period for dumping behavior descriptors
    :param dump_period_pop: the period for dumping the current population
    :param evolvability_period: period of the evolvability computation
    :param evolvability_nb_samples: the number of samples to generate from each individual in the population to estimate their evolvability (WARNING: it will significantly slow down a run and it is used only for statistical reasons
    """
    
    print("CMA-NS Novelty search algorithm")

    toolbox=build_toolbox_cmans(evaluate,params,pool)

    population = toolbox.population(n=params["pop_size"])
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals']
    if (params["stats"] is not None):
        logbook.header += params["stats"].fields
    if (params["stats_offspring"] is not None):
        logbook.header += params["stats_offspring"].fields
    #logbook=None

    # The size of the population is initially mu. We generate mu other random individuals
    population+=toolbox.population(n=params["pop_size"])
    archive=None

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fit = fit[0]
        #ind.fitness.values = fit[0]
        #            ind.parent_bd=ind.bd
        ind.bd=listify(fit[1])

        
    # Begin the generational process
    for gen in range(params["nb_gen"] + 1):

        # The population contains a set of seeds

        # Generate a set of points for each seed
        all_samples=[]
        samples_per_seed={}
        for s in range(len(population)):
            samples = population[s].strategy.generate()
            samples_per_seed[s]=samples
            all_samples+=samples

        # Evaluate the individuals with an invalid fitness
        fitnesses = toolbox.map(toolbox.evaluate, all_samples)
        for ind, fit in zip(all_samples, fitnesses):
            ind.fit = fit[0]
            ind.bd=listify(fit[1])
            
        archive=updateNovelty(all_samples,all_samples,archive,params)

        # Compute the fitness values for each seed: uniformity and cumulated novelty
        for s in range(len(population)):
            #print("s="+str(s)+ " population[s]="+str(population[s]))
            cumul=0
            for i in samples_per_seed[s]:
                i.dist_to_model=np.linalg.norm(np.array(population[s].bd)-np.array(i.bd))
                cumul+=i.novelty
            population[s].fitness.values=(cumul_distance(samples_per_seed[s]), cumul)
            population[s].fit=(cumul_distance(samples_per_seed[s]), cumul)
            population[s].novelty=-1 # to simplify stats

            population[s].strategy.update(samples_per_seed[s], params["variant"])
            #print("Cumul distance: %f, cumul novelty: %f"%(population[s].fitness.values[0], population[s].fitness.values[1]))

        # Select the seeds to survive with NSGA-2
        population[:] = toolbox.select(population, params["pop_size"])        

        # Add new seeds: the most novel (and distant ) ones
        all_samples.sort(key=lambda x:x.novelty)

        #print("Novelty: min=%f, max=%f"%(all_samples[0].novelty, all_samples[-1].novelty))


        if (verbosity(params)):
            print("Gen %d"%(gen))
        else:
            if(gen%100==0):
                print(" %d "%(gen), end='', flush=True)
            elif(gen%10==0):
                print("+", end='', flush=True)
            else:
                print(".", end='', flush=True)
                
        dump_data(population, gen, params, prefix="population", attrs=["all"])
        dump_data(population, gen, params, prefix="bd", complementary_name="population", attrs=["bd"])
        dump_data(all_samples, gen, params, prefix="bd", complementary_name="all_samples", attrs=["bd"])
        dump_data(archive.get_content_as_list(), gen, params, prefix="archive", attrs=["all"])
        
        generate_evolvability_samples(params, population, gen, toolbox)
        
        # Update the statistics with the new population
        record = params["stats"].compile(population) if params["stats"] is not None else {}
#        record_offspring = stats_offspring.compile(all_samples) if stats_offspring is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record) #, **record_offspring)
        if (verbosity(params)):
            print(logbook.stream)

        for ind in population:
            ind.evolvability_samples=None

        population+=all_samples[-len(population):] # update the number of elements to add

            
    return population, archive, logbook

if (__name__=='__main__'):
    pass


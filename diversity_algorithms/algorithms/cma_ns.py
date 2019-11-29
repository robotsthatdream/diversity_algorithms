#!/usr/bin python -w

# Seed-ES: a new Novelty Search algorithm

import random
from scipy.spatial import KDTree
import numpy as np
import datetime
import os
import array


#from diversity_algorithms.controllers import DNN, initDNN, mutDNN, mateDNNDummy

creator = None
def set_creator_cmans(cr):
    global creator
    creator = cr

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.novelty_search import *
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

class CMANS_Strategy_C_rank_one:
    def __init__(self, ind_init, centroid, sigma, w, lambda_, ccov=0.2):
        self.ind_init = ind_init
        self.centroid = np.array(centroid)
        self.C = np.identity(len(centroid))
        self.sigma=sigma
        self.w=[ww/sum(w) for ww in w] # weights must sum to one
        self.ccov=ccov
        self.muw=1./sum([ww**2 for ww in w])
        self.lambda_=lambda_
        self.mu=len(w)


    def generate_samples(self, lambda_):
        arz = [ self.centroid + self.sigma*np.dot(np.random.standard_normal(self.centroid.shape),self.C.T) for _ in range(lambda_)]
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
        ywywT=np.dot(yw.T,yw)
        #print("yw.ywT="+str(ywywT))
        self.C = (1.-self.ccov)*self.C + self.ccov * self.muw * ywywT
        #print("Updated C: "+str(self.C))
        #print("After update of C: min=%f max=%f"%(self.C.min(), self.C.max()))
        
def generate_CMANS(icls, scls, size, xmin, xmax, sigma, w, lambda_=100, ccov=0.2):
    centroid = [random.uniform(xmin, xmax) for _ in range(size)]
    ind = icls(centroid)
    ind.strategy = scls(icls, centroid, sigma, w, lambda_, ccov)
    return ind


## DEAP compatible algorithm
def cmans(population, toolbox, mu, lambda_, ngen,k,add_strategy,lambdaNov,
                          stats=None, stats_offspring=None, halloffame=None, dump_period_bd=1, dump_period_pop=10, evolvability_period=50, evolvability_nb_samples=0, verbose=__debug__, run_name="runXXX", variant="CMANS"):
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

        
    if(halloffame!=None):
        print("WARNING: the hall of fame argument is ignored in the Novelty Search Algorithm")
    
    print("CMA-NS Novelty search algorithm")
    print("     variant="+variant)
    print("     lambda=%d, mu=%d, ngen=%d, k=%d, lambda_nov=%d"%(lambda_,mu,ngen,k,lambdaNov))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals']
    if (stats is not None):
        logbook.header += stats.fields
    if (stats_offspring is not None):
        logbook.header += stats_offspring.fields
    #logbook=None

    # The size of the population is initially mu. We generate mu other random individuals
    population+=toolbox.population(n=mu)
    archive=None

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fit = fit[0]
        #ind.fitness.values = fit[0]
        #            ind.parent_bd=ind.bd
        ind.bd=listify(fit[1])

        
    # Begin the generational process
    for gen in range(ngen + 1):

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
            
        archive=updateNovelty(all_samples,all_samples,archive,k,add_strategy,lambdaNov)

        # Compute the fitness values for each seed: uniformity and cumulated novelty
        for s in range(len(population)):
            #print("s="+str(s)+ " population[s]="+str(population[s]))
            cumul=0
            for i in samples_per_seed[s]:
                i.dist_to_model=np.linalg.norm(np.array(population[s].bd)-np.array(i.bd))
                cumul+=i.novelty
            population[s].fitness.values=(cumul_distance(samples_per_seed[s]), cumul)
            population[s].novelty=-1 # to simplify stats

            population[s].strategy.update(samples_per_seed[s], variant)
            #print("Cumul distance: %f, cumul novelty: %f"%(population[s].fitness.values[0], population[s].fitness.values[1]))

        # Select the seeds to survive with NSGA-2
        population[:] = toolbox.select(population, mu)        

        # Add new seeds: the most novel (and distant ) ones
        all_samples.sort(key=lambda x:x.novelty)

        #print("Novelty: min=%f, max=%f"%(all_samples[0].novelty, all_samples[-1].novelty))


        if (verbose):
            print("Gen %d"%(gen))
        else:
            if(gen%100==0):
                print(" %d "%(gen), end='', flush=True)
            elif(gen%10==0):
                print("+", end='', flush=True)
            else:
                print(".", end='', flush=True)

        generate_dumps(run_name, dump_period_bd, dump_period_pop, population, all_samples, gen, pop1label="population", pop2label="all_samples", archive=archive, logbook=logbook)
        
        generate_evolvability_samples(run_name, population, evolvability_nb_samples, evolvability_period, gen, toolbox)
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
#        record_offspring = stats_offspring.compile(all_samples) if stats_offspring is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record) #, **record_offspring)
        if verbose:
            print(logbook.stream)

        for ind in population:
            ind.evolvability_samples=None

        population+=all_samples[-len(population):] # update the number of elements to add

            
    return population, archive, logbook




def CMA_NS(evaluate,myparams,pool=None, run_name="runXXX", geno_type="realarray"):
    """CMA_NS Diversity algorithm."""

    params={"IND_SIZE":1, 
            "NGEN":1000, # number of generations
            "STATS":None, # Statistics
            "STATS_OFFSPRING":None, # Statistics on offspring
            "MIN": -5, # Min of genotype values
            "MAX": 5, # Max of genotype values
            "MU": 20, # Number of individuals selected at each generation
            "SIGMA": 1,
            "LAMBDA": 100, # Number of offspring generated at each generation
            "K":15, # Number of neighbors to consider in the archive for novelty computation
            "ADD_STRATEGY":"random", # Selection strategy to add individuals to the archive
            "LAMBDANOV":6, # How many individuals to add to the archive at each gen
            "EVOLVABILITY_NB_SAMPLES":0, # How many children to generate to estimate evolvability
            "EVOLVABILITY_PERIOD": 100, # Period to estimate evolvability
            "DUMP_PERIOD_POP": 10, # Period to dump population
            "DUMP_PERIOD_BD": 1, # Period to dump behavior descriptors
            "VARIANT": "CMANS" # "NS", "Fit", "NS+Fit", "NS+BDDistP", "NS+Fit+BDDistP" or any variant with "," at the end ("NS," for instance) if selection within the offspring only ("," selection scheme of ES) 
    }
    
    
    for key in myparams.keys():
        params[key]=myparams[key]

         
    toolbox = base.Toolbox()

    print("** Unsing fixed structure networks (MLP) parameterized by a real array **")
    # With fixed NN
    # -------------
    toolbox.register("attr_float", lambda : random.uniform(params["MIN"], params["MAX"]))
    
    toolbox.register("individual", generate_CMANS, creator.Individual, CMANS_Strategy_C_rank_one, size=params["IND_SIZE"], xmin=params["MIN"], xmax=params["MAX"], sigma=params["SIGMA"], w=[1]*params["MU"], lambda_ = params["LAMBDA"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #toolbox.register("mate", tools.cxBlend, alpha=params["ALPHA"])

    #def __init__(self, centroid, sigma, w, lambda_, ccov=0.2):

    #strategy=CMANS_Strategy([0]*params["IND_SIZE"], 5, [1]*params["MU"], params["LAMBDA"])
    
    # Polynomial mutation with eta=15, and p=0.1 as for Leni
    #toolbox.register("mutate", tools.mutPolynomialBounded, eta=params["ETA_M"], indpb=params["INDPB"], low=params["MIN"], up=params["MAX"])
    

    #Common elements - selection and evaluation
    toolbox.register("select", tools.selNSGA2)
        
    toolbox.register("evaluate", evaluate)
    
    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)
    

    pop = toolbox.population(n=params["MU"])
    
    rpop, archive, logbook = cmans(pop, toolbox, mu=params["MU"], lambda_=params["LAMBDA"], ngen=params["NGEN"], k=params["K"], add_strategy=params["ADD_STRATEGY"], lambdaNov=params["LAMBDANOV"],stats=params["STATS"], stats_offspring=params["STATS_OFFSPRING"], halloffame=None, evolvability_nb_samples=params["EVOLVABILITY_NB_SAMPLES"], evolvability_period=params["EVOLVABILITY_PERIOD"], dump_period_bd=params["DUMP_PERIOD_BD"], dump_period_pop=params["DUMP_PERIOD_POP"], verbose=False, run_name=run_name, variant=params["VARIANT"])
        
    return rpop, archive, logbook
  
if (__name__=='__main__'):
    pass


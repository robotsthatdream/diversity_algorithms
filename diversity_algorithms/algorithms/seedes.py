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
def set_creator(cr):
    global creator
    creator = cr

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.novelty_search import *
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *


## DEAP compatible algorithm
def seed_ES(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,k,add_strategy,lambdaNov,
                          stats=None, stats_offspring=None, halloffame=None, dump_period_bd=1, dump_period_pop=10, evolvability_period=50, evolvability_nb_samples=0, verbose=__debug__, run_name="runXXX", variant="NS"):
    """Novelty Search algorithm
 
    Novelty Search algorithm. Parameters:
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
    
    print("Seed-ES Novelty search algorithm")
    print("     variant="+variant)
    print("     lambda=%d, mu=%d, cxpb=%.2f, mutpb=%.2f, ngen=%d, k=%d, lambda_nov=%d"%(lambda_,mu,cxpb,mutpb,ngen,k,lambdaNov))

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
            samples = algorithms.varOr([population[s]], toolbox, lambda_, 0, mutpb)
            samples_per_seed[s]=samples
            all_samples+=samples

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in all_samples if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fit = fit[0]
            #ind.fitness.values = fit[0]
#            ind.parent_bd=ind.bd
            ind.bd=listify(fit[1])

        archive=updateNovelty(all_samples,all_samples,archive,k,add_strategy,lambdaNov)

        # Compute the fitness values for each seed: uniformity and cumulated novelty
        for s in range(len(population)):
            #print("s="+str(s)+ " population[s]="+str(population[s]))
            cumul=0
            for i in samples_per_seed[s]:
                cumul+=i.novelty
            population[s].fitness.values=(cumul_distance(samples_per_seed[s]), cumul)
            population[s].novelty=-1 # to simplify stats
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
        
        generate_evolvability_samples(run_name, population, toolbox, evolvability_nb_samples, evolvability_period, gen, cxpb, mutpb)
        
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




def SeedES(evaluate,myparams,pool=None, run_name="runXXX", geno_type="realarray"):
    """Seed-ES Novelty search."""

    params={"IND_SIZE":1, 
            "CXPB":0, # crossover probility
            "MUTPB":1, # probability to mutate an individual
            "NGEN":1000, # number of generations
            "STATS":None, # Statistics
            "STATS_OFFSPRING":None, # Statistics on offspring
            "MIN": 0, # Min of genotype values
            "MAX": 1, # Max of genotype values
            "MU": 20, # Number of individuals selected at each generation
            "LAMBDA": 100, # Number of offspring generated at each generation
            "ALPHA": 0.1, # Alpha parameter of Blend crossover
            "ETA_M": 15.0, # Eta parameter for polynomial mutation
            "INDPB": 0.1, # probability to mutate a specific genotype parameter given that the individual is mutated. (The unconditional probability of a parameter being mutated is INDPB*MUTPB
            "K":15, # Number of neighbors to consider in the archive for novelty computation
            "ADD_STRATEGY":"random", # Selection strategy to add individuals to the archive
            "LAMBDANOV":6, # How many individuals to add to the archive at each gen
            "EVOLVABILITY_NB_SAMPLES":0, # How many children to generate to estimate evolvability
            "EVOLVABILITY_PERIOD": 100, # Period to estimate evolvability
            "DUMP_PERIOD_POP": 10, # Period to dump population
            "DUMP_PERIOD_BD": 1, # Period to dump behavior descriptors
            "VARIANT": "NS" # "NS", "Fit", "NS+Fit", "NS+BDDistP", "NS+Fit+BDDistP" or any variant with "," at the end ("NS," for instance) if selection within the offspring only ("," selection scheme of ES) 
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
    toolbox.register("select", tools.selNSGA2)
        
    toolbox.register("evaluate", evaluate)
    
    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)
    

    pop = toolbox.population(n=params["MU"])
    
    rpop, archive, logbook = seed_ES(pop, toolbox, mu=params["MU"], lambda_=params["LAMBDA"], cxpb=params["CXPB"], mutpb=params["MUTPB"], ngen=params["NGEN"], k=params["K"], add_strategy=params["ADD_STRATEGY"], lambdaNov=params["LAMBDANOV"],stats=params["STATS"], stats_offspring=params["STATS_OFFSPRING"], halloffame=None, evolvability_nb_samples=params["EVOLVABILITY_NB_SAMPLES"], evolvability_period=params["EVOLVABILITY_PERIOD"], dump_period_bd=params["DUMP_PERIOD_BD"], dump_period_pop=params["DUMP_PERIOD_POP"], verbose=False, run_name=run_name, variant=params["VARIANT"])
        
    return rpop, archive, logbook
  
if (__name__=='__main__'):
    print("Test of the Novelty-based ES")

    OK=True

    print("Test of the archive")

    lbd=[[i] for i in range(100)]
    archive=NovArchive(lbd,5)
    test=[[[50],6./5.], [[0],2.]]
    for t in test:
        if(archive.get_nov(t[0])!=t[1]):
            print("ERROR: Estimated value: %f, ground truth: %f"%(archive.get_nov(t[0]),t[1]))
            OK=False
        else:
            print('.', end='')
    print("")

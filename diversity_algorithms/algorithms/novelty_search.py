#!/usr/bin python -w

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
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *

from diversity_algorithms.algorithms.novelty_management import *

__all__=["novelty_ea"]

def build_toolbox_ns(evaluate,params,pool=None):
         
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
    
    elif(params["geno_type"] == "dnn"):
        print("** Using dymamic structure networks (DNN) **")
        # With DNN (dynamic structure networks)
        #---------
        toolbox.register("individual", initDNN, creator.Individual, in_size=params["geno_n_in"],out_size=params["geno_n_out"])

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        #toolbox.register("mate", mateDNNDummy, alpha=params["alpha"])
    
        # Polynomial mutation with eta=15, and p=0.1 as for Leni
        toolbox.register("mutate", mutDNN, mutation_rate_params_wb=params["dnn_mut_pb_wb"], mutation_eta=params["dnn_mut_eta_wb"], mutation_rate_add_conn=params["dnn_mut_pb_add_conn"], mutation_rate_del_conn=params["dnn_mut_pb_del_conn"], mutation_rate_add_node=params["dnn_mut_pb_add_node"], mutation_rate_del_node=params["dnn_mut_pb_del_node"])
    else:
        raise RuntimeError("Unknown genotype type %s" % geno_type)

    #Common elements - selection and evaluation
    variant=params["variant"].replace(",","")
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
def novelty_ea(evaluate, params, pool=None):
    """Novelty Search algorithm
 
    Novelty Search algorithm. Parameters:
    :param population: the population to start from
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
    print("Novelty search algorithm")

    variant=params["variant"]
    if ("+" in variant):
        emo=True
    else:
        emo=False

    lambda_ = int(params["lambda"]*params["pop_size"])

    toolbox=build_toolbox_ns(evaluate,params,pool)

    population = toolbox.population(n=params["pop_size"])

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals']

    if (params["stats"] is not None):
        logbook.header += params["stats"].fields
    if (params["stats_offspring"] is not None):
        logbook.header += params["stats_offspring"].fields

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # fit is a list of fitness (that is also a list) and behavior descriptor

    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fit = fit[0] # fit is an attribute just used to store the fitness value
        ind.parent_bd=None
        ind.bd=listify(fit[1])

    for ind in population:
        ind.am_parent=0
        
    archive=updateNovelty(population,population,None,params)

    varian=params["variant"].replace(",","")

    
    for ind in population:

        if (emo):
            if (varian == "NS+Fit"):
                ind.fitness.values = (ind.novelty, ind.fit)
            elif (varian == "NS+BDDistP"):
                ind.fitness.values = (ind.novelty, 0)
            elif (varian == "NS+Fit+BDDistP"):
                ind.fitness.values = (ind.novelty, ind.fit, 0)
            else:
                print("WARNING: unknown variant: "+variant)
                ind.fitness.values=ind.fit
        else:
            ind.fitness.values=ind.fit
    
    gen=0    

    # Do we look at the evolvability of individuals (WARNING: it will make runs much longer !)
    generate_evolvability_samples(params, population, gen, toolbox)

    record = params["stats"].compile(population) if params["stats"] is not None else {}
    record_offspring = params["stats_offspring"].compile(population) if params["stats_offspring"] is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record, **record_offspring)
    if (verbosity(params)):
        print(logbook.stream)
    
    #generate_dumps(params, population, None, gen, pop1label="population", archive=None, logbook=None)

    for ind in population:
        ind.evolvability_samples=None # To avoid memory to inflate too much..
        
    # Begin the generational process
    for gen in range(1, params["nb_gen"] + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, params["cxpb"], params["mutpb"])

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fit = fit[0]
            ind.fitness.values = fit[0]
            ind.parent_bd=ind.bd
            ind.bd=listify(fit[1])

        for ind in population:
            ind.am_parent=1
        for ind in offspring:
            ind.am_parent=0
            
        pq=population+offspring

        
        archive=updateNovelty(pq,offspring,archive,params)

        for ind in pq:
            if (emo):
                if (varian == "NS+Fit"):
                    ind.fitness.values = (ind.novelty, ind.fit)
                elif (varian == "NS+BDDistP"):
                    if (ind.parent_bd is None):
                        bddistp=0
                    else:
                        bddistp=np.linalg.norm(np.array(ind.bd) - np.array(ind.parent_bd))
                    ind.fitness.values = (ind.novelty, bddistp)
                elif (varian == "NS+Fit+BDDistP"):
                    if (ind.parent_bd is None):
                        bddistp=0
                    else:
                        bddistp=np.linalg.norm(np.array(ind.bd) - np.array(ind.parent_bd))
                    ind.fitness.values = (ind.novelty, ind.fit, bddistp)
                else:
                    print("WARNING: unknown variant: "+variant)
                    ind.fitness.values=ind.fit

            else:
                ind.fitness.values=ind.fit

        if ((emo) and (offspring[0].fitness.values == offspring[0].fit)):
            print ("WARNING: EMO and the fitness is just the fitness !")

        if (verbosity(params)):
            print("Gen %d"%(gen))
        else:
            if(gen%100==0):
                print(" %d "%(gen), end='', flush=True)
            elif(gen%10==0):
                print("+", end='', flush=True)
            else:
                print(".", end='', flush=True)

        
        # Select the next generation population
        if ("," in variant):
            population[:] = toolbox.select(offspring, params["pop_size"])        
        else:
            population[:] = toolbox.select(pq, params["pop_size"])        

        dump_data(population, gen, params, prefix="population", attrs=["all"])
        dump_data(population, gen, params, prefix="bd", complementary_name="population", attrs=["bd"])
        dump_data(offspring, gen, params, prefix="bd", complementary_name="offspring", attrs=["bd"])
        dump_data(archive.get_content_as_list(), gen, params, prefix="archive", attrs=["all"])

        generate_evolvability_samples(params, population, gen, toolbox)
        
        # Update the statistics with the new population
        record = params["stats"].compile(population) if params["stats"] is not None else {}
        record_offspring = params["stats_offspring"].compile(offspring) if params["stats_offspring"] is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record, **record_offspring)
        if (verbosity(params)):
            print(logbook.stream)

        for ind in population:
            ind.evolvability_samples=None

            
    return population, archive, logbook




  
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

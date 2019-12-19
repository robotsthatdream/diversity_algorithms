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
def set_creator_seedes(cr):
    global creator
    creator = cr

from deap import tools, base, algorithms

from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.novelty_management import *
from diversity_algorithms.analysis.population_analysis import *
from diversity_algorithms.analysis.data_utils import *

def build_toolbox_seedes(evaluate, params, pool=None):

    toolbox = base.Toolbox()

    if(params["geno_type"] == "realarray"):
        print("** Using fixed structure networks (MLP) parameterized by a real array **")
        # With fixed NN
        # -------------
        toolbox.register("attr_float", lambda : random.uniform(params["min"], params["max"]))
        
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=params["ind_size"])
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxBlend, alpha=params["alpha"])
    
        # Polynomial mutation with eta=15, and p=0.1 as for Leni
        if(params["variant"] == "SES-Gaussian"):
            toolbox.register("mutate", tools.mutGaussian, mu=0., sigma=params["sigma_gaussian"], indpb=params["indpb"])
        else:
            toolbox.register("mutate", tools.mutPolynomialBounded, eta=params["eta_m"], indpb=params["indpb"], low=params["min"], up=params["max"])
    
    elif(params["geno_type"] == "dnn"):
        print("** Unsing dymamic structure networks (DNN) **")
        # With DNN (dynamic structure networks)
        #---------
        toolbox.register("individual", initDNN, creator.Individual, in_size=params["geno_n_in"],out_size=params["geno_n_out"])

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", mateDNNDummy, alpha=params["alpha"])
    
        # Polynomial mutation with eta=15, and p=0.1 as for Leni
        toolbox.register("mutate", mutDNN, mutation_rate_params_wb=params["dnn_mut_pb_wb"], mutation_eta=params["dnn_mut_eta_wb"], mutation_rate_add_conn=params["dnn_mut_pb_add_conn"], mutation_rate_del_conn=params["dnn_mut_pb_del_conn"], mutation_rate_add_node=params["dnn_mut_pb_add_node"], mutation_rate_del_node=params["dnn_mut_pb_del_node"])
    else:
        raise RuntimeError("Unknown genotype type %s" % geno_type)

    #Common elements - selection and evaluation
    toolbox.register("select", tools.selNSGA2)
        
    toolbox.register("evaluate", evaluate)
    
    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)
    
    return toolbox


def seedes(evaluate, params, pool):
    
    print("Seed-ES Novelty search algorithm")

    toolbox=build_toolbox_seedes(evaluate,params,pool)

    population = toolbox.population(n=params["pop_size"])

    lambda_ = params["seed_lambda"]

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

    nb_eval=0

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    nb_eval+=len(invalid_ind)
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
            samples = algorithms.varOr([population[s]], toolbox, lambda_, 0, params["mutpb"])
            samples_per_seed[s]=samples
            all_samples+=samples

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in all_samples if not ind.fitness.valid]
        nb_eval+=len(invalid_ind)
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fit = fit[0]
            #ind.fitness.values = fit[0]
#            ind.parent_bd=ind.bd
            ind.bd=listify(fit[1])

        archive=updateNovelty(all_samples,all_samples,archive,params)

        # Compute the fitness values for each seed: uniformity and cumulated novelty
        for s in range(len(population)):
            #print("s="+str(s)+ " population[s]="+str(population[s]))
            cumul=0
            for i in samples_per_seed[s]:
                cumul+=i.novelty
            population[s].fitness.values=(cumul_distance(samples_per_seed[s]), cumul)
            population[s].fit=(cumul_distance(samples_per_seed[s]), cumul)
            population[s].novelty=-1 # to simplify stats
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

        if (("eval_budget" in params.keys()) and (params["eval_budget"]!=-1) and (nb_eval>=params["eval_budget"])): 
            params["nb_gen"]=gen
            terminates=True
        else:
            terminates=False

        dump_data(population, gen, params, prefix="population", attrs=["all"], force=terminates)
        dump_data(population, gen, params, prefix="bd", complementary_name="population", attrs=["bd"], force=terminates)
        dump_data(all_samples, gen, params, prefix="bd", complementary_name="all_samples", attrs=["bd"], force=terminates)
        dump_data(archive.get_content_as_list(), gen, params, prefix="archive", attrs=["all"], force=terminates)

        generate_evolvability_samples(params, population, gen, toolbox)
        
        # Update the statistics with the new population
        record = params["stats"].compile(population) if params["stats"] is not None else {}
#        record_offspring = params["stats_offspring"].compile(all_samples) if params["stats_offspring"] is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record) #, **record_offspring)
        if (verbosity(params)):
            print(logbook.stream)

        for ind in population:
            ind.evolvability_samples=None

        if (terminates):
            break


        population+=all_samples[-len(population):] # update the number of elements to add

            
    return population, archive, logbook, nb_eval

  
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

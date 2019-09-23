#!/usr/bin python -w

import random
from scipy.spatial import KDTree
import numpy as np
import datetime
import os
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

# ### Novelty-based Evolution Strategies

class NovArchive:
    """Archive used to compute novelty scores."""
    def __init__(self, lbd, k=15):
        self.all_bd=lbd
        self.kdtree=KDTree(self.all_bd)
        self.k=k
        print("Archive constructor. size = %d"%(len(self.all_bd)))
        
    def update(self,new_bd):
        oldsize=len(self.all_bd)
        self.all_bd=self.all_bd + new_bd
        self.kdtree=KDTree(self.all_bd)
        print("Archive updated, old size = %d, new size = %d"%(oldsize,len(self.all_bd)))
    def get_nov(self,bd, population=[]):
        dpop=[]
        for ind in population:
            dpop.append(np.linalg.norm(np.array(bd)-np.array(ind.bd)))
        darch,ind=self.kdtree.query(np.array(bd),self.k)
        d=dpop+list(darch)
        d.sort()
        if (d[0]!=0):
            print("WARNING in novelty search: the smallest distance should be 0 (distance to itself). If you see it, you probably try to get the novelty with respect to a population your indiv is not in. The novelty value is then the sum of the distance to the k+1 nearest divided by k.")
        return sum(d[:self.k+1])/self.k # as the indiv is in the population, the first value is necessarily a 0.

    def size(self):
        return len(self.all_bd)
    
def updateNovelty(population, offspring, archive, k=15, add_strategy="random", _lambda=6, verbose=False):
   """"Update the novelty criterion (including archive update) 

   Implementation of novelty search following (Gomes, J., Mariano, P., & Christensen, A. L. (2015, July). Devising effective novelty search algorithms: A comprehensive empirical study. In Proceedings of GECCO 2015 (pp. 943-950). ACM.).
   :param population: is the set of indiv for which novelty needs to be computed
   :param offspring: is the set of new individuals that need to be taken into account to update the archive (may be the same as population, but it may also be different as population may contain the set of parents)
   :param k: is the number of nearest neighbors taken into account
   :param add_strategy: is either "random" (a random set of indiv is added to the archive) or "novel" (only the most novel individuals are added to the archive).
   :param _lambda: is the number of individuals added to the archive for each generation
   The default values correspond to the one giving the better results in the above mentionned paper.

   The function returns the new archive
   """
   
   # Novelty scores updates
   if (archive) and (archive.size()>=k):
       if (verbose):
           print("Update Novelty. Archive size=%d"%(archive.size())) 
       for ind in population:
           ind.novelty=archive.get_nov(ind.bd, population)
   else:
       if (verbose):
           print("Update Novelty. Initial step...") 
       for ind in population:
           ind.novelty=0.

   if (verbose):
       print("Fitness (novelty): ",end="") 
       for ind in population:
           print("%.2f, "%(ind.novelty),end="")
       print("")
   if (len(offspring)<_lambda):
       print("ERROR: updateNovelty, lambda(%d)<offspring size (%d)"%(_lambda, len(offspring)))
       return None

   lbd=[]
   # Update of the archive
   if(add_strategy=="random"):
       l=list(range(len(offspring)))
       random.shuffle(l)
       if (verbose):
           print("Random archive update. Adding offspring: "+str(l[:_lambda])) 
       lbd=[offspring[l[i]].bd for i in range(_lambda)]
   elif(add_strategy=="novel"):
       soff=sorted(offspring,lambda x:x.novelty)
       ilast=len(offspring)-_lambda
       lbd=[soff[i].bd for i in range(ilast,len(soff))]
       if (verbose):
           print("Novel archive update. Adding offspring: ")
           for offs in soff[iLast:len(soff)]:
               print("    nov="+str(offs.novelty)+" fit="+str(offs.fitness.values)+" bd="+str(offs.bd))
   else:
       print("ERROR: updateNovelty: unknown add strategy(%s), valid alternatives are \"random\" and \"novel\""%(add_strategy))
       return None
       
   if(archive==None):
       archive=NovArchive(lbd,k)
   else:
       archive.update(lbd)

   return archive

def generate_evolvability_samples(run_name, population, toolbox, evolvability_nb_samples, evolvability_period, gen, cxpb, mutpb):
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

def generate_dumps(run_name, dump_period_bd, dump_period_pop, pop1, pop2, gen, pop1label="population", pop2label="offspring", archive=None, logbook=None):
    if(dump_period_bd and (gen % dump_period_bd == 0)): # Dump behavior descriptors
        dump_bd=open(run_name+"/bd_%04d_%s.log"%(gen,pop1label),"w")
        for ind in pop1:
            dump_bd.write(" ".join(map(str,ind.bd))+"\n")
        dump_bd.close()
        if (pop2 is not None):
            dump_bd=open(run_name+"/bd_%04d_%s.log"%(gen,pop2label),"w")
            for ind in pop2:
                dump_bd.write(" ".join(map(str,ind.bd))+"\n")
            dump_bd.close()
    
    if(dump_period_pop and(gen % dump_period_pop == 0)): # Dump populatio    if dump_period_pop:
        if(pop1 is not None):
            dump_pop(pop1, gen, run_name, pop1label)
        if(pop2 is not None):
            dump_pop(pop2, gen,run_name, pop2label)
        if (archive is not None):
            dump_archive(archive, gen,run_name)
        if (logbook is not None):
            dump_logbook(logbook, gen,run_name)


## DEAP compatible algorithm
def noveltyEa(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,k,add_strategy,lambdaNov,
                          stats=None, stats_offspring=None, halloffame=None, dump_period_bd=1, dump_period_pop=10, evolvability_period=50, evolvability_nb_samples=0, verbose=__debug__, run_name="runXXX", variant="NS"):
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

    if ("+" in variant):
        emo=True
    else:
        emo=False
        
    if(halloffame!=None):
        print("WARNING: the hall of fame argument is ignored in the Novelty Search Algorithm")
    
    print("     variant="+variant)
    print("     lambda=%d, mu=%d, cxpb=%.2f, mutpb=%.2f, ngen=%d, k=%d, lambda_nov=%d"%(lambda_,mu,cxpb,mutpb,ngen,k,lambdaNov))

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
        ind.fit = fit[0] # fit is an attribute just used to store the fitness value
        ind.parent_bd=None
        ind.bd=listify(fit[1])

    for ind in population:
        ind.am_parent=0
        
    archive=updateNovelty(population,population,None,k,add_strategy,lambdaNov)

    for ind in population:
        if (emo):
            if (variant == "NS+Fit"):
                ind.fitness.values = (ind.novelty, ind.fit)
            elif (variant == "NS+BDDistP"):
                ind.fitness.values = (ind.novelty, 0)
            elif (variant == "NS+Fit+BDDistP"):
                ind.fitness.values = (ind.novelty, ind.fit, 0)
            else:
                print("WARNING: unknown variant: "+variant)
                ind.fitness.values=ind.fit
        else:
            ind.fitness.values=ind.fit
    
    gen=0    

    # Do we look at the evolvability of individuals (WARNING: it will make runs much longer !)
    generate_evolvability_samples(run_name, population, toolbox, evolvability_nb_samples, evolvability_period, gen, cxpb, mutpb)

    record = stats.compile(population) if stats is not None else {}
    record_offspring = stats_offspring.compile(population) if stats_offspring is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record, **record_offspring)
    if verbose:
        print(logbook.stream)
    
    generate_dumps(run_name, dump_period_bd, dump_period_pop, population, None, gen, pop1label="population", archive=None, logbook=None)

    for ind in population:
        ind.evolvability_samples=None # To avoid memory to inflate too much..
        
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

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

        
        archive=updateNovelty(pq,offspring,archive,k,add_strategy,lambdaNov)

        for ind in pq:
            if (emo):
                if (variant == "NS+Fit"):
                    ind.fitness.values = (ind.novelty, ind.fit)
                elif (variant == "NS+BDDistP"):
                    if (ind.parent_bd is None):
                        bddistp=0
                    else:
                        bddistp=np.linalg.norm(np.array(ind.bd) - np.array(ind.parent_bd))
                    ind.fitness.values = (ind.novelty, bddistp)
                elif (variant == "NS+Fit+BDDistP"):
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

        print("Gen %d"%(gen))

        
        # Select the next generation population
        if ("," in variant):
            population[:] = toolbox.select(offspring, mu)        
        else:
            population[:] = toolbox.select(pq, mu)        

        generate_dumps(run_name, dump_period_bd, dump_period_pop, population, offspring, gen, pop1label="population", pop2label="offspring", archive=archive, logbook=logbook)
        
        generate_evolvability_samples(run_name, population, toolbox, evolvability_nb_samples, evolvability_period, gen, cxpb, mutpb)
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        record_offspring = stats_offspring.compile(offspring) if stats_offspring is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record, **record_offspring)
        if verbose:
            print(logbook.stream)

        for ind in population:
            ind.evolvability_samples=None

            
    return population, archive, logbook




def NovES(evaluate,myparams,pool=None, run_name="runXXX", geno_type="realarray"):
    """Novelty-based Mu plus lambda ES."""

    params={"IND_SIZE":1, 
            "CXPB":0, # crossover probility
            "MUTPB":0.5, # probability to mutate an individual
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
    if (params["VARIANT"] == "NS"):
        toolbox.register("select", tools.selBest, fit_attr='novelty')
    elif (params["VARIANT"] == "Fit"):
        toolbox.register("select", tools.selBest, fit_attr='fitness')
    else:
        toolbox.register("select", tools.selNSGA2)
        
    toolbox.register("evaluate", evaluate)
    
    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)
    

    pop = toolbox.population(n=params["MU"])
    
    rpop, archive, logbook = noveltyEa(pop, toolbox, mu=params["MU"], lambda_=params["LAMBDA"], cxpb=params["CXPB"], mutpb=params["MUTPB"], ngen=params["NGEN"], k=params["K"], add_strategy=params["ADD_STRATEGY"], lambdaNov=params["LAMBDANOV"],stats=params["STATS"], stats_offspring=params["STATS_OFFSPRING"], halloffame=None, evolvability_nb_samples=params["EVOLVABILITY_NB_SAMPLES"], evolvability_period=params["EVOLVABILITY_PERIOD"], dump_period_bd=params["DUMP_PERIOD_BD"], dump_period_pop=params["DUMP_PERIOD_POP"], verbose=False, run_name=run_name, variant=params["VARIANT"])
        
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

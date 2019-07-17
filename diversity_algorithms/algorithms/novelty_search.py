#!/usr/bin python -w

import random
from scipy.spatial import KDTree
import numpy as np
import datetime
import os
creator = None
def set_creator(cr):
    global creator
    creator = cr

from deap import tools, base, algorithms

period_dump_pop = 10 # Population dump interval - 0 to disable

# in_ipython = False
# try:
#     get_ipython
#     print("We are in iPython / Jupyter")
#     in_ipython = True
# except NameError:
#     print("We are NOT in iPython / Jupyter")
#     pass


# if(not in_ipython):
#     from scoop import futures

# ### Novelty-based Evolution Strategies

class NovArchive:
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
    def get_nov(self,bd):
        d,ind=self.kdtree.query(np.array(bd),self.k)
        return sum(d)/self.k

    def size(self):
        return len(self.all_bd)
    
def updateNovelty(population, offspring, archive, k=15, add_strategy="random", _lambda=6):
   """" Implementation of novelty search following (Gomes, J., Mariano, P., & Christensen, A. L. (2015, July). Devising effective novelty search algorithms: A comprehensive empirical study. In Proceedings of GECCO 2015 (pp. 943-950). ACM.).
   - population is the set of indiv for which novelty needs to be computed
   - offspring is the set of new individuals that need to be taken into account to update the archive (may be the same as population, but it may also be different as population may contain the set of parents)
   - k is the number of nearest neighbors taken into account
   - add_strategy is either "random" (a random set of indiv is added to the archive) or "novel" (only the most novel individuals are added to the archive).
   - _lambda is the number of individuals added to the archive for each generation
   The default values correspond to the one giving the better results in the above mentionned paper.

   The function returns the new archive
   """
   
   # Novelty scores updates
   if (archive) and (archive.size()>=k):
       print("Update Novelty. Archive size=%d"%(archive.size()))
       for ind in population:
           ind.novelty=archive.get_nov(ind.fitness.bd)
   else:
       print("Update Novelty. Initial step...")
       for ind in population:
           ind.novelty=0.

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
       print("Random archive update. Adding offspring: "+str(l[:_lambda]))
       lbd=[offspring[l[i]].fitness.bd for i in range(_lambda)]
   elif(add_strategy=="novel"):
       soff=sorted(offspring,lambda x:x.novelty)
       ilast=len(offspring)-_lambda
       lbd=[soff[i].fitness.bd for i in range(ilast,len(soff))]
       print("Novel archive update. Adding offspring: ")
       for offs in soff[iLast:len(soff)]:
           print("    nov="+str(offs.novelty)+" fit="+str(offs.fitness.values)+" bd="+str(offs.fitness.bd))
   else:
       print("ERROR: updateNovelty: unknown add strategy(%s), valid alternatives are \"random\" and \"novel\""%(add_strategy))
       return None
       
   if(archive==None):
       archive=NovArchive(lbd,k)
   else:
       archive.update(lbd)

   return archive


def dump_pop(pop, gen, run_name="runXXX"):
    out_dict = {"gen": gen, "size": len(pop)}
    for (i,ind) in enumerate(pop):
        out_dict["geno_%d" % i] = np.array(ind)
        if(ind.fitness.valid):
            out_dict["fitness_%d" % i] = ind.fitness.values
            out_dict["novelty_%d" % i] = ind.novelty
            out_dict["bd_%d" % i] = ind.fitness.bd
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    np.savez(run_name+"/pop_gen%d.npz" % gen, **out_dict) 

## DEAP compatible algorithm
def noveltyEaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,k,add_strategy,lambdaNov,
                   stats=None, halloffame=None, verbose=__debug__):

    d=datetime.datetime.today()
    run_name=d.strftime("%Y_%m_%d-%H:%M:%S")
    nb=0
    not_created=True
    while(not_created):
        try:
            os.mkdir(run_name+"_%d"%nb)
            run_name+="_%d"%nb
            not_created=False
        except OSError:
            nb+=1

    print("Run name: "+run_name)

    print("     lambda=%d, mu=%d, cxpb=%.2f, mutpb=%.2f, ngen=%d, k=%d, lambda_nov=%d"%(lambda_,mu,cxpb,mutpb,ngen,k,lambdaNov))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # fit is a list of fitness (that is also a list) and behavior descriptor

    # some sanity check
    #fit=fitnesses[0]
    #if (len(fit)<2) or (not isinstance(fit[0],list)):
    #    print("Error: the evaluation should return a behavior descriptor besides the fitness")
    #    return None
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0]
        ind.fitness.bd = fit[1]

    archive=updateNovelty(population,population,None,k,add_strategy,lambdaNov)
             
    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    gen=0    
    dump_bd=open(run_name+"/bd_%04d.log"%gen,"w")
    for ind in population:
        dump_bd.write(" ".join(map(str,ind.fitness.bd))+"\n")
    dump_bd.close()
    
    if period_dump_pop:
        dump_pop(population, 0, run_name) # Dump initial pop
    
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit[0] 
            ind.fitness.bd = fit[1]

        pq=population+offspring
        archive=updateNovelty(pq,offspring,archive,k,add_strategy,lambdaNov)

        dump_bd=open(run_name+"/bd_%04d.log"%gen,"w")
        for ind in offspring:
            dump_bd.write(" ".join(map(str,ind.fitness.bd))+"\n")
        dump_bd.close()

        if(period_dump_pop and(gen % period_dump_pop == 0)): # Dump population
            dump_pop(pq, gen,run_name)
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        print("Gen %d : whole pop:"%(gen),end="")
        for ind in pq:
            print(" %02.2f "%(ind.novelty), end="")
        print("")

        # Select the next generation population
        population[:] = toolbox.select(pq, mu)

        print("Gen %d : new pop:"%(gen),end="")
        for ind in population:
            print(" %02.2f "%(ind.novelty), end="")
        print("")
        
        
        
        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook, run_name

 # Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategyMin(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

def checkStrategyMinMax(minstrategy,maxstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
                    if s > maxstrategy:
                        child.strategy[i] = maxstrategy
            return children
        return wrappper
    return decorator



def NovES(evaluate,myparams,pool=None):
    """Novelty-based Mu plus lambda ES."""

    params={"IND_SIZE":1, 
            "CXPB":1,
            "MUTPB":1,
            "NGEN":1000,
            "STATS":None,
            "MIN": 0,
            "MAX": 1,
            "MIN_STRATEGY": 0,
            "MAX_STRATEGY": 1,
            "MU": 20,
            "LAMBDA": 100,
            "ALPHA": 0.1,
            "C": 1.0,
            "INDPB": 0.03,
            "K":15,
            "ADD_STRATEGY":"random",
            "LAMBDANOV":6
           }
    
    
    for key in myparams.keys():
        params[key]=myparams[key]

         
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
        params["IND_SIZE"], params["MIN"], params["MAX"], params["MIN_STRATEGY"], params["MAX_STRATEGY"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=params["ALPHA"])
    toolbox.register("mutate", tools.mutESLogNormal, c=params["C"], indpb=params["INDPB"])
    toolbox.register("select", tools.selBest, fit_attr='novelty')
    toolbox.register("evaluate", evaluate)

    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)
    
#    toolbox.decorate("mate", checkStrategy(params["MIN_STRATEGY"], params["MAX_STRATEGY"]))
#    toolbox.decorate("mutate", checkStrategy(params["MIN_STRATEGY"], params["MAX_STRATEGY"]))
    toolbox.decorate("mate", checkStrategyMin(params["MIN_STRATEGY"]))
    toolbox.decorate("mutate", checkStrategyMin(params["MIN_STRATEGY"]))

    pop = toolbox.population(n=params["MU"])
    hof = tools.HallOfFame(1)
    
    rpop, logbook, run_name = noveltyEaMuPlusLambda(pop, toolbox, mu=params["MU"], lambda_=params["LAMBDA"], cxpb=params["CXPB"], mutpb=params["MUTPB"], ngen=params["NGEN"], k=params["K"], add_strategy=params["ADD_STRATEGY"], lambdaNov=params["LAMBDANOV"],stats=params["STATS"], halloffame=hof, verbose=False)
        
    return rpop, logbook, hof, run_name
  
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

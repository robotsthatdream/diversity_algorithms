
# coding: utf-8
import numpy
import random
import math
from deap import creator, tools, base

in_ipython = False
try:
    get_ipython
    print("We are in iPython / Jupyter")
    in_ipython = True
except NameError:
    print("We are NOT in iPython / Jupyter")
    pass

creator = None
def set_creator(cr):
    global creator
    creator = cr


# # Evolutionary Algorithms

#
def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    child[i] = numpy.clip(child[i], min, max)
            return offspring
        return wrapper
    return decorator



# ## Single Objective Evolutionary Algorithms

# ### Microbial GA
# 
# Introduction with the most basic Evolutionary Algorithm
def microbial_GA(evaluate,myparams):
    """Steady state simple GA: the looser of a tournament of size 2 is crossed with the winner and mutated."""
    
    params={"IND_SIZE":1, 
           "POP_SIZE":100,
           "CXPB":1,
           "MUTPB":1,
           "NGEN":1000,
           "STATS":None,
           "MIN": 0,
           "MAX": 1,
           "MUT_SIGMA": 1}
    
    for key in myparams.keys():
        params[key]=myparams[key]
    
    # Initializing the different algorithms' parts
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.uniform,params["MIN"],params["MAX"]) 
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attribute, n=params["IND_SIZE"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxBlend)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=params["MUT_SIGMA"], indpb=0.1)
    toolbox.register("select", tools.selRandom)
    toolbox.register("evaluate", evaluate)
    
    toolbox.decorate("mate", checkBounds(params["MIN"], params["MAX"]))
    toolbox.decorate("mutate", checkBounds(params["MIN"], params["MAX"]))
    
    logbook = tools.Logbook()
    
    pop = toolbox.population(params["POP_SIZE"])

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    hall_of_fame=tools.HallOfFame(10)
    
    for g in range(params["NGEN"]):
        # Select the next generation individuals
        offspring = toolbox.select(pop, 2)
    
        if(creator.MyFitness(offspring[0].fitness.values)>creator.MyFitness(offspring[1].fitness.values)):
            winner=toolbox.clone(offspring[0])
            looser=offspring[1]
        else:
            winner=toolbox.clone(offspring[1])
            looser=offspring[0]

        #print("Sel indiv: "+str(offspring)+" Winner: "+str(winner)+" Looser: "+str(looser)),
        del looser.fitness.values
        
        if random.random() < params["CXPB"]:
            toolbox.mate(winner, looser, 0.5)

        if random.random() < params["MUTPB"]:
            toolbox.mutate(looser)

        fitness = toolbox.evaluate(looser)
        looser.fitness.values = fitness

        #print(" New indiv: "+str(looser))
        record = params["STATS"].compile(pop)
        logbook.record(gen=g, evals=g, **record)

        hall_of_fame.update(pop)
        
    return pop,logbook,hall_of_fame



# ### Evolution Strategies


from deap import algorithms

import multiprocessing

#if(not in_ipython):
#    from scoop import futures


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


#from joblib_hack import mymap


def ES(evaluate,myparams,pool=None):
    """Mu plus lambda ES."""
    
    params={"IND_SIZE":1, 
            "POP_SIZE":100,
            "CXPB":1,
            "MUTPB":1,
            "NGEN":1000,
            "STATS":stats,
            "MIN": 0,
            "MAX": 1,
            "MIN_STRATEGY":0,
            "MAX_STRATEGY":1,
            "MU": 100,
            "LAMBDA": 1000,
            "ALPHA": 0.1,
            "C": 1.0,
            "INDPB": 0.03,
            "TOURNSIZE":3,
            "VARIANT": "+"
           }
    
    
    for key in myparams.keys():
        params[key]=myparams[key]
    
    toolbox = base.Toolbox()
    toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
        params["IND_SIZE"], params["MIN"], params["MAX"], params["MIN_STRATEGY"], params["MAX_STRATEGY"])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxESBlend, alpha=params["ALPHA"])
    toolbox.register("mutate", tools.mutESLogNormal, c=params["C"], indpb=params["INDPB"])
    toolbox.register("select", tools.selTournament, tournsize=params["TOURNSIZE"])
    toolbox.register("evaluate", evaluate)

    # Parallelism
#    if(not in_ipython):
#        toolbox.register("map", futures.map)

#    toolbox.register("map", mymap)
    if(pool):
        toolbox.register("map", pool.map)

    
#    toolbox.decorate("mate", checkStrategy(params["MIN_STRATEGY"], params["MAX_STRATEGY"]))
#    toolbox.decorate("mutate", checkStrategy(params["MIN_STRATEGY"], params["MAX_STRATEGY"]))
    toolbox.decorate("mate", checkStrategyMin(params["MIN_STRATEGY"]))
    toolbox.decorate("mutate", checkStrategyMin(params["MIN_STRATEGY"]))

    pop = toolbox.population(n=params["MU"])
    hof = tools.HallOfFame(1)
    
    if (params["VARIANT"]=="+"):
        print("Mu+Lambda ES")
        rpop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=params["MU"], lambda_=params["LAMBDA"], 
                                                  cxpb=params["CXPB"], mutpb=params["MUTPB"], ngen=params["NGEN"], stats=params["STATS"], halloffame=hof, verbose=False)
    else:
        print("Mu,Lambda ES")
        rpop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=params["MU"], lambda_=params["LAMBDA"], 
                                                   cxpb=params["CXPB"], mutpb=params["MUTPB"], ngen=params["NGEN"], stats=params["STATS"], halloffame=hof, verbose=False)
        
    return rpop, logbook, hof
  
    
    
   


# ### CMA-ES

from deap import algorithms
from deap import base
from deap import benchmarks
from deap import cma
from deap import creator
from deap import tools

def CMA_ES(evaluate,myparams, pool=None, run_name="runCMAES_XXX"):
    """CMA-ES."""

    numpy.random.seed()
    params={"IND_SIZE":1, 
            "NGEN":1000,
            "STATS":None,
            "CENTROID":[0],
            "SIGMA": 5
           }
    
    
    for key in myparams.keys():
        params[key]=myparams[key]

    N=params["IND_SIZE"]
        
    if ("LAMBDA" not in myparams.keys()):
        if (("MU" in myparams.keys()) and (params["MU"] is not None)):
            params["LAMBDA"]=int(2*params["MU"])
        else:
            params["LAMBDA"]=int(4+3*math.log(N)) #4+3*math.floor(3*math.log(params["IND_SIZE"])),
    
    if (("MU" not in myparams.keys()) or (params["MU"] is None)):
        params["MU"]=int(params["LAMBDA"]/2)
    
        
    if(len(params["CENTROID"])!=N):
        if(len(params["CENTROID"])==1):
            # if the centroid has not been initialized to the right size, we update it
            params["CENTROID"]=params["CENTROID"]*N
        else:
            print("ERROR: the centroid has not been properly set.")
            return [],tools.Logbook(),tools.HallOfFame(1)
    
    toolbox = base.Toolbox()


    
    # Parallelism
    if(pool):
        toolbox.register("map", pool.map)

    
    toolbox.register("evaluate", evaluate)

    print("CMA-ES, lambda="+str(params["LAMBDA"])+" mu="+str(params["MU"]))
    
    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES    
    strategy = cma.Strategy(centroid=params["CENTROID"], sigma=params["SIGMA"],lambda_=params["LAMBDA"], mu=params["MU"])
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    
        
    rpop,logbook = algorithms.eaGenerateUpdate(toolbox, ngen=params["NGEN"], stats=params["STATS"], halloffame=hof, verbose=False)    
        
        
    return rpop, logbook, hof



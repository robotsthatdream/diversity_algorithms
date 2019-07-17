# coding: utf-8

# This set of functions tries to characterize set of individuals

import numpy as np
import math as m

from deap import tools, base, algorithms

def coverage(x, min_x, max_x, nb_bin):
    assert(len(min_x)==len(max_x)),"Problem with the size of min and max"
    dim=len(min_x)
    grid=np.zeros(shape=[nb_bin]*dim,dtype=np.int)
    for px in x:
        assert(len(px)==len(max_x)),"Problem with the size of a point:  "+str(px)+" min_x="+str(min_x)
        ix=[0]*dim
        for i in range(dim):
            ix[i] = m.floor((px[i]-min_x[i])/(max_x[i]-min_x[i]+0.01)*nb_bin)
        #print("Adding a point to "+str(ix))
        grid[tuple(ix)]+=1
    #print(str(grid))
    nb_points=nb_bin**dim
    return float(np.count_nonzero(grid))/float(nb_points)

def radius(x):
    max_d=0
    d=[]
    for i1 in range(len(x)):
        for i2 in range(i1+1,len(x)):
            px1=x[i1]
            px2=x[i2]
            dist=np.linalg.norm(px1-px2)
            d.append(dist)
    return max(d),np.percentile(d,75),np.average(d),np.median(d)

# Generating a set of individuals from a current population
def sampleFromPop(population, toolbox, lambda_, cxpb, mutpb, verbose=__debug__):
    # Vary the population
    offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0] 
        ind.fitness.bd = fit[1]
        
    return offspring


if __name__ == '__main__':
    # Some tests
    x=[np.array([0,0]),np.array([1,2]),np.array([3,4]),np.array([5,5])]
    min_x=[0,0]
    max_x=[5,5]
    nb_bin=5
    cov=coverage(x,min_x,max_x,nb_bin)
    print("Coverage: "+str(cov))
    print("Radius: "+str(radius(x)))

# coding: utf-8

# This set of functions tries to characterize set of individuals

import numpy as np
import numpy.ma as ma
import math as m
from functools import *
from deap import tools, base, algorithms

def build_grid(min_x, max_x, nb_bin):
    """Build an outcome space grid.

    Build an outcome space grid:
    :param min_x: minimum values on each dimension
    :param max_x: maximum values on each dimension
    :param nb_bin: number of bins per dimensions. Vector of the nubmer of bins for each dimension. If scalar, we will assume the same dimension for each dimension.
    :returns: the generated grid
    """
    assert(len(min_x)==len(max_x)),"Problem with the size of min and max"
    dim=len(min_x)
    if(hasattr(nb_bin, '__iter__')):
        lnb_bin=nb_bin
    else:
        lnb_bin=[nb_bin]*dim
    grid=np.zeros(shape=lnb_bin,dtype=np.int)
    return grid

def update_grid(grid,min_x, max_x, x):
    """Update a grid with the given points.

    Update a grid with the given points:
    :param grid: grid to update (None if it is to be built)
    :param min_x: minimum values on each dimension
    :param max_x: maximum values on each dimension
    :param x: set of points to take into account
    """
    assert(len(min_x)==len(max_x)),"Problem with the size of min and max"
    dim=len(min_x)
    nb_bin=np.shape(grid)
    for px in x:
        assert(len(px)==len(max_x)),"Problem with the size of a point:  "+str(px)+" min_x="+str(min_x)
        ix=[0]*dim
        for i in range(dim):
            ix[i] = m.floor((px[i]-min_x[i])/(max_x[i]-min_x[i]+0.01)*nb_bin[i])
        #print("Adding a point to "+str(ix))
        grid[tuple(ix)]+=1

def coverage(grid):
    """Return the coverage, the ratio of non zero cells on the total number of cells."""
    nb_bin=np.shape(grid)
    nbc=reduce(lambda x,y:x*y,nb_bin,1)
    return float(np.count_nonzero(grid))/float(nbc)

def jensen_shannon_distance(grid1,grid2):
    grid3=grid1+grid2
    grid4=grid1*np.log(2*grid1/grid3)+grid2*np.log(2*grid2/grid3)
    grid5=ma.masked_invalid(grid4)
    return grid5.sum()
    
def radius(x):
    """Return statistics about the distances between the points in x.

    Return statistics about the distances between the points in x. Values returned:
    :returns:
       maximum distance between points in x
       75 percentile of the distances
       average value of the distances
       median value of the distances
    """
    max_d=0
    d=[]
    for i1 in range(len(x)):
        for i2 in range(i1+1,len(x)):
            px1=x[i1]
            px2=x[i2]
            dist=np.linalg.norm(px1-px2)
            d.append(dist)
    return max(d),np.percentile(d,75),np.average(d),np.median(d)

def sample_from_pop(population, toolbox, lambda_, cxpb, mutpb):
    """Generate a set of individuals from a population.

    Generate a set of individuals from a population. Parameters:
    :param population: the population to start from
    :param toolbox: the DEAP framework toolbox that contains the variation operators and the evaluation function
    :param lambda_: number of individuals to generate
    :param cxpb: cross-over probability (set to 0 to test only mutation)
    :param mutbp: mutation probability

    WARNING: if cxpb>0, the population size needs to be >2 (it thus won't work to sample individuals from a single individual)
    """
    # Vary the population
    offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit[0] 
        ind.fitness.bd = fit[1]
        
    return offspring


def density(grid):
    """Return the density of the population.

    Return the density of the population.
    """
    print("TODO...")
    pass


if __name__ == '__main__':
    # Some tests
    pass

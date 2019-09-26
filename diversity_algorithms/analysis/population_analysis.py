# coding: utf-8

# This set of functions tries to characterize set of individuals

import random
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

def entropy(grid):
    """Return the entropy of the grid (close to 1 == uniform)."""
    nb_bin=np.shape(grid)
    nbc=reduce(lambda x,y:x*y,nb_bin,1)
    n=np.sum(grid)
    #print("Number of cells: %d, number of points: %d"%(nbc, n))
    if (n==0):
        return float('NaN')
    entropy=np.array(grid)
    entropy=entropy/float(n)
    entropy=entropy*np.log(entropy)
    return -1.*float(np.sum(entropy))/float(np.log(nbc))

def coverage(grid):
    """Return the coverage, the ratio of non zero cells on the total number of cells."""
    nb_bin=np.shape(grid)
    nbc=reduce(lambda x,y:x*y,nb_bin,1)
    return float(np.count_nonzero(grid))/float(nbc)

def get_coverage(min_x,max_x, nb_bin, x):
    """Getting the coverage of a given set of points.
    """
    grid=build_grid(min_x, max_x, nb_bin)
    update_grid(grid,min_x, max_x, x)
    return coverage(grid)
    
def generate_uniform_grid(grid):
    """Generate a uniform grid with the same shape and same number of points than grid."""
    grid_uniform=np.ones(np.shape(grid))
    nb_bin=np.shape(grid)    
    nbc=reduce(lambda x,y:x*y,nb_bin,1)
    nbsamples=np.sum(grid)
    grid_uniform=nbsamples/nbc*grid_uniform
    if (nbsamples<nbc):
        print("Warning, too few samples to estimate coverage: nbsamples=%d, nbcells=%d"%(nbsamples,nbc))
    return grid_uniform

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
        ind.bd = fit[1]
        ind.evolvability_samples=None # SD: required, otherwise, the memory usage explodes... I do not understand why yet.
        
    return offspring


def density(grid):
    """Return the density of the population.

    Return the density of the population.
    """
    print("TODO...")
    pass


if __name__ == '__main__':


    random.seed()
    
    min_x=[0,0]
    max_x=[600,600]
    nb_bin=10
    grid=build_grid(min_x, max_x, nb_bin)
    nbpts=10000
    x=[[random.uniform(min_x[0], max_x[0]), random.uniform(min_x[1], max_x[1])] for p in range(nbpts)]
    update_grid(grid,min_x, max_x, x)

    grid2=build_grid(min_x, max_x, nb_bin)
    x2=[[random.uniform(min_x[0], max_x[0]/2), random.uniform(min_x[1], max_x[1]/2)] for p in range(nbpts)]
    update_grid(grid2,min_x, max_x, x2)


    uniform_grid=generate_uniform_grid(grid)
    print("Coverage of grid: %.2f, coverage of the uniform grid: %.2f of grid2: %.2f"%(coverage(grid), coverage(uniform_grid), coverage(grid2)))
    print("Jensen-Shannon distance between the 2: %f"%(jensen_shannon_distance(grid,uniform_grid)))
    print("Jensen-Shannon distance between grid2 and uniform grid: %f"%(jensen_shannon_distance(grid2,uniform_grid)))
        
    

from deap import tools
import numpy

from diversity_algorithms.analysis.population_analysis import *

# useful classes
class Perc:
    def __init__(self,val):
        self.val=val
    def __call__(self,l):
        return numpy.percentile(l,self.val)


# fitness-based statistics
stats_fitness = tools.Statistics(key=lambda ind: ind.fitness.values)
    
stats_fitness.register("median", numpy.median)
stats_fitness.register("std", numpy.std)
stats_fitness.register("min", numpy.min)
stats_fitness.register("max", numpy.max)
stats_fitness.register("perc25", Perc(25))
stats_fitness.register("perc75", Perc(75))

# novelty-based statistics
stats_novelty = tools.Statistics(key=lambda ind: ind.fitness.novelty)
    
stats_novelty.register("median", numpy.median)
stats_novelty.register("std", numpy.std)
stats_novelty.register("min", numpy.min)
stats_novelty.register("max", numpy.max)
stats_novelty.register("perc25", Perc(25))
stats_novelty.register("perc75", Perc(75))

# Multi stats (compute several statistical values during the same compile call)

# Fitness + novelty
mstats_fit_nov = tools.MultiStatistics(fitness=stats_fitness, novelty=stats_novelty)
mstats_fit_nov.register("median", numpy.median)
mstats_fit_nov.register("std", numpy.std)
mstats_fit_nov.register("min", numpy.min)
mstats_fit_nov.register("max", numpy.max)
mstats_fit_nov.register("perc25", Perc(25))
mstats_fit_nov.register("perc75", Perc(75))

### Statistics on the coverage

## Useful functions
def get_updated_coverage(grid,lbd,x,min_x=None, max_x=None, gen_window=10):
    """Compute the coverage after having added a new set of points

    Compute the coverage after having added a new set of points:
    :param grid: the grid that is progressively filled
    :param lbd: the list of behavior descriptors (one list per generation), used to take into account only a window of generations, and not all of them
    :param x: the population to look at
    :param min_x: the minimum value of x
    :param max_x: the maximum value of x
    :param gen_window: the number of generations to take into account (the last gen_window are taken into account)
    :returns: the coverage and the jensen-shannon distance to a uniform distribution, or (None, None) is there are not enough generations
    """
    bdx=[ind.fitness.bd for ind in x]
    ready=True
    if(gen_window>0):
        lbd.append(bdx)
        if(len(lbd)<gen_window):
            ready=False
        while(len(lbd)>gen_window):
            grid_to_remove=np.zeros(np.shape(grid))
            update_grid(grid_to_remove,min_x, max_x,lbd[0])
            grid=grid-grid_to_remove
            lbd.pop(0)
    update_grid(grid,min_x, max_x,bdx)

    if ready:
        grid_uniform=np.ones(np.shape(grid))
        nb_bin=np.shape(grid)    
        nbc=reduce(lambda x,y:x*y,nb_bin,1)
        nbsamples=np.sum(grid)
        grid_uniform=nbsamples/nbc*grid_uniform
        if (nbsamples<nbc):
            print("Warning, too few samples to estimate coverage: nbsamples=%d, nbcells=%d"%(nbsamples,nbc))
        return coverage(grid),jensen_shannon_distance(grid,grid_uniform)
    else:
        return None,None

def get_indiv_coverage(x, min_x=None, max_x=None,nb_bin=None):
    """Compute the coverage of individuals, on the basis of samples drawn for each individual

    Compute the coverage of individuals, on the basis of samples drawn for each individual:
    :param x: the population to look at
    :param min_x: the minimum value of x
    :param max_x: the maximum value of x
    :param nb_bin: the number of bins
    :returns: the coverage of the individual and its specialization

    """
    icov=[]
    specialization=[]

    if (not hasattr(x[0],'evolvability_grid')):
        return None, None
    
    # computing the grid of offpsring and the corresponding coverage
    for ind in x:
        grid=build_grid(min_x,max_x,nb_bin)
        icov.append(get_updated_coverage(grid,[], ind.evolvability_samples, min_x=min_x, max_x=max_x, gen_window=0))
        ind.evolvability_grid=grid
    # Computing the specialization
    for ind1 in x:
        grid=ind1.evolvability_grid
        for ind2 in x:
            if (ind1 == ind2):
                continue
            grid=grid*ind2.evolvability_grid
        specialization.append(np.count_nonzero(ind1.evolvability_grid)-np.count_nonzero(grid))

    for ind in x:
        ind.evolvability_grid=None
    return icov,specialization



def get_stat_coverage(grid, indiv=False, min_x=None, max_x=None,nb_bin=None, gen_window_global=10):
    """Create a stat on the coverage

    Create a stat on the coverage:
    :param grid: grid to complete for the coverage of the whole population
    :param indiv: does the stat include individual coverage ?
    :param min_x: the minimum value of x (behavior descriptors)
    :param max_x: the maximum value of x
    :param nb_bin: the number of bins
    :param gen_window_global: the size of the window of generations to take into account for the estimation of the coverage, minimal suggested value: nbcells/pop_size, so that there is at least one point per cell (otherwise, the estimation won't be significant) 
    """
    stat_coverage = tools.Statistics(key=lambda ind: ind)
    lbd_global=[]
    stat_coverage.register("glob_cov",get_updated_coverage,grid, lbd_global, min_x=min_x, max_x=max_x, gen_window_global=10)
    if (indiv):
        stat_coverage.register("indiv_cov",get_indiv_coverage,min_x=min_x, max_x=max_x, nb_bin=nb_bin)
        
    return stat_coverage


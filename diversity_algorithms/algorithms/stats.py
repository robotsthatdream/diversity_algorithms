from deap import tools
import numpy
import sys
from diversity_algorithms.analysis.population_analysis import *

# useful classes
class Perc:
    def __init__(self,val):
        self.val=val
    def __call__(self,l):
        return numpy.percentile(l,self.val)


# fitness-based statistics
def get_stats_fitness(prefix=""):
    stats_fitness = tools.Statistics(key=lambda ind: ind.fitness.values)
    
    stats_fitness.register(prefix+"fit_median", numpy.median)
    stats_fitness.register(prefix+"fit_std", numpy.std)
    stats_fitness.register(prefix+"fit_min", numpy.min)
    stats_fitness.register(prefix+"fit_max", numpy.max)
    stats_fitness.register(prefix+"fit_perc25", Perc(25))
    stats_fitness.register(prefix+"fit_perc75", Perc(75))
    return stats_fitness

# novelty-based statistics
def get_stats_novelty(prefix=""):
    stats_novelty = tools.Statistics(key=lambda ind: ind.novelty)

    stats_novelty.register(prefix+"nov_median", numpy.median)
    stats_novelty.register(prefix+"nov_std", numpy.std)
    stats_novelty.register(prefix+"nov_min", numpy.min)
    stats_novelty.register(prefix+"nov_max", numpy.max)
    stats_novelty.register(prefix+"nov_perc25", Perc(25))
    stats_novelty.register(prefix+"nov_perc75", Perc(75))
    return stats_novelty

# Multi stats (compute several statistical values during the same compile call)

# Fitness + novelty
def get_stats_fit_nov(prefix=""):
    mstats_fit_nov = tools.MultiStatistics(fitness=get_stats_fitness(prefix), novelty=get_stats_novelty(prefix))
    mstats_fit_nov.register(prefix+"median", numpy.median)
    mstats_fit_nov.register(prefix+"std", numpy.std)
    mstats_fit_nov.register(prefix+"min", numpy.min)
    mstats_fit_nov.register(prefix+"max", numpy.max)
    mstats_fit_nov.register(prefix+"perc25", Perc(25))
    mstats_fit_nov.register(prefix+"perc75", Perc(75))
    return mstats_fit_nov
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
            grid_to_remove=np.zeros(np.shape(grid),dtype=np.int)
            update_grid(grid_to_remove,min_x, max_x,lbd[0])
            grid-=grid_to_remove
            lbd.pop(0)
    update_grid(grid,min_x, max_x,bdx)
    
    if ready:
        return coverage(grid),jensen_shannon_distance(grid,generate_uniform_grid(grid))
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

    if (not hasattr(x[0],'evolvability_samples')) or (x[0].evolvability_samples==None):
        return None, None
    
    # computing the grid of offpsring and the corresponding coverage
    for ind in x:
        grid=build_grid(min_x,max_x,nb_bin)
        lbd=[]
        icov.append(get_updated_coverage(grid,lbd, ind.evolvability_samples, min_x=min_x, max_x=max_x, gen_window=0))
        ind.evolvability_grid=grid

    # Computing the specialization
    for ind1 in x:
        spec=[]
        for ind2 in x:
            if (ind1 == ind2):
                continue
            grid=np.array(ind1.evolvability_grid)
            grid=grid*ind2.evolvability_grid
            spec.append(np.count_nonzero(ind1.evolvability_grid)-np.count_nonzero(grid))
        specialization.append([min(spec),float(sum(spec))/float(len(spec)),max(spec)])

    return icov,specialization


def get_stat_coverage(grid, prefix="", indiv=False, min_x=None, max_x=None,nb_bin=None, gen_window_global=10):
    """Create a stat on the coverage

    Create a stat on the coverage:
    :param grid: grid to complete for the coverage of the whole population
    :param indiv: does the stat include individual coverage ?
    :param min_x: the minimum value of x (behavior descriptors)
    :param max_x: the maximum value of x
    :param nb_bin: the number of bins
    :param gen_window_global: the size of the window of generations to take into account for the estimation of the coverage, minimal suggested value: nbcells/pop_size, so that there is at least one point per cell (otherwise, the estimation won't be significant) 
    """
    stat_coverage = tools.Statistics()
    lbd_global=[]
    stat_coverage.register(prefix+"glob_cov",get_updated_coverage,grid, lbd_global, min_x=min_x, max_x=max_x, gen_window=gen_window_global)
    if (indiv):
        stat_coverage.register(prefix+"indiv_cov",get_indiv_coverage,min_x=min_x, max_x=max_x, nb_bin=nb_bin)
        
    return stat_coverage


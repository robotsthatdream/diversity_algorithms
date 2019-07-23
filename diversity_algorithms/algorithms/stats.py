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
def get_updated_coverage(grid,x,min_x=None, max_x=None):
    bdx=[ind.fitness.bd for ind in x]
    update_grid(grid,min_x, max_x,bdx)
    return coverage(grid)

def get_indiv_coverage(x, min_x=None, max_x=None,nb_bin=None):
    # WARNING: this function needs to be called with a non empty set of evolvability samples (i.e. evolvability_nb_samples needs to be >0)
    icov=[]
    for ind in x:
        grid=build_grid(min_x,max_x,nb_bin)
        bdx=[sind.fitness.bd for sind in ind.evolvability_samples]
        icov.append(get_updated_coverage(grid,min_x, max_x, bdx))
    # TODO: complete the stats, notably look at how each grid is different from the others
    return icov

def get_stat_coverage(grid, indiv=False, min_x=None, max_x=None,nb_bin=None):
    stat_coverage = tools.Statistics(key=lambda ind: ind)
    stat_coverage.register("glob_cov",get_updated_coverage,grid, min_x=min_x, max_x=max_x)
    if (indiv):
        stat_coverage.register("indiv_cov",get_indiv_coverage,grid,min_x=min_x, max_x=max_x, nb_bin=nb_bin)
        
    return stat_coverage


# coding: utf-8
import numpy
import random
import math

import cma
from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.algorithms.novelty_management import *

with_scoop=True

if with_scoop:
	from scoop import futures


def generate_evolvability_samples_cmaes(es, evaluate, params, gen, force=False):
    if (force==True) or ((params["evolvability_nb_samples"]>0) and (params["dump_period_evolvability"]>0) and ((gen>0) and (gen % params["dump_period_evolvability"]==0))):
        print("\nWARNING: evolvability_nb_samples>0. We generate %d individuals for each indiv in the population for statistical purposes"%(params["evolvability_nb_samples"]))
        print("sampling for evolvability... ",end='', flush=True)
        evolvability_samples=es.ask(number=params["evolvability_nb_samples"])
        fit_bd = futures.map(evaluate,evolvability_samples) #[eval_with_functor(g) for g in solutions]
        dump_bd_evol=open(params["run_name"]+"/bd_evol_model_gen%04d.log"%(gen),"w")
        for fbd in fit_bd:
                dump_bd_evol.write(" ".join(map(str,fbd[1]))+"\n")
        dump_bd_evol.close()
        print("done")

   

def cmaes(evaluate, params, pool):
        """Launch a cmaes search run on a gym environment

        Launch a cmaes search run on a gym environment.
        :param variant: the variant to launch, can be "NS" or "DM" (fitness = - distance to current model)
        """

        center=[0]*params["ind_size"]
        sigma=5./3. #stdev, min-max of -5;5, which suggests a value of 5/3 for the stdev (see http://cma.gforge.inria.fr/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html) 
        
        if (params["variant"] == "CMAES_NS_mu1"):
                opts = cma.CMAOptions()
                opts.set('CMA_mu', 1)		

        es = cma.CMAEvolutionStrategy(center, sigma)
        i=0
        j=0
        archive=None
        gen=0
        while (not es.stop()) and (i<params["eval_budget"]):
                print(".", end="", flush=True)
                j+=1
                gen+=1
                solutions = es.ask()
                i+=len(solutions)
                fit_bd = futures.map(evaluate,solutions) #[eval_with_functor(g) for g in solutions]
                pop=[]
                nov=[]
                fit=[]
                for g,fbd in zip(solutions,fit_bd):
                        ind=Indiv(g,fbd[0],fbd[1])
                        pop.append(ind)
                        fit.append(fbd[0][0])

                if (params["variant"] == "DM"):
                        model=es.mean
                        fit_bd_model = list(futures.map(evaluate,[model])) #[eval_with_functor(g) for g in solutions]
                        model_bd=fit_bd_model[0][1]
                        dm_fit=[]
                        for ind in pop:
                                dm_fit.append(-np.linalg.norm(np.array(model_bd)-np.array(ind.bd)))
                        es.tell(solutions,dm_fit)
                        #print("Gen=%d, min dist_to_model=%f, max dist_to_model=%f, min fit=%f, max fit=%f (evals remaining=%d)"%(gen,min(dm_fit),max(dm_fit), min(fit), max(fit), params["eval_budget"]-i))
                        
                        
                if (params["variant"] in ["CMAES_NS", "CMAES_NS_mu1"]):
                        
                        if ((archive is not None) and (archive.ready())):
                                update_model=True
                        else:
                                update_model=False

                        archive=updateNovelty(pop,pop,archive,params)

                        for ind in pop:
                                nov.append(ind.novelty)

                        if(update_model):
                                es.tell(solutions, [-ind.novelty for ind in pop])
                        else:
                                print("No model update, the archive still needs to grow to estimate novelty...")

                        #print("Gen=%d, min novelty=%f, max novelty=%f, min fit=%f, max fit=%f (evals remaining=%d)"%(gen,min(nov),max(nov), min(fit), max(fit), params["eval_budget"]-i))

                dump_data(pop, gen, params, prefix="population", attrs=["all"])
                dump_data(pop, gen, params, prefix="bd", complementary_name="population", attrs=["bd"])
                dump_data(archive.get_content_as_list(), gen, params, prefix="archive", attrs=["all"])

                generate_evolvability_samples_cmaes(es, evaluate, params, gen)

                es.disp()

        if (params["dump_period_evolvability"]>0):
                generate_evolvability_samples_cmaes(es, evaluate, params, gen, force=True)

        params["nb_gen"]=gen # for the terminating_run function to know how many gens were run
        #es.result_pretty()
        return es.result, archive, i



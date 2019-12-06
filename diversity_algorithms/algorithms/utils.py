import numpy as np
import datetime
import os
import subprocess
import dill
import pickle

from diversity_algorithms.analysis.population_analysis import *

class Fitness:
        def __init__(self,fit):
            if (fit is None):
                self.values=None
                self.valid=False
            else:
                self.values=fit
                self.valid=True
        
class Indiv:
       def __init__(self, g, fit, bd):
           self.g=list(g)
           self.fitness=Fitness(fit)
           self.bd=bd
       def __len__(self):
           return len(self.g)
       def __getitem__(self,i):
           return self.g[i]
       def __setitem__(self,i,v):
           self.g[i]=v

def generate_exp_name(name=""):
    d=datetime.datetime.today()
    if(name!=""):
        sep="_"
    else:
        sep=""
    run_name=d.strftime(name+sep+"%Y_%m_%d-%H:%M:%S")
    nb=0
    not_created=True
    while(not_created):
        try:
            os.mkdir(run_name+"_%d"%nb)
            run_name+="_%d"%nb
            not_created=False
        except OSError:
            nb+=1
    return run_name

def dump_exp_details(argv,run_name, params):
    print("Dumping exp details: "+" ".join(argv))
    gdir=os.path.dirname(argv[0])
    if (gdir==""):
        gdir="."
    r=subprocess.run(["git", "rev-parse", "--short", "HEAD"], stdout=subprocess.PIPE, cwd=gdir)
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    f=open(run_name+"/info.log","w")
    f.write("## Features of the experiment ##\n")
    f.write("Git hash: "+r.stdout.decode("utf-8"))
    f.write("Command: "+" ".join(argv)+"\n")
    f.write("Params: \n")
    for k in params.keys():
	    if (params[k].is_default()):
		    defstr="(default)"
	    else:
		    defstr=""
	    f.write("\t"+k+" "+str(params[k].get_value())+" "+defstr+"\n")
    d=datetime.datetime.today()
    sd=d.strftime("<%Y_%m_%d-%H:%M:%S>")

    f.write("\n++ Started at: "+sd+"\n")
    f.close()

def dump_end_of_exp(run_name):
    f=open(run_name+"/info.log","a")
    d=datetime.datetime.today()
    sd=d.strftime("<%Y_%m_%d-%H:%M:%S>")

    f.write("-- Ended at: "+sd+"\n")
    f.close()

    
def dump_pop(pop, gen, run_name="runXXX", prefix="pop"):
    out_dict = {"gen": gen, "size": len(pop)}
    for (i,ind) in enumerate(pop):
        if(hasattr(ind,'strategy')):
            out_dict["centroid_%d" % i] = np.array(ind.strategy.centroid)
            out_dict["C_%d" % i] = np.array(ind.strategy.C)
            out_dict["sigma_%d" % i] = np.array(ind.strategy.sigma)
            out_dict["w_%d" % i] = np.array(ind.strategy.w)
            out_dict["ccov_%d" % i] = np.array(ind.strategy.ccov)
        else:
            out_dict["geno_%d" % i] = np.array(ind)
        if(ind.fitness.valid):
            out_dict["fitness_%d" % i] = ind.fitness.values
            if(hasattr(ind,'novelty')):
                out_dict["novelty_%d" % i] = ind.novelty
            if(hasattr(ind,'bd')):
                out_dict["bd_%d" % i] = ind.bd
            if ((hasattr(ind,'evolvability_samples')) and (ind.evolvability_samples is not None)):
                for (j,indj) in enumerate(ind.evolvability_samples):
                    out_dict["es_%d_%d" %(i,j)] = indj.bd 

                
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    np.savez(run_name+"/"+prefix+"_gen%d.npz" % gen, **out_dict) 

def load_pop(dumpfile):
    pop_dict=np.load(dumpfile, allow_pickle=True)
    pop=[]
    for i in range(pop_dict["size"]):
        if ("fitness_%d"%(i) in pop_dict.keys()):
            fit=pop_dict["fitness_%d"%(i)]
        else:
            continue
            fit=None
        if ("bd_%d"%(i) in pop_dict.keys()):
            bd=pop_dict["bd_%d"%(i)]
        else:
            continue
            bd=None
        if ("centroid_%d"%(i) in pop_dict.keys()):
            sigma=pop_dict["sigma_%d"%(i)][0]
            w=pop_dict["w_%d"%(i)]
            ind=generate_CMANS(creator.individual, CMANS_Strategy_C_rank_one, pop_dict["centroid_%d"%(i)], pop_dict["min"], pop_dict["max"], sigma, w)
            ind.set_centroid(pop_dict["centroid_%d"%(i)])
            ind.set_C(pop_dict["C_%d"%(i)])
            ind.fitness=Fitness(fit)
            ind.bd=bd
        else:
            ind=Indiv(pop_dict["geno_%d"%(i)], fit,bd)
        if ("novelty_%d"%(i) in pop_dict.keys()):
            ind.novelty=pop_dict["novelty_%d"%(i)]
        pop.append(ind)
    return pop

def load_pop_toolbox(dumpfile, toolbox):
    pop_dict=np.load(dumpfile, allow_pickle=True)
    pop=[]
    for i in range(pop_dict["size"]):
        if ("fitness_%d"%(i) in pop_dict.keys()):
            fit=pop_dict["fitness_%d"%(i)]
        else:
            continue
            fit=None
        if ("bd_%d"%(i) in pop_dict.keys()):
            bd=pop_dict["bd_%d"%(i)]
        else:
            continue
            bd=None


        if ("centroid_%d"%(i) in pop_dict.keys()):
            ind=toolbox.individual()
            #generate_CMANS(creator.individual, CMANS_Strategy_C_rank_one, pop_dict["centroid_%d"%(i)], pop_dict["min"], pop_dict["max"], sigma, w)
            ind.set_centroid(pop_dict["centroid_%d"%(i)])
            ind.strategy.C = np.array(pop_dict["C_%d"%(i)])
            ind.strategy.sigma = pop_dict["sigma_%d"%(i)]
            ind.strategy.w = pop_dict["w_%d"%(i)]            
        else:
            ind=Indiv(pop_dict["geno_%d"%(i)], fit,bd)
            #geno=pop_dict["geno_%d"%(i)]

            #ind=toolbox.individual()
            #for i in range(len(geno)):
            #    ind.g[i]=geno[i]

        ind.fitness=Fitness(fit)
        ind.bd=bd

        if ("novelty_%d"%(i) in pop_dict.keys()):
            ind.novelty=pop_dict["novelty_%d"%(i)]
        pop.append(ind)
    return pop
		    
def verbosity(params, value=["all"]):
	return params["verbosity"] in value
			    

def dump_archive(archive, gen, run_name="runXXX"):
    out_dict = {"gen": gen, "size": archive.size()}
    for (i,ind) in enumerate(archive.all_bd):
        out_dict["bd_%d" % i] = np.array(ind)
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    np.savez(run_name+"/archive_gen%d.npz" % gen, **out_dict) 

# TODO: Should unify
def dump_archive_qd(archive, gen, run_name="runXXX"):
    out_dict = {"gen": gen, "size": archive.size()}
    content = archive.get_content_as_list()
    for (i,ind) in enumerate(content):
        out_dict["bd_%d" % i] = np.array(ind.bd)
        out_dict["nov_%d" % i] = ind.novelty
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    np.savez(run_name+"/archive_gen%d.npz" % gen, **out_dict) 



def dump_params(params, run_name="runXXX"):
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    #stat=params["STATS"]
    #params["STATS"]=None # pickle can't save some parts of the stat
    np.savez(run_name+"/params.npz", **params) 
    #params["STATS"]=stat

def dump_logbook(logbook,gen,run_name="runXXX"):
    if (logbook is None):
        return
    out_dict = {}
    for k in logbook.header:
        out_dict[k]=logbook.select(k)
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    np.savez(run_name+"/logbook_gen%d.npz" % gen, **out_dict) 

def generate_evolvability_samples(params, population, gen, toolbox):
    """Generates a sample of individuals from the given population. 

    Generates a sample of individuals from the given population. It either relies on the toolbox (with the crossover and mutation probabilities) or on the strategy (if the individuals have one) to generate the points. 
    """

    if (params["evolvability_nb_samples"]>0) and (params["evolvability_period"]>0) and (gen>0) and (gen % params["evolvability_period"]==0):
        print("\nWARNING: evolvability_nb_samples>0. We generate %d individuals for each indiv in the population for statistical purposes"%(params["evolvability_nb_samples"]))
        print("sampling for evolvability: ",end='', flush=True)
        ig=0
        for ind in population:
            print(".", end='', flush=True)
            if (hasattr(ind, 'strategy')):
                ind.evolvability_samples=ind.strategy.generate_samples(params["evolvability_nb_samples"])
                fitnesses = toolbox.map(toolbox.evaluate, ind.evolvability_samples)
                for indes, fit in zip(ind.evolvability_samples, fitnesses):
                    indes.fitness.values = fit[0] 
                    indes.bd = fit[1]
                    indes.evolvability_samples=None # SD: required, otherwise, the memory usage explodes... I do not understand why yet.
                
            else:
                    ind.evolvability_samples=sample_from_pop([ind],toolbox,params["evolvability_nb_samples"],params["cxpb"],params["mutpb"])
            dump_bd_evol=open(run_name+"/bd_evol_indiv%04d_gen%04d.log"%(ig,gen),"w")
            for inde in ind.evolvability_samples:
                dump_bd_evol.write(" ".join(map(str,inde.bd))+"\n")
            dump_bd_evol.close()
            ig+=1
        print("")


def generate_dumps(run_name, dump_period_bd, dump_period_pop, pop1, pop2, gen, pop1label="population", pop2label="offspring", archive=None, logbook=None, pop_to_dump=[True, True]):
    #print("Dumping data. Gen="+str(gen)+" dump_period_bd="+str(dump_period_bd)+" dump_period_pop="+str(dump_period_pop))
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
        if(pop1 is not None and pop_to_dump[0]):
            dump_pop(pop1, gen, run_name, pop1label)
        if(pop2 is not None and pop_to_dump[1]):
            dump_pop(pop2, gen,run_name, pop2label)
        if (archive is not None):
            dump_archive(archive, gen,run_name)
        if (logbook is not None):
            dump_logbook(logbook, gen,run_name)

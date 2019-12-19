import numpy as np
import datetime
import os
import subprocess
import dill
import pickle
import sys

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
            os.makedirs(run_name+"_%d"%nb)
            run_name+="_%d"%nb
            not_created=False
        except OSError:
            nb+=1
            if (nb>10):
                    print("Problem when trying to create the dir to host exp files. Dir="+run_name+"_0")
                    sys.exit(1)
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

def dump_end_of_exp(run_name, nb_eval):
    f=open(run_name+"/info.log","a")
    d=datetime.datetime.today()
    sd=d.strftime("<%Y_%m_%d-%H:%M:%S>")

    f.write("== Nb eval: %d\n"%(nb_eval))
    f.write("-- Ended at: "+sd+"\n")
    f.close()

def verbosity(params, value=["all"]):
	return params["verbosity"] in value
			    

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
    
def dump_data(data_list, gen, params, prefix="population", complementary_name="", attrs=["all"], force=False):
	# should fit to dump any data required. Examples:
	# dump_data(pop, gen, params, prefix="pop", complementary_name="", attrs=["all"]) to dump the whole pop
	# dump_data(pop[i].evolvability_samples, gen, params, prefix="bd_es", complementary_name="%d"%(i), attrs=["bd"]) to dump the bd of evolvability samples
	# dump_data(archive.all_bd, gen, params, prefix="archive", complementary_name="", attrs=["ind"]) to dump the archive (novelty search)
	# dump_data(archive.get_content_as_list(), gen, params, prefix="archive_qd", complementary_name="", attrs=["bd", "novelty"]) to dump the archive (qd)
	

	if (not force) and ("dump_period_"+prefix not in params):
		print("ERROR: tryind to dump data without saying at what period to do it. You need to define a dump_period_"+prefix+" parameter.")
		return

	if (force) or ((params["dump_period_"+prefix] >0) and (gen % params["dump_period_"+prefix] == 0)): 
		out_dict = {"gen": gen, "size": len(data_list)}
		for (i,ind) in enumerate(data_list):
			#print("Ind attributes: "+str(ind.__dict__.keys()))
			try:
				ind.dump_to_dict(out_dict,i, attrs)
			except AttributeError:
				if ("all" in attrs):
					myattrs=attrs+["ind", "fit", "novelty", "bd"]
				else:
					myattrs=attrs
				if ("ind" in myattrs):
					out_dict["ind_%d" % i] = np.array(ind)
				if (hasattr(ind, "__dict__")):
					for k in ind.__dict__.keys():
						if (k in myattrs):
							try:
								out_dict[k+"_%d" % (i)] = np.array(getattr(ind,k))
							except AttributeError:
								pass

		    #if (("evolvability_samples" in attrs) or ("all" in attrs)) and ((hasattr(ind,'evolvability_samples')) and (ind.evolvability_samples is not None)):
		    #	    for (j,indj) in enumerate(ind.evolvability_samples):
		    #		    out_dict["es_%d_%d" %(i,j)] = indj.bd 
		try:
			os.mkdir(params["run_name"])
		except OSError:
			pass
		if (complementary_name!= ""):
			complementary_name+="_"
		np.savez(params["run_name"]+"/"+prefix+"_"+complementary_name+"_".join(attrs)+"_gen%d.npz" % gen, **out_dict) 

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
		    
def generate_evolvability_samples(params, population, gen, toolbox):
    """Generates a sample of individuals from the given population. 

    Generates a sample of individuals from the given population. It either relies on the toolbox (with the crossover and mutation probabilities) or on the strategy (if the individuals have one) to generate the points. 
    """

    if (params["evolvability_nb_samples"]>0) and (params["dump_period_evolvability"]>0) and (gen>0) and (gen % params["dump_period_evolvability"]==0):
        print("\nWARNING: evolvability_nb_samples>0. We generate %d individuals for each indiv in the population for statistical purposes"%(params["evolvability_nb_samples"]))
        print("sampling for evolvability: ",end='', flush=True)
        ig=0
        for (i,ind) in enumerate(population):
            print(".", end='', flush=True)
            try:
                ind.evolvability_samples=ind.strategy.generate_samples(params["evolvability_nb_samples"])
                fitnesses = toolbox.map(toolbox.evaluate, ind.evolvability_samples)
                for indes, fit in zip(ind.evolvability_samples, fitnesses):
                    indes.fitness.values = fit[0] 
                    indes.bd = fit[1]
                    indes.evolvability_samples=None # SD: required, otherwise, the memory usage explodes... I do not understand why yet.
            except AttributeError:
                    ind.evolvability_samples=sample_from_pop([ind],toolbox,params["evolvability_nb_samples"],params["cxpb"],params["mutpb"])

            dump_data(ind.evolvability_samples,gen, params, prefix="evolvability", complementary_name="ind%d"%(i), attrs=["bd"])
            ig+=1
        print("")


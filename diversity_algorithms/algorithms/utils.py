import numpy as np
import datetime
import os
import subprocess
import dill
import pickle

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

def dump_exp_details(argv,run_name):
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

    d=datetime.datetime.today()
    sd=d.strftime("<%Y_%m_%d-%H:%M:%S>")

    f.write("++ Started at: "+sd+"\n")
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
    out_dict = {"gen": gen, "size": archive.get_size()}
    content = archive.get_content_as_list()
    for (i,ind) in enumerate(content):
        out_dict["bd_%d" % i] = np.array(ind.bd)
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
    out_dict = {}
    for k in logbook.header:
        out_dict[k]=logbook.select(k)
    try:
        os.mkdir(run_name)
    except OSError:
        pass
    np.savez(run_name+"/logbook_gen%d.npz" % gen, **out_dict) 

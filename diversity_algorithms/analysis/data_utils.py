#!/usr/bin python -w

import matplotlib.pyplot as plt
import os,sys
import re

def listify(x):
    if(type(x) is list or x is None): # If it's already a list, return it
        return x
    elif(type(x) is str):
    	return [x]
    elif(hasattr(x, '__iter__')): # If it's some other iterable, convert to list
        # SD: warning, a string is iterable and is thus decomposed into a set of characters...
        return list(x)
    else: # If it's a random object, wrap in a list
        return [x]




re_bd_evolvability_file = re.compile("bd_evol_indiv...._gen(....).log")
re_bdfile = re.compile("bd_(....)_offspring.log")
re_bdfile_model = re.compile("bd_evol_model_gen(....).log")
re_bdfile_old = re.compile("bd_(....).log")
re_bdpopfile = re.compile("bd_pop_(....).log")
re_genfile = re.compile("pop_gen(.+).npz")


# From BD files
def get_files_per_gen(regex, data_dirs=".", gens=None): #Default : local dir, all gens
    files = dict()
    gens_ok = listify(gens)
    dirs = listify(data_dirs)
    for data_dir in dirs:
        if (not os.path.exists(data_dir)):
        	print("The data dir does not exist: "+data_dir)
        	return None
        for f in os.listdir(data_dir):
            good = regex.match(f)
            if(good):
                gen = int(good.groups()[0])
                if((gens_ok is None) or gen in gens_ok):
                    if(gen not in files):
                        files[gen] = list()
                    files[gen].append(data_dir+"/"+f)
    return files

def get_bd_evolvability_files_per_gen(data_dirs=".", gens=None):
    return get_files_per_gen(re_bd_evolvability_file, data_dirs, gens)

def get_bdfiles_per_gen(data_dirs=".", gens=None):
    return get_files_per_gen(re_bdfile, data_dirs, gens)

def get_bdfiles_model_per_gen(data_dirs=".", gens=None):
    return get_files_per_gen(re_bdfile_model, data_dirs, gens)

def get_bdfiles_old_per_gen(data_dirs=".", gens=None):
    return get_files_per_gen(re_bdfile_old, data_dirs, gens)

def get_bdpopfiles_per_gen(data_dirs=".", gens=None):
    return get_files_per_gen(re_bdpopfile, data_dirs, gens)

def get_genfiles_per_gen(data_dirs=".", gens=None):
    return get_files_per_gen(re_genfile, data_dirs, gens)

def get_points_from_bdfile(bdfile):
    points = list()
    with open(bdfile,'r') as fd:
        for line in fd:
            [x,y] = line.strip().split(' ')
            points.append((float(x),float(y)))
    return points

def get_points_from_genfile(genfile):
    points = list()
    archive = np.load(genfile)
    size = int(archive['size'])
    for idx in range(size):
        points.append(tuple(archive["bd_%d" % idx]))
    return points

def get_points_per_gen_from_files(files, extractfunc):
    out = dict()
    for gen in files:
        out[gen] = list()
        for f in files[gen]:
            points = extractfunc(f)
            out[gen] += points
    return out

def get_points_per_gen_from_bdfiles(bdfiles):
    return get_points_per_gen_from_files(bdfiles,extractfunc=get_points_from_bdfile)

def get_points_per_gen_from_genfiles(bdfiles):
    return get_points_per_gen_from_files(bdfiles,extractfunc=get_points_from_genfile)


def merge_gens(gendict,max_gen=-1, min_gen=-1):
    out = list()
    gen=list(gendict.keys())
    for g in gen:
        if ((max_gen<0) or (g<max_gen)) and ((min_gen<0) or (g>=min_gen)): 
            out += gendict[g]
    return out


def get_exp_files(resdir, variant, cond_files):
    """Gets only the experiments that respect some conditions.

    Gets only the experiments that respect some conditions.
    :param resdir: the dir to explore
    :param variant: the name of the variant to take into account
    :param cond_files: a list of files that must exist.
    """
    eres=[]
    with os.scandir(resdir) as it:
        for f in it:
            if variant in f.name:
                # this exp can be considered
                to_keep=True
                edir=resdir+"/"+f.name
                for p in cond_files:
                    if not os.path.isfile(edir+"/"+p):
                        print(p+" not found in :\n\t"+edir)
                        to_keep=False
                        break
                if to_keep:
                    print("Result to keep: "+edir)
                    eres.append(edir)
    return eres
                    

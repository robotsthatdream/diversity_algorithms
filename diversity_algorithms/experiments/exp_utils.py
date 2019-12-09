import sys
import getopt
from diversity_algorithms.algorithms.utils import *
from diversity_algorithms.environments import EvaluationFunctor
from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis import build_grid
from diversity_algorithms.algorithms.stats import * 

from diversity_algorithms.algorithms import grid_features

#__all__={"RunParam", "analyse_params", "get_simple_params_dict"}

class RunParam(object):
    def __init__(self, short_name, default_value, doc):
        if (len(short_name)>1):
            print("Short parameter names should have only one character ! Value given: "+str(short_name))
            sys.exit(1)
        # some params may have an empty short_name: ""
        self.short_name=short_name
        self.default_value=default_value
        self.doc=doc
        self.value=None

    def set_value(self, value):
        try:
            self.value=type(self.default_value)(value)
        except ValueError:
            print("ERROR: the provided value for the argument is inappropriate. Value="+str(value)+", expected type="+str(type(self.default_value)))

    def get_value(self):
        if (self.value is not None):
            return self.value
        else:
            return self.default_value

    def is_default(self):
        return self.value==None

def check_params(params):
	short_params=["h"]
	OK=True
	for k in params.keys():
		if (params[k].short_name!="") and (params[k].short_name in short_params):
			print("ERROR: params are invalid, "+params[k].short_name+" appears several times !")
			OK=False
		short_params.append(params[k].short_name)
	if (not OK):
		sys.exit(1)

def get_param_from_short_name(params, short_name):
    for k in params.keys():
        if (short_name=="-"+params[k].short_name):
            return k
    return None

def get_simple_params_dict(params):
    dparams={}
    for k in params.keys():
        dparams[k]=params[k].get_value()
    return dparams

def analyze_params(params, argv):

    check_params(params)

    optstr="hv"+"".join([params[k].short_name+":" for k in params.keys()])
    optargs=[k+"=" for k in params.keys()]
    helpstr="-h -v ".join(["-"+params[k].short_name+" <"+k+">" for k in params.keys()])

    try:
        opts, args = getopt.getopt(argv[1:],optstr, optargs)
    except getopt.GetoptError:
        print(argv[0]+helpstr)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(sys.argv[0]+helpstr)
            for k in params.keys():
                print("\t-"+params[k].short_name+" --"+k+": (default value: ("+params[k]+") "+params[k].doc)
            sys.exit(1)
        k=get_param_from_short_name(params, opt)
        if (k==None):
            if (opt[2:] in params.keys()):
                k=opt[2:]
            else:
                print("ERROR: unknown param: "+opt)
                sys.exit(1)
        if (k not in params.keys()):
            print("ERROR: non valid params: "+str(k))
            sys.exit(1)
        params[k].set_value(arg)

def preparing_run(eval_gym, params, with_scoop, deap=True):

    if with_scoop:
        from scoop import futures

    # Dumping how the run has been launched
    run_name=generate_exp_name(params["env_name"].get_value()+"_"+params["variant"].get_value())
    print("Saving logs in "+run_name)
    dump_exp_details(sys.argv,run_name, params)

    # Completing the parameters (and putting them in a simple dict for future use)
    sparams=get_simple_params_dict(params)

    if (sparams["env_name"] in grid_features.keys()):
        min_bd=grid_features[sparams["env_name"]]["min_x"]
        max_bd=grid_features[sparams["env_name"]]["max_x"]
        nb_bin_bd=grid_features[sparams["env_name"]]["nb_bin"]
        
        grid=build_grid(min_bd, max_bd, nb_bin_bd)
        grid_offspring=build_grid(min_bd, max_bd, nb_bin_bd)
        stats=None
        stats_offspring=None
        nbc=nb_bin_bd**2
        nbs=nbc*2 # min 2 samples per bin
        evolvability_nb_samples=nbs
    else:
        grid=None
        grid_offspring=None
        min_bd=None
        max_bd=None
        nb_bin_bd=None
        evolvability_nb_samples=0
        nbs=0
        
    sparams["ind_size"]=eval_gym.controller.n_weights
	
    sparams["evolvability_nb_samples"]=evolvability_nb_samples
    sparams["min_bd"]=min_bd # not used by NS. It is just to keep track of it in the saved param file
    sparams["max_bd"]=max_bd # not used by NS. It is just to keep track of it in the saved param file
    
    if deap:
        # We use a different window size to compute statistics in order to have the same number of points for population and offspring statistics
        window_population=nbs/sparams["pop_size"]
        if ("lambda" in sparams):
            window_offspring=nbs/(sparams["lambda"])
        elif("cma_lambda" in sparams):
            window_offspring=nbs/(sparams["cma_lambda"]*sparams["pop_size"])
        elif("seed_lambda" in sparams):
            window_offspring=nbs/(sparams["seed_lambda"]*sparams["pop_size"])
        else:
            print("ERROR: we don't know how to set window_offspring")
            window_offspring=None

        if (sparams["dump_period_evolvability"]>0) and (evolvability_nb_samples>0):
            stats=get_stat_fit_nov_cov(grid,prefix="population_",indiv=True,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_population)
            if (window_offspring is not None):
                stats_offspring=get_stat_fit_nov_cov(grid_offspring,prefix="offspring_",indiv=True,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_offspring)
        else:
            stats=get_stat_fit_nov_cov(grid,prefix="population_",indiv=False,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_population)
            if (window_offspring is not None):
                stats_offspring=get_stat_fit_nov_cov(grid_offspring,prefix="offspring_", indiv=False,min_x=min_bd,max_x=max_bd,nb_bin=nb_bin_bd, gen_window_global=window_offspring)
            
        sparams["stats"] = stats # Statistics
        sparams["window_population"]=window_population
        if (window_offspring is not None):
            sparams["stats_offspring"] = stats_offspring # Statistics on offspring
            sparams["window_offspring"]=window_offspring
        else:
            sparams["stats_offspring"] = None
            sparams["window_offspring"]= None

    sparams["run_name"]=run_name
    
    print("Launching a run with the following parameter values:")
    for k in sparams.keys():
        print("\t"+k+": "+str(sparams[k]))
    if (grid is None):
        print("WARNING: grid features have not been defined for env "+sparams["env_name"]+". This will have no impact on the run, except that the coverage statistic has been turned off")
    if (sparams["dump_period_evolvability"]>0) and (evolvability_nb_samples>0):
        print("WARNING, evolvability_nb_samples>0. The run will last much longer...")

    if with_scoop:
        pool=futures
    else:
        pool=None
        
    dump_params(sparams,run_name)

    return sparams, pool

def terminating_run(sparams, pop, archive, logbook):

    if (pop is not None):
        dump_data(pop,sparams["nb_gen"],sparams, prefix="final_pop", attrs=["all"], force=True)
    if (logbook is not None):
        dump_logbook(logbook,sparams["nb_gen"],sparams["run_name"])
    if (archive is not None):
        dump_data(archive.get_content_as_list(),sparams["nb_gen"],sparams, prefix="final_archive", attrs=["bd"], force=True)
    
    dump_end_of_exp(sparams["run_name"])
    
    print("The population, log, archives, etc have been dumped in: "+sparams["run_name"])

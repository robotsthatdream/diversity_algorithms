#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    10/03/2020
""" 

import numpy as np

from diversity_algorithms.controllers import SimpleNeuralController
from diversity_algorithms.analysis.data_utils import listify


def fitness_last_bd_other(genotype):
    """ Toy mapping where the last genotype dim is the fitness and the rest is the BD
    """
    fitness = genotype[-1]
    bd = genotype[:-1]
    return fitness, bd


mappings_dict = {"fitness_last":fitness_last_bd_other}




class DummyController:
    def __init__(self,n_params):
        # For some compatibility reasons, the number of parameters must be
        # stored in self.n_weights (see exp_utils.py:152)
        self.n_weights=n_params

    def set_parameters(self, theta):
        assert (len(theta) == self.n_weights), "Bad number of params in dummy controller"
        self.theta = theta

    # Just return the parameters !
    def __call__(self,_):
        return self.theta


class SimpleMappingEvaluator:
    def __init__(self, geno_size, mapping="fitness_last", controller_type=None, controller_params=None):
        """ geno_size: number of parameters in the genotype
            mapping: defines a mapping between genotype and (fitness, bd), must be in
                     the mappings_dict above
            controller_type, controller_params : present for API compatibility reasons, will
                                                 be ignored (DummyController is always used)
        """
        self.controller = DummyController(geno_size)
        if(mapping not in mappings_dict):
            raise(RuntimeError("Unknown mapping '%s'" % mapping))
        self.mapping = mappings_dict[mapping]
    
    
    def __call__(self,genotype):
        self.controller.set_parameters(genotype)
        fitness, bd = self.mapping(genotype)
        return [[fitness], bd]
    

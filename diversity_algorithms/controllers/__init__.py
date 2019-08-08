# coding: utf-8

#from diversity_algorithms.controllers.fixed_structure_nn import SimpleNeuralControllerKeras as SimpleNeuralController
from diversity_algorithms.controllers.fixed_structure_nn_numpy import SimpleNeuralControllerNumpy as SimpleNeuralController
from diversity_algorithms.controllers.graph_tool_dnn import DNN, DNNController
from diversity_algorithms.controllers.deap_tools_dnn import initDNN, mutDNN, mateDNNDummy

__all__ = ["fixed_structure_nn", "fixed_structure_nn_numpy", "graph_tool_dnn", "deap_tools_dnn"] 

#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    31/07/2019
""" 

import numpy as np

from diversity_algorithms.controllers.graph_tool_dnn import DNN

from deap.tools import mutPolynomialBounded


def initDNN(Indivclass, in_size, out_size):
	return Indivclass(in_size, out_size)


def mutDNN(dnn, indiv_mutation_rate_wb, mutation_eta):
	for e in dnn.nn.edges():
		w = dnn.nn.ep.weights[e]
		new_w = mutPolynomialBounded([w], mutation_eta, dnn.min_w, dnn.max_w, indiv_mutation_rate_wb)
		dnn.nn.ep.weights[e] = new_w
	non_input_nodes = dnn.out_nodes + dnn.hidden_nodes
	for v in non_input_nodes:
		b = dnn.nn.vp.bias[v]
		new_b = mutPolynomialBounded([b], mutation_eta, dnn.min_w, dnn.max_w, indiv_mutation_rate_wb)
		dnn.nn.ep.bias[v] = new_b
		


def mateDNNDummy(dnn1, dnn2, alpha=0.5):
	return (dnn1, dnn2)


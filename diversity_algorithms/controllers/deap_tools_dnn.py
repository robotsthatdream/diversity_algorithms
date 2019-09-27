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


def mutDNN(dnn, # Individual to mutte
		mutation_rate_params_wb=0.1, # Individual mutation rate for the weights and bias
		mutation_eta=15., # Eta parameter of the polynomial mutation for the weights and bias
		mutation_rate_add_conn=0.1, # Chance to add a connection
		mutation_rate_del_conn=0.01, # Chance to delete a connection
		mutation_rate_add_node=0.1, # Chance to add a hidden neuron
		mutation_rate_del_node=0.01, # Chance to delete a hidden neuron
		n_min_hidden=0, # Minimum number of hidden neurons
		n_max_hidden=30, # Maximum number of hidden neurons
		n_min_conns=8, # Minimum number of edges
		n_max_conns=250 # Maximum number of edges
		):
	#print("Before mutation:")
	#dnn.describe()
	# Mutate weights
	for e in dnn.nn.edges():
		w = dnn.nn.ep.weights[e]
		new_w = mutPolynomialBounded([w], mutation_eta, dnn.min_w, dnn.max_w, mutation_rate_params_wb)[0][0]
		dnn.nn.ep.weights[e] = new_w
	# Mutate bias
	non_input_nodes = dnn.out_nodes + dnn.hidden_nodes
	for v in non_input_nodes:
		b = dnn.nn.vp.bias[v]
		new_b = mutPolynomialBounded([b], mutation_eta, dnn.min_w, dnn.max_w, mutation_rate_params_wb)[0][0]
		dnn.nn.vp.bias[v] = new_b
	# Add connection
	if((dnn.n_conns() < n_max_conns) and (np.random.random() < mutation_rate_add_conn)):
		#print("Added conn!!")
		dnn.add_random_conn()
	# Del connection
	if((dnn.n_conns() > n_min_conns) and (np.random.random() < mutation_rate_del_conn)):
		#print("Deleted conn!!")
		dnn.del_random_conn()
	# Add node on a random edge
	if((dnn.n_hidden() < n_max_hidden) and (np.random.random() < mutation_rate_add_node)):
		#print("Added node!!")
		dnn.add_node_on_random_edge()
	
	# Del random hidden node
	if((dnn.n_hidden() > n_min_hidden) and (np.random.random() < mutation_rate_del_node)):
		#print("Deleted node!!")
		dnn.del_random_hidden()
	#print("After mutation:")
	#dnn.describe()
	return (dnn,)


def mateDNNDummy(dnn1, dnn2, alpha=0.5): # Dummy crossover
	return (dnn1, dnn2)


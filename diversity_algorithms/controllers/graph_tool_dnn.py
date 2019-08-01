#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    31/07/2019
""" 

import numpy as np

from graph_tool.all import *

def sigmoid(x):
	return 1./(1 + np.exp(-x))

def tanh(x):
	return np.tanh(x)

def linear(x):
	return x

def relu(x):
	return max(x,0.)



def init_nn():
	nn = Graph(directed=True) # Create a directed graph
	
	w = nn.new_edge_property("double") # Create weights
	nn.edge_properties["weights"] = w
	
	b = nn.new_vertex_property("double")  # Create bias
	nn.vertex_properties["bias"] = b
	
	act = nn.new_vertex_property("double")  # Create activations
	nn.vertex_properties["activations"] = act
	
	out = nn.new_vertex_property("double")  # Create outs
	nn.vertex_properties["outputs"] = act
	
	ok = nn.new_vertex_property("bool")  # Create out OK flag
	nn.vertex_properties["out_ok"] = ok
	
	# Save in/out node status in the graph for pickling
	is_in = nn.new_vertex_property("bool")
	nn.vertex_properties["is_in"] = is_in
	
	is_out = nn.new_vertex_property("bool")
	nn.vertex_properties["is_out"] = is_out
	return nn




class DNN:
	def __init__(self, n_in, n_out, activations=sigmoid, activations_out=tanh, min_w=-5, max_w=5, param_initializer_func=np.random.uniform):
		self.nn = init_nn()
		self.af = activations
		self.af_out = activations_out
		self.min_w = min_w
		self.max_w = max_w
		self.param_initializer_func = param_initializer_func

		# Add ins
		self.in_nodes = list()
		for i in range(n_in):
			v_in = self.nn.add_vertex()
			self.in_nodes.append(v_in)
			self.nn.vp.outputs[v_in] = 0.
			self.nn.vp.out_ok[v_in] = False
			self.nn.vp.is_in[v_in] = True
			self.nn.vp.is_out[v_in] = False
	
		# Add outs and create basic single layer perceptron structure
		self.out_nodes = list()
		for i in range(n_out):
			v_out = self.nn.add_vertex()
			self.out_nodes.append(v_out)
			#Initialize unit
			self.nn.vp.bias[v_out] = self.param_initializer_func(self.min_w, self.max_w)
			self.nn.vp.activations[v_out] = 0.
			self.nn.vp.outputs[v_out] = 0.
			self.nn.vp.out_ok[v_out] = False
			self.nn.vp.is_in[v_out] = False
			self.nn.vp.is_out[v_out] = True
			for v_in in self.in_nodes:
				e = self.nn.add_edge(v_in, v_out)
				self.nn.ep.weights[e] = self.param_initializer_func(self.min_w, self.max_w)
		
		# Initialize hidden nodes list
		self.hidden_nodes = list()
	
	def _propagate(self):
		nodes_to_activate = self.hidden_nodes + self.out_nodes
		while nodes_to_activate:
			new_nodes = list()
			for n in nodes_to_activate:
				a = 0.
				good = True
				for e in n.in_edges():
					in_neigh = e.source()
					if(self.nn.vp.out_ok[in_neigh] == True): # If output has been computed
						a += self.nn.vp.outputs[in_neigh]*self.nn.ep.weights[e] # Compute contribution to activation
					else:
						good = False
						break
				if(good): # If all in_neighbours could be processed
					self.nn.vp.activations[n] = a # Set activtion
					if n in self.out_nodes: # If output node...
						self.nn.vp.outputs[n] = self.af_out(a)  # ...use output activation func
					else:
						self.nn.vp.outputs[n] = self.af(a) # ...else use regular activation func
					self.nn.vp.out_ok[n] = True # the neuron has been processed
				else:
					new_nodes.append(n) # We will try again at next iteration
			nodes_to_activate = new_nodes # Update list
	
	
	def output(self):
		outs = list()
		# Read output
		for v_out in self.out_nodes:
			assert (self.nn.vp.out_ok[v_out]), "Error - output has not been computed"
			outs.append(self.nn.vp.outputs[v_out])
		return outs
	
	
	def step(self,inputs):
		assert (len(inputs) == len(self.in_nodes)), "Wrong NN input dimension"
		# Invalidate all nodes
		for v in self.nn.vertices():
			self.nn.vp.out_ok[v] = False
		# Set inputs
		for (i, v_in) in enumerate(self.in_nodes):
			v_in = self.in_nodes[i]
			self.nn.vp.outputs[v_in] = inputs[i]
			self.nn.vp.out_ok[v_in] = True
		# Propagate
		self._propagate()
		# Read and return output
		return self.output()
	
	def __getstate__(self):
		state = self.__dict__.copy()
		# Vertex list not inside a Graph are not picklable
		# Remove them and rebuild them from graph info at unpickle time
		del state["hidden_nodes"]
		del state["out_nodes"]
		del state["in_nodes"]
		return state
	
	def __setstate__(self, state):
		# Restore picklable members
		self.__dict__.update(state)
		# Vertex list not inside a Graph are not picklable
		# Rebuild them from graph info at unpickle time
		self.in_nodes = list()
		self.hidden_nodes = list()
		self.out_nodes = list()
		for v in self.nn.vertices():
			if(self.nn.vp.is_in[v]):
				self.in_nodes.append(v)
			elif(self.nn.vp.is_out[v]):
				self.out_nodes.append(v)
			else:
				self.hidden_nodes.append(v)


class DNNController: # Wrapper compatible with fixed structure controller API
	def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5, params=None):
		#TODO: implement initialization with hidden neurons
		self.n_in = n_in
		self.n_out = n_out
		self.dnn = DNN(n_in, n_out)
	
	def get_parameters(self):
		return self.dnn
	
	def set_parameters(self, dnn):
		self.dnn = dnn
	
	def __call__(self,x):
		return self.dnn.step(x)
	
	def predict(self,x):
		if(len(x.shape) == 1):
			return self.__call__(x)
		else:
			# TODO: Implement - maybe
			return None

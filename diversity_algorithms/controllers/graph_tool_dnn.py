#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    31/07/2019
""" 

import numpy as np
import pickle as pk
import sys

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


max_tries_new_conn = 100 # How many times the algorithms will try to find a new valid connection target before giving up

class DNN:
	def __init__(self, n_in, n_out, n_hidden=0, activations=sigmoid, activations_out=tanh, min_w=-5, max_w=5, param_initializer_func=np.random.uniform):
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
			self.nn.vp.bias[v_out] = self._random_weight()
			self.nn.vp.activations[v_out] = 0.
			self.nn.vp.outputs[v_out] = 0.
			self.nn.vp.out_ok[v_out] = False
			self.nn.vp.is_in[v_out] = False
			self.nn.vp.is_out[v_out] = True
			for v_in in self.in_nodes:
				e = self.nn.add_edge(v_in, v_out)
				self.nn.ep.weights[e] = self._random_weight()
		
		# Initialize hidden nodes list
		self.hidden_nodes = list()
		# Add random hidden units
		for i in range(n_hidden):
			# Pick random edge
			edges = list(self.nn.edges())
			e = np.random.choice(edges)
			# Put a new unit on it with random weights
			self._add_node_on_edge(e, weights="random")
	
	def _random_weight(self):
		return self.param_initializer_func(self.min_w, self.max_w)
	
	def _propagate(self):
		nodes_to_activate = self.hidden_nodes + self.out_nodes
		while nodes_to_activate:
			new_nodes = list()
			for n in nodes_to_activate:
				a = 0.
				good = True
				try:
					for e in n.in_edges():
						in_neigh = e.source()
						if(self.nn.vp.out_ok[in_neigh]): # If output has been computed
							a += self.nn.vp.outputs[in_neigh]*self.nn.ep.weights[e] # Compute contribution to activation
						else:
							good = False
							break
					if(good): # If all in_neighbours could be processed
						a += self.nn.vp.bias[n] # Add the bias
						self.nn.vp.activations[n] = a # Set activtion
						if n in self.out_nodes: # If output node...
							self.nn.vp.outputs[n] = self.af_out(a)  # ...use output activation func
						else:
							self.nn.vp.outputs[n] = self.af(a) # ...else use regular activation func
						self.nn.vp.out_ok[n] = True # the neuron has been processed
					else:
						new_nodes.append(n) # We will try again at next iteration
				except ValueError as e:
					print("*****************ERROR*****************")
					print("Exception: %s" % str(e))
					print("Pickling objects...")
					with open("err_graph.pk",'wb') as fd:
						pk.dump(self.nn,fd)
					with open("err_dnn.pk",'wb') as fd:
						pk.dump(self,fd)
					print("Current nodes_to_activate: %s" % str(nodes_to_activate))
					self.describe()
					print("in_nodes : %s" % str(self.in_nodes))
					print("out_nodes : %s" % str(self.out_nodes))
					print("hidden_nodes : %s" % str(self.hidden_nodes))
					print("Exiting...")
					sys.exit(1)
			nodes_to_activate = new_nodes # Update list
	
	
	def describe(self):
		print("Feedforward DNN with %d inputs, %d outputs, %d hidden units and %d connections" % (self.n_in(), self.n_out(), self.n_hidden(), self.n_conns()))
	
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
	
	def n_units(self): # Return the number of true neurons i.e. non-input vertices
		return (len(self.hidden_nodes)+len(self.out_nodes))

	def n_in(self): # Return the number of inputs
		return len(self.in_nodes)

	def n_out(self): # Return the number of outputs
		return len(self.out_nodes)

	def n_hidden(self): # Return the number of hidden neurons
		return len(self.hidden_nodes)
	
	def n_conns(self): # Return the number of connections
		return self.nn.num_edges()
	
	def _get_in_candidate(self): # A candidate for a new in edge is a non-output node
		return np.random.choice(self.hidden_nodes+self.in_nodes)
	
	def _get_out_candidate(self): # A candidate for a new in edge is a non-input node
		return np.random.choice(self.hidden_nodes+self.out_nodes)
	
	# Checks if adding an edge from in_candidate to out_candidate will create a loop. We don't want that for a FF network
	def check_no_loop(self, in_candidate, out_candidate):
		# See if there is a path from *out* to *in*. If there is, adding the edge would close the loop and we shouldn't
		d = shortest_distance(self.nn, out_candidate, in_candidate)
		return (d == np.iinfo('int32').max) # Returns max int32 value if no path
	
	# Checks if in_candidate and out_candidate are not already connected (dist = 1) or the same neuron (dist = 0)
	def check_not_same_or_neighbors(self, in_candidate, out_candidate):
		# See if the distance from in to out is not 1 (neighbours) or 0 (same neuron)
		d = shortest_distance(self.nn, in_candidate, out_candidate)
		return (d > 1)
	
	# Add a random connection between two neurons :
	# - Different
	# - Not already directly connected
	# - So that creating the connection would not create a cycle
	# - The out unit cannot be input and the in units cannot be outputs
	def add_random_conn(self):
		for i in range(max_tries_new_conn):
			v1 = self._get_in_candidate()
			v2 = self._get_out_candidate()
			if(self.check_no_loop(v1,v2) and self.check_not_same_or_neighbors(v1,v2)): # Valid pair - not actually the same neuron, not already connected, will not create a cycle
				e = self.nn.add_edge(v1, v2)
				self.nn.ep.weights[e] = self._random_weight()
				return True
		return False # could not create connection
	
	
	# Delete random connection
	def del_random_conn(self):
		edges = list(self.nn.edges())
		e = np.random.choice(edges)
		self.nn.remove_edge(e)
	
	# Add new hidden node on random edge
	def add_node_on_random_edge(self):
		edges = list(self.nn.edges())
		e = np.random.choice(edges)
		# Put a new unit on it with random weights
		self._add_node_on_edge(e)

	# Delete random hidden node
	def del_random_hidden(self):
		v = np.random.choice(self.hidden_nodes)
		self.hidden_nodes.remove(v)
		self.nn.remove_vertex(v)
		self._regenerate_node_lists() # Necessary to remove issus
	
	def _add_node_on_edge(self,e,weights="copytoboth"):
		# Determine weights of future edges
		old_w = self.nn.ep.weights[e]
		if(weights=="copytoboth"):
			w1 = old_w
			w2 = old_w
		elif(weights=="random"):
			w1 = self._random_weight()
			w2 = self._random_weight()
		v1 = e.source()
		v2 = e.target()
		# Remove edge
		self.nn.remove_edge(e)
		# Create new node and its attributes
		v_new = self.nn.add_vertex()
		self.nn.vp.bias[v_new] = self._random_weight()
		self.nn.vp.is_in[v_new] = False
		self.nn.vp.is_out[v_new] = False
		self.nn.vp.out_ok[v_new] = False
		self.hidden_nodes.append(v_new)
		# Create edges and weights
		e1 = self.nn.add_edge(v1, v_new)
		self.nn.ep.weights[e1] = w1
		e2 = self.nn.add_edge(v_new, v2)
		self.nn.ep.weights[e2] = w2
	
	# Rebuild node lists from graph info
	def _regenerate_node_lists(self):
		self.in_nodes.clear()
		self.out_nodes.clear()
		self.hidden_nodes.clear()
		for v in self.nn.vertices():
			if(self.nn.vp.is_in[v]):
				self.in_nodes.append(v)
			elif(self.nn.vp.is_out[v]):
				self.out_nodes.append(v)
			else:
				self.hidden_nodes.append(v)
	
	
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
		self._regenerate_node_lists()



class DNNController: # Wrapper compatible with fixed structure controller API
	def __init__(self, n_in, n_out, n_hidden=5, params=None):
		#TODO: implement initialization with hidden neurons
		self.n_in = n_in
		self.n_out = n_out
		self.dnn = DNN(n_in, n_out, n_hidden)
	
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

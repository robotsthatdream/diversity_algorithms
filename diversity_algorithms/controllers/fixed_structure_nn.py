
# coding: utf-8

import numpy as np

## Suppress TF info messages

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

 
# ## Controller

import keras
from keras.layers import Dense, Input
from keras.models import Model

def gen_simplemlp(n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5):
    n_neurons = [n_neurons_per_hidden]*n_hidden_layers if np.isscalar(n_neurons_per_hidden) else n_neurons_per_hidden
    i = Input(shape=(n_in,))
    x = i
    for n in n_neurons:
        x = Dense(n, activation='sigmoid')(x)
    o = Dense(n_out, activation='tanh')(x)
    m = Model(inputs=i, outputs=o)
    return m
    

class SimpleNeuralControllerKeras():
    def __init__(self, n_in, n_out, n_hidden_layers=2, n_neurons_per_hidden=5, params=None):
        self.dim_in = n_in
        self.dim_out = n_out

        # if params is provided, we look for the number of hidden layers and neuron per layer into that parameter (a dicttionary)
        if (not params==None):
            if ("n_hidden_layers" in params.keys()):
                n_hidden_layers=params["n_hidden_layers"]
            if ("n_neurons_per_hidden" in params.keys()):
                n_neurons_per_hidden=params["n_neurons_per_hidden"]
        #print("Creating a simple mlp with %d inputs, %d outputs, %d hidden layers and %d neurons per layer"%(n_in, n_out,n_hidden_layers, n_neurons_per_hidden))
        self.keras_model = gen_simplemlp(self.dim_in, self.dim_out, n_hidden_layers, n_neurons_per_hidden)
        self.keras_model.compile(optimizer='sgd',loss='mse') # useless ?
        
        keras_weights = self.keras_model.get_weights()
        self.weights_shape = [arr.shape for arr in keras_weights]
        self.n_weights = np.sum([np.product(s) for s in self.weights_shape])
    
    def get_parameters(self):
        """
        Returns all network parameters as a single array
        """
        keras_weights = self.keras_model.get_weights()
        flat_weights = np.hstack([arr.flatten() for arr in keras_weights])
        return flat_weights

    def set_parameters(self, flat_parameters):
        """
        Set all network parameters from a single array
        """
        i = 0 # index
        to_set = []
        for arr_shape in self.weights_shape:
            n_weights_arr = np.product(arr_shape)
            flat_weights_arr = np.array(flat_parameters[i:(i+n_weights_arr)])
            to_set.append(flat_weights_arr.reshape(arr_shape))
            i += n_weights_arr
        self.keras_model.set_weights(to_set)
    
    def predict(self,x):
        """
        Wrapper for the Keras model's predict. Takes both individual examples
        and batches 
        """
        if(len(x.shape) == 1):
            y = self.predict(x.reshape(1,x.shape[0]))
            return y[0]
        else:
            y_batch = self.keras_model.predict(x)
            return y_batch

    def __call__(self,x):
        """Calling the controller calls predict"""
        return self.predict(x)



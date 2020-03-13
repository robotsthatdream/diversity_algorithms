#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    19/12/2019
""" 
import os

from diversity_algorithms.environments.behavior_descriptors import *

from diversity_algorithms.environments import gym_env, dummy_env

registered_environments = dict()

registered_environments["Fastsim-LS2011"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{}}, # Default
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-LS2011-EnergyFitness"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"reward_func":"minimize_energy"},
		"output":"total_reward"},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-Pugh2015"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/pugh_maze.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-16x16"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/realhard_maze.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-12x12"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/maze_12x12.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-8x8"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/maze_8x8.xml"}},
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["BipedalWalker"] = {
	"bd_func": bipedal_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"BipedalWalker-v2",
		"gym_params":{}}, # Default
	"grid_features": {
		"min_x": [-600,-600],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}


registered_environments["DummyMapping3D"] = {
	"eval": dummy_env.SimpleMappingEvaluator,
	"eval_params": {
		"geno_size":3,
		"mapping":"fitness_last"}, # Default
	"grid_features": {
		"min_x": [-5,-5],
		"max_x": [5, 5],
		"nb_bin": 50
	}
}

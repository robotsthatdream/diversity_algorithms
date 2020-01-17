#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alex Coninx
    ISIR - Sorbonne Universite / CNRS
    19/12/2019
""" 
import os

from diversity_algorithms.environments.behavior_descriptors import *

from diversity_algorithms.environments import gym_env

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

registered_environments["Fastsim-Pugh2015"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/pugh_maze.xml"}}, # should be reasonably robust
	"grid_features": {
		"min_x": [0,0],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}

registered_environments["Fastsim-16x16realhard"] = {
	"bd_func": maze_behavior_descriptor,
	"eval": gym_env.EvaluationFunctor,
	"eval_params": {
		"gym_env_name":"FastsimSimpleNavigation-v0",
		"gym_params":{"xml_env":os.path.dirname(os.path.realpath(__file__))+"/assets/fastsim/realhard_maze.xml"}}, # should be reasonably robust
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
		"min_x": [-600,600],
		"max_x": [600, 600],
		"nb_bin": 50
	}
}



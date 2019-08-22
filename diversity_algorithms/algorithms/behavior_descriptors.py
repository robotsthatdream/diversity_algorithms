#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Alexandre Coninx
    ISIR CNRS/UPMC
    22/08/2019
""" 

def maze_behavior_descriptor(traj):
	"""
	Computes the behavior descriptor from a trajectorty.
	Computes the behavior descriptor from a trajectory. A trajectory is a list of tuples (obs,reward,end,info)  (depends on the environment, see gym_env.py). For the maze, we output the last robot position (x,y only, we discard theta).
	"""
	last_step_data=traj[-1]
	last_info=last_step_data[3]
	return last_info['robot_pos'][:2]

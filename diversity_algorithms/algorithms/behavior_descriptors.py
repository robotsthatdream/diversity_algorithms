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

def bipedal_behavior_descriptor(traj):
        """
        Computes the behavior descriptor from a trajectorty.

        Computes the behavior descriptor from a trajectory. A trajectory is a list of tuples (state,reward,end,info)  (depends on the environment, see gym_env.py). 
        """
        states=[x[0] for x in traj]
        horizontal_speed=[x[2] for x in states]
        vertical_speed=[x[3] for x in states]
        return [sum(horizontal_speed),sum(vertical_speed)]


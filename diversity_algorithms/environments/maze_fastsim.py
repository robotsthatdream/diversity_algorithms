
# coding: utf-8

# pyMaze expriments

import gym, gym_fastsim
import numpy as np
import time
#import resource

from diversity_algorithms.controllers.fixed_structure_nn import SimpleNeuralController

# Fitness/evaluation function

default_max_step = 2000 # same as C++ sferes experiments

def generate_gym_env_and_controller(controller_type=SimpleNeuralController, params=None):
	"""
	Generate a gym environmenent and a controller.
	"""
	print("Generate gym env + controller")
	env = gym.make('FastsimSimpleNavigation-v0')
	env.reset()
	controller = controller_type(env.observation_space.shape[0],env.action_space.shape[0], params=params)
	return env, controller


current_serial=1
do_dump=False



class EvaluationFunctor:
	def __init__(self, env, controller, output='dist_to_goal',max_step=default_max_step, with_behavior_descriptor=False,dump=do_dump):
		global current_serial
		print("Eval functor created")
		self.env = env
		self.controller = controller
		self.out = output
		self.max_step = max_step
		self.with_behavior_descriptor = with_behavior_descriptor
		self.serial = current_serial
		current_serial += 1
		self.dump = dump
		self.evals = 0
	
	#def evaluate_maze(self, )
	
	
	def load_indiv(self, genotype):
		self.controller.set_parameters(genotype)
	
	
	def evaluate_maze(self):
		"""
		Evaluate individual genotype (list of controller.n_weights floats) in environment env using
		given controller and max step number, and returns the required output:
		- dist_to_goal: final distance to goal (list of 1 scalar)
		- bd_finalpos: final robot position and orientation (list [x,y,theta])
		- total_reward: cumulated reward on episode (list of 1 scalar)
		"""
		# Inits
		#print("env reset")
		self.evals += 1
		initial_obs = self.env.reset()
		action_scale_factor = self.env.action_space.high # The nn generate an output in ]-1;1[ (tanh out layer); this scales to action space range
		obs = initial_obs
		total_reward = 0.
		# Dumping setup
		if (self.dump): #(nb_ind%10==0):
			dump_traj=open("traj_%06d.log"%(self.serial),"w")
		else:
			dump_traj=None
		# Main loop
		then = time.time()
		for i in range(self.max_step):
			self.env.render()
			action = action_scale_factor*self.controller(np.array(obs))
			obs, reward, end, info = self.env.step(action) # take a random action
			total_reward += reward
			#print("Step %d : Obs=%s Action=%s Reward=%f  Dist. to objective=%f  Robot position=%s  End of ep=%s" % (i, str(obs), str(action), reward, info["dist_obj"], str(in	fo["robot_pos"]), str(end)))
			if(dump_traj):
				dump_traj.write(" ".join(map(str,info["robot_pos"]))+"\n")
			if end:
				break
		now = time.time()
		#print("Main eval loop exit; took %f" % (now - then))
		if(dump_traj):
			dump_traj.write("# dist: %f total_reward: %f\n"%(info['dist_obj'],total_reward))
			dump_traj.close()
		#print("Eval ran for %d timesteps; final distance %f pos %s" % (i+1, info['dist_obj'], str(info['robot_pos'])))
		return reward, end, total_reward, info

	
	def __call__(self, genotype):
		#print("Eval functor CALL")
		# Load genotype
		#print("Load gen")
		if(type(genotype)==tuple):
			gen, ngeneration, idx = genotype
#			print("Start main eval loop -- #%d evals for this functor so far" % self.evals)
#			print("Evaluating indiv %d of gen %d" % (idx, ngeneration))
#			print('Eval thread: memory usage: %s (kb)' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
		else:
			gen = genotype
		self.load_indiv(gen)
		# Run eval genotype
		#print("Start eval")
		final_reward, end, total_reward, info = self.evaluate_maze()
		#print("Eval done !")
		# Select fitness
		
		if(self.out=='dist_to_goal'):
			fitness = [info['dist_obj']]
		elif(self.out=='bd_finalpos'):
			fitness = info['robot_pos'] # dim 3 - pos and angle
		elif(self.out=='total_reward'):
			fitness = [total_reward]
		elif(self.out=='final_reward'):
			fitness = [final_reward]
		elif(self.out==None or self.out=='none'):
			fitness = [None]
		else:
			print("ERROR: No known output %s" % output)
			return None
		
		if not self.with_behavior_descriptor:
			return fitness
		else:
			return [fitness,info['robot_pos'][:2]]



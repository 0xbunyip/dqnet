#!/usr/bin/env python

import os.path, sys
import argparse
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__))
							, os.pardir, os.pardir))
from grid_environment import Environment
from agent import Agent
from network import Network
import numpy as np

############### Hyper-parameters ###############
Environment.BUFFER_LEN = 1
Environment.EPISODE_STEPS = 18000
Environment.EPOCH_COUNT = 20
Environment.EPOCH_STEPS = 10000
Environment.EVAL_EPS = 0.05
Environment.FRAMES_SKIP = 1
Environment.FRAME_HEIGHT = 4
Environment.FRAME_WIDTH = 4
Environment.MAX_NO_OP = 0
Environment.MAX_REWARD = 0

Agent.DISCOUNT = 0.99
Agent.EXPLORE_FRAMES = 100000
Agent.FINAL_EXPLORE = 0.1
Agent.HISTORY = 2
Agent.INIT_EXPLORE = 1.0
Agent.MINIBATCH_SIZE = 32
Agent.REPLAY_SIZE = 500000
Agent.REPLAY_START = 5000
Agent.UPDATE_FREQ = 4
Agent.VALID_SIZE = 1024

Network.CLONE_FREQ = 10000
Network.GRAD_MOMENTUM = 0.95
Network.INPUT_SCALE = 20.0
Network.LEARNING_RATE = 0.0025
Network.MAX_ERROR = 0.0
Network.MIN_SGRAD = 0.01
Network.SGRAD_MOMENTUM = 0.95
################################################

def get_arguments(argv):
	parser = argparse.ArgumentParser(description = 'Run deep Q-network')
	parser.add_argument('-e', '--evaluate-only', dest = 'evaluating'
		, action = 'store_true'
		, help = 'Enable evaluating process (default: %(default)s)')
	parser.add_argument('-d', '--display-screen', dest = 'display_screen'
		, action = 'store_true'
		, help = 'Display screen while evaluating')

	parser.add_argument('-t', '--network-type', dest = 'network_type'
		, default = 'grid', type = str
		, choices = ['nature', 'nips', 'simple', 'bandit', 'grid', 'linear']
		, help = 'Type of network to use as function approximator')
	parser.add_argument('-a', '--algorithm', dest = 'algorithm'
		, default = 'double_q_learning', type = str
		, choices = ['q_learning', 'double_q_learning']
		, help = "Reinforcement learning algorithm to use as update rules")

	parser.add_argument('-f', '--file-network', dest = 'network_file'
		, default = None
		, help = 'Network file to load from')
	parser.add_argument('-i', '--ignore-layers', dest = 'ignore_layers'
		, default = 0, type = int
		, help = "Number of layers of params in network file to ignore")

	parser.add_argument('-x', '--file-exp', dest = 'exp_file'
		, default = None
		, help = "Experience file to load from")

	parser.add_argument('-o', '--continue-folder', dest = 'continue_folder'
		, default = None
		, help = "Folder to load network and experience to continue training")
	parser.add_argument('-c', '--continue-epoch', dest = 'continue_epoch'
		, default = 0, type = int
		, help = "Last epoch before continue training")

	parser.add_argument('-y', '--store-frequency', dest = 'store_frequency'
		, default = -1, type = int
		, help = "Save experience every this amount of epoch"\
					" (-1 for no save, 0 to save at last epoch)")

	parser.add_argument('-u', '--random-run', dest = 'random_run'
		, action = 'store_true'
		, help = 'Totally randomize train/evaluate process')
	return parser.parse_args(argv)

def build_rl_components(argv):
	arg = get_arguments(argv)
	if arg.random_run:
		rng = np.random.RandomState()
	else:
		rng = np.random.RandomState(333)

	env = Environment(rng, display_screen = arg.display_screen)

	if arg.continue_folder is not None:
		network_file = arg.continue_folder +\
						 '/network/%03d.npz' % (arg.continue_epoch)
		exp_file = arg.continue_folder + '/network/exp.npz'

		agn = Agent(env.get_action_count()
					, Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, rng
					, arg.network_type, arg.algorithm, network_file
					, arg.ignore_layers, exp_file)
	else:
		agn = Agent(env.get_action_count()
					, Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, rng
					, arg.network_type, arg.algorithm, arg.network_file
					, arg.ignore_layers, arg.exp_file)
	return env, agn

def main(argv):
	arg = get_arguments(argv)

	if not arg.evaluating:
		env, agn = build_rl_components(argv)
		env.train(agn, arg.store_frequency, arg.continue_folder
					, arg.continue_epoch, ask_for_more = True)
	elif arg.network_file is not None:
		env, agn = build_rl_components(argv)
		env.evaluate(agn)

if __name__ == '__main__':
	main(sys.argv[1:])
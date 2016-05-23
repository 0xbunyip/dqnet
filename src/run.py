#!/usr/bin/env python

from environment import Environment
from agent import Agent
from network import Network
import numpy as np
import argparse
import sys

############### Hyper-parameters ###############
Environment.BUFFER_LEN = 2
Environment.EPISODE_STEPS = 18000
Environment.EPOCH_COUNT = 200
Environment.EPOCH_STEPS = 250000
Environment.FRAMES_SKIP = 4
Environment.FRAME_HEIGHT = 84
Environment.FRAME_WIDTH = 84
Environment.MAX_NO_OP = 30
Environment.MAX_REWARD = 1

Agent.DISCOUNT = 0.99
Agent.EXPLORE_FRAMES = 1000000
Agent.FINAL_EXPLORE = 0.1
Agent.HISTORY = 4
Agent.INIT_EXPLORE = 1.0
Agent.MINIBATCH_SIZE = 32
Agent.REPLAY_SIZE = 500000
Agent.REPLAY_START = 50000
Agent.UPDATE_FREQ = 4
Agent.VALID_SIZE = 3200

Network.CLONE_FREQ = 10000
Network.GRAD_MOMENTUM = 0.95
Network.INPUT_SCALE = 255.0
Network.LEARNING_RATE = 0.00025
Network.MAX_ERROR = 1.0
Network.MIN_SGRAD = 0.01
Network.SGRAD_MOMENTUM = 0.95
################################################

def get_arguments(argv):
	parser = argparse.ArgumentParser(description = 'Run deep Q-network')
	parser.add_argument('-r', '--rom', dest = 'rom_name'
		, default = 'breakout.bin'
		, help = 'ROM file name without path (default: %(default)s)')
	parser.add_argument('-e', '--evaluate-only', dest = 'evaluating'
		, action = 'store_true'
		, help = 'Enable evaluating process (default: %(default)s)')
	parser.add_argument('-d', '--display-screen', dest = 'display_screen'
		, action = 'store_true'
		, help = 'Display screen while evaluating')
	parser.add_argument('-t', '--network-type', dest = 'network_type'
		, default = 'nature', type = str
		, choices = ['nature', 'nips', 'simple', 'bandit', 'grid', 'linear']
		, help = 'Type of network to use as function approximator')
	parser.add_argument('-f', '--file-network', dest = 'network_file'
		, default = None
		, help = 'Network file to load from')
	parser.add_argument('-u', '--random-run', dest = 'random_run'
		, action = 'store_true'
		, help = 'Totally randomize train/evaluate process')
	return parser.parse_args(argv)

def main(argv):
	arg = get_arguments(argv)

	if arg.random_run:
		rng = np.random.RandomState()
	else:
		rng = np.random.RandomState(333)

	if not arg.evaluating:
		env = Environment(arg.rom_name, rng
						, display_screen = arg.display_screen)
		agn = Agent(env.get_action_count()
				, Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH
				, rng, arg.network_type)
		env.train(agn)
	elif arg.network_file is not None:
		env = Environment(arg.rom_name, rng
						, display_screen = arg.display_screen)
		agn = Agent(env.get_action_count()
						, Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH
						, rng, arg.network_type, arg.network_file)
		env.evaluate(agn)

if __name__ == '__main__':
	main(sys.argv[1:])
	
#!/usr/bin/env python

from environment import Environment
from agent import Agent
from network import Network
import numpy as np
import argparse
import sys

############### Hyper-parameters ###############
Environment.EPOCH_COUNT = 200
Environment.FRAMES_SKIP = 4
Environment.FRAME_HEIGHT = 84
Environment.FRAME_WIDTH = 84
Environment.MAX_NO_OP = 30
Environment.MAX_REWARD = 1
Environment.ORIGINAL_HEIGHT = 210
Environment.ORIGINAL_WIDTH = 160
Environment.STEPS_PER_EPISODE = 18000
Environment.STEPS_PER_EPOCH = 50000

Agent.AGENT_HISTORY_LENGTH = 4
Agent.DISCOUNT_FACTOR = 0.95
Agent.FINAL_EXPLORATION = 0.1
Agent.FINAL_EXPLORATION_FRAME = 1000000
Agent.INITIAL_EXPLORATION = 1.0
Agent.MINIBATCH_SIZE = 32
Agent.REPLAY_MEMORY_SIZE = 1000000
Agent.REPLAY_START_SIZE = 50000
Agent.TARGET_NETWORK_UPDATE_FREQUENCY = 10000
Agent.UPDATE_FREQUENCY = 4
Agent.VALIDATION_SET_SIZE = 2048 # MINIBATCH_SIZE <= VALIDATION_SET_SIZE <= REPLAY_START_SIZE * AGENT_HISTORY_LENGTH

Network.GRAD_MOMENTUM = 0.95
Network.LEARNING_RATE = 0.00025
Network.MAX_ERROR = 1.0
Network.MIN_SQR_GRAD = 0.01
Network.SCALE_FACTOR = 255.0
Network.SQR_GRAD_MOMENTUM = 0.95
################################################

def get_arguments(argv):
	parser = argparse.ArgumentParser(description = 'Train/Evaluate deep Q-network')
	parser.add_argument('-r', '--rom', dest = 'rom_name', default = 'breakout.bin'
		, help = 'ROM file name without path (default: %(default)s)')
	parser.add_argument('-e', '--evaluate-only', dest = 'evaluating', default = 0, type = int
		, help = 'Enable evaluating process (default: %(default)s)')
	parser.add_argument('-d', '--display-screen', dest = 'display_screen', default = 0, type = int
		, help = 'Display screen while evaluating')
	parser.add_argument('-f', '--file-network', dest = 'network_file', default = None
		, help = 'Network file to load from')
	return parser.parse_args(argv)

def main(argv):
	rng = np.random.RandomState(123)
	arg = get_arguments(argv)

	if not arg.evaluating:
		env = Environment(arg.rom_name, rng, display_screen = bool(arg.display_screen))
		agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, rng)
		env.train(agn)
		env.evaluate(agn)
	elif arg.network_file is not None:
		env = Environment(arg.rom_name, rng, display_screen = bool(arg.display_screen))
		agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, rng, arg.network_file)
		env.evaluate(agn)

if __name__ == '__main__':
	main(sys.argv[1:])
	
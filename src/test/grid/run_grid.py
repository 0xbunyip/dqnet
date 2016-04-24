#!/usr/bin/env python

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from grid_environment import Environment
from grid_agent import Agent
from grid_network import Network
import numpy as np

rng = np.random.RandomState(123)

############### Hyper-parameters ###############
Environment.STEPS_PER_EPOCH = 10000
Environment.TRAIN_FRAMES_SKIP = 1
Environment.EVAL_FRAMES_SKIP = 1
Environment.ORIGINAL_HEIGHT = 3
Environment.ORIGINAL_WIDTH = 3
Environment.FRAME_HEIGHT = 3
Environment.FRAME_WIDTH = 3

Agent.REPLAY_MEMORY_SIZE = 50000
Agent.REPLAY_START_SIZE = 500
Agent.AGENT_HISTORY_LENGTH = 1
Agent.UPDATE_FREQUENCY = 4
Agent.TARGET_NETWORK_UPDATE_FREQUENCY = 500
Agent.DISCOUNT_FACTOR = 0.99
Agent.INITIAL_EXPLORATION = 1.0
Agent.FINAL_EXPLORATION = 0.1
Agent.FINAL_EXPLORATION_FRAME = 100000
Agent.MINIBATCH_SIZE = 32

Network.LEARNING_RATE = 0.0025
Network.SCALE_FACTOR = 15.0

VALIDATION_SET_SIZE = 128 # should be <= REPLAY_START_SIZE * AGENT_HISTORY_LENGTH
################################################

def main():
	mbsize = 32
	# f = open('log.txt', 'w')
	# sys.stdout = f
	env = Environment(rng, display_screen = False)
	agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, mbsize, rng)
	env.train(agn, epoch_count = 1)
	# f.close()

if __name__ == '__main__':
	main()
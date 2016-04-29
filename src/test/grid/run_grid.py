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
Environment.ORIGINAL_HEIGHT = 4
Environment.ORIGINAL_WIDTH = 4
Environment.FRAME_HEIGHT = 4
Environment.FRAME_WIDTH = 4

Agent.REPLAY_MEMORY_SIZE = 50000
Agent.REPLAY_START_SIZE = 500
Agent.AGENT_HISTORY_LENGTH = 1
Agent.UPDATE_FREQUENCY = 4
Agent.TARGET_NETWORK_UPDATE_FREQUENCY = 10000
Agent.DISCOUNT_FACTOR = 0.99
Agent.INITIAL_EXPLORATION = 1.0
Agent.FINAL_EXPLORATION = 0.1
Agent.FINAL_EXPLORATION_FRAME = 100000
Agent.MINIBATCH_SIZE = 32

Network.LEARNING_RATE = 0.0025
Network.SCALE_FACTOR = 20.0

VALIDATION_SET_SIZE = 512 # MINIBATCH_SIZE <= VALIDATION_SET_SIZE <= REPLAY_START_SIZE * AGENT_HISTORY_LENGTH
################################################

def main():
	# f = open('log.txt', 'w')
	# sys.stdout = f
	env = Environment(rng, display_screen = False)
	agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, VALIDATION_SET_SIZE, rng)
	env.train(agn, epoch_count = 20, ask_for_more = True)
	# f.close()

if __name__ == '__main__':
	main()
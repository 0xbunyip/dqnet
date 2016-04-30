#!/usr/bin/env python

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from grid_environment import Environment
from agent import Agent
from network import Network
import numpy as np

############### Hyper-parameters ###############
Environment.FRAMES_SKIP = 1
Environment.FRAME_HEIGHT = 3
Environment.FRAME_WIDTH = 3
Environment.MAX_NO_OP = 0
Environment.MAX_REWARD = 0
Environment.ORIGINAL_HEIGHT = 3
Environment.ORIGINAL_WIDTH = 3
Environment.STEPS_PER_EPISODE = 100
Environment.STEPS_PER_EPOCH = 10000

Agent.AGENT_HISTORY_LENGTH = 1
Agent.DISCOUNT_FACTOR = 0.99
Agent.FINAL_EXPLORATION = 0.1
Agent.FINAL_EXPLORATION_FRAME = 100000
Agent.INITIAL_EXPLORATION = 1.0
Agent.MINIBATCH_SIZE = 32
Agent.REPLAY_MEMORY_SIZE = 50000
Agent.REPLAY_START_SIZE = 500
Agent.TARGET_NETWORK_UPDATE_FREQUENCY = 10000
Agent.UPDATE_FREQUENCY = 4
Agent.VALIDATION_SET_SIZE = 256 # MINIBATCH_SIZE <= VALIDATION_SET_SIZE <= REPLAY_START_SIZE * AGENT_HISTORY_LENGTH

Network.LEARNING_RATE = 0.0025
Network.MAX_DELTA = 0.0
Network.SCALE_FACTOR = 20.0
################################################

def main():
	rng = np.random.RandomState(123)
	env = Environment(rng, display_screen = False)
	agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, rng)
	env.train(agn, epoch_count = 20, ask_for_more = True)

if __name__ == '__main__':
	main()
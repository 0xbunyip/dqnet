#!/usr/bin/env python

import sys

from environment import Environment
from agent import Agent
from network import Network
import numpy as np
rng = np.random.RandomState(123)

############### Hyper-parameters ###############
Environment.STEPS_PER_EPOCH = 200
Environment.TRAIN_FRAMES_SKIP = 4
Environment.EVAL_FRAMES_SKIP = 6
Environment.ORIGINAL_HEIGHT = 210
Environment.ORIGINAL_WIDTH = 160
Environment.FRAME_HEIGHT = 8
Environment.FRAME_WIDTH = 8

Agent.REPLAY_MEMORY_SIZE = 1000
Agent.REPLAY_START_SIZE = 50
Agent.AGENT_HISTORY_LENGTH = 4
Agent.UPDATE_FREQUENCY = 4
Agent.TARGET_NETWORK_UPDATE_FREQUENCY = 50
Agent.DISCOUNT_FACTOR = 0.99
Agent.INITIAL_EXPLORATION = 1.0
Agent.FINAL_EXPLORATION = 0.1
Agent.FINAL_EXPLORATION_FRAME = 100
Agent.MINIBATCH_SIZE = 32

Network.LEARNING_RATE = 0.0025
Network.SCALE_FACTOR = 255.0

VALIDATION_SET_SIZE = 32 # should be <= REPLAY_START_SIZE
################################################

def main():
	f = open('log.txt', 'w')
	sys.stdout = f
	env = Environment(rom_name = 'breakout.bin', display_screen = False)
	agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, VALIDATION_SET_SIZE, rng)
	env.train(agn, epoch_count = 2)
	f.close()

if __name__ == '__main__':
	main()
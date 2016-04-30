#!/usr/bin/env python

from environment import Environment
from agent import Agent
from network import Network
import numpy as np

############### Hyper-parameters ###############
Environment.FRAMES_SKIP = 4
Environment.FRAME_HEIGHT = 8
Environment.FRAME_WIDTH = 8
Environment.MAX_NO_OP = 30
Environment.MAX_REWARD = 0
Environment.ORIGINAL_HEIGHT = 210
Environment.ORIGINAL_WIDTH = 160
Environment.STEPS_PER_EPISODE = 100
Environment.STEPS_PER_EPOCH = 200

Agent.AGENT_HISTORY_LENGTH = 4
Agent.DISCOUNT_FACTOR = 0.99
Agent.FINAL_EXPLORATION = 0.1
Agent.FINAL_EXPLORATION_FRAME = 100
Agent.INITIAL_EXPLORATION = 1.0
Agent.MINIBATCH_SIZE = 32
Agent.REPLAY_MEMORY_SIZE = 100
Agent.REPLAY_START_SIZE = 50
Agent.TARGET_NETWORK_UPDATE_FREQUENCY = 50
Agent.UPDATE_FREQUENCY = 4
Agent.VALIDATION_SET_SIZE = 32 # MINIBATCH_SIZE <= VALIDATION_SET_SIZE <= REPLAY_START_SIZE * AGENT_HISTORY_LENGTH

Network.LEARNING_RATE = 0.0025
Network.MAX_DELTA = 1.0
Network.SCALE_FACTOR = 255.0
################################################

def main():
	rng = np.random.RandomState(123)
	env = Environment(rom_name = 'breakout.bin', rng = rng, display_screen = False)
	agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, rng)
	env.train(agn, epoch_count = 2)

if __name__ == '__main__':
	main()
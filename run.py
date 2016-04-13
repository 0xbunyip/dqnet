#!/usr/bin/env python

from environment import Environment
from agent import Agent

mbsize = 32
env = Environment(rom_name = 'breakout.bin', display_screen = False)
agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, 32)
env.hook_agent(agn)
env.train_agent(epoch_count = 2)
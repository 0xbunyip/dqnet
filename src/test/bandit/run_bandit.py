#!/usr/bin/env python

import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
from bandit_environment import Environment
from bandit_agent import Agent
import numpy as np

rng = np.random.RandomState(123)

mbsize = 32
env = Environment(rng, display_screen = False)
agn = Agent(env.get_action_count(), Environment.FRAME_HEIGHT, Environment.FRAME_WIDTH, mbsize, rng)
env.hook_agent(agn)
env.train_agent(epoch_count = 2)
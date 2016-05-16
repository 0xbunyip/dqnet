"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
import logging
import numpy as np
import cv2
import time
import os

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8


class ALEExperiment(object):
    def __init__(self, ale, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng, exp_pre):
        self.ale = ale
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        self.min_action_set = ale.getMinimalActionSet()
        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.width, self.height = ale.getScreenDims()

        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = np.empty((self.buffer_length,
                                       self.height, self.width),
                                      dtype=np.uint8)

        self.terminal_lol = False # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        self.rng = rng

        # CREATE A FOLDER TO HOLD RESULTS
        time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
        exp_dir = exp_pre + time_str + \
                       "{}".format(agent.network.learning_rate).replace(".", "p") \
                       + "_" \
                       + "{}".format(agent.network.discount).replace(".", "p")

        try:
            os.stat(exp_dir)
        except OSError:
            os.makedirs(exp_dir)

        self.exp_dir = exp_dir
        self.results_file = open(self.exp_dir + '/results.csv', 'w', 0)
        self.results_file.write(\
            'epoch,num_episodes,total_reward,reward_per_epoch,mean_q\n')
        self.results_file.flush()

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)

            with open(self.exp_dir + '/network_file_' + str(epoch) + \
                        '.pkl', 'wb') as f:
                self.agent.dump(f)
            # self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                # self.agent.start_testing()
                self.total_reward = 0
                self.episode_counter = 0
                self.run_epoch(epoch, self.test_length, True)

                holdout_sum = self.agent.get_validate_values()
                out = "{},{},{},{},{}\n".format(epoch, self.episode_counter, 
                    self.total_reward, self.total_reward / float(self.episode_counter),
                    holdout_sum)
                self.results_file.write(out)
                self.results_file.flush()
                # self.agent.finish_testing(epoch)

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        steps_left = num_steps
        while steps_left > 0:
            prefix = "testing" if testing else "training"
            logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(steps_left))
            _, num_steps = self.run_episode(steps_left, testing)

            if testing:
                self.episode_counter += 1
                self.total_reward += self.episode_reward

            steps_left -= num_steps


    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

        if not self.terminal_lol or self.ale.game_over():
            self.ale.reset_game()

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self._act(0) # Null action

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self._act(0)
        self._act(0)


    def _act(self, action):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        reward = self.ale.act(action)
        index = self.buffer_count % self.buffer_length

        self.ale.getScreenGrayscale(self.screen_buffer[index, ...])

        self.buffer_count += 1
        return reward

    def _step(self, action):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        for _ in range(self.frame_skip):
            reward += self._act(action)

        return reward

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode()
        self.episode_reward = 0

        start_lives = self.ale.lives()

        obs = self.get_observation()
        action, _ = self.agent.get_action(obs, 0.05, testing)
        # action = self.agent.start_episode(self.get_observation())
        num_steps = 0
        while True:
            reward = self._step(self.min_action_set[action])
            self.episode_reward += reward
            reward = np.clip(reward, -1, 1)
            # reward = self._step(self.min_action_set[action])

            self.terminal_lol = (self.death_ends_episode and not testing and
                                 self.ale.lives() < start_lives)
            terminal = self.ale.game_over() or self.terminal_lol
            num_steps += 1

            if terminal or num_steps >= max_steps:
                # self.agent.end_episode(reward, terminal)
                self.agent.add_experience(obs, True, action, reward, testing)
                break
            else:
                self.agent.add_experience(obs, False, action, reward, testing)

            # if terminal or num_steps >= max_steps:
            #     self.agent.end_episode(reward, terminal)
            #     break

            obs = self.get_observation()
            action, _ = self.agent.get_action(obs, 0.05, testing)
            # action = self.agent.step(reward, self.get_observation())
        return terminal, num_steps


    def get_observation(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.resize_image(max_image)

    def resize_image(self, image):
        """ Appropriately resize a single image """

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
                              crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'scale':
            return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')


"""
Gridworld class and util
"""

import random
import matplotlib.pyplot as plt
import numpy as np


class Gridworld:
    """
    Gridworld environment
    """
    index_to_direction = [
        [0, -1],
        [0, 1],
        [-1, 0],
        [1, 0]]

    def __init__(self, n, start_x, start_y, end_states, random_start):
        """
        Parameters:
        n - Side of grid
        start_x - start on non random reset
        start_y - start on non random reset
        end_states - List of end states
        random_start - bool indicating should start randomly on reset
        """
        self.start_x = start_x
        self.start_y = start_y
        self.x = self.start_x
        self.y = self.start_y
        self.n = n
        self.end_states = end_states
        self.random_start = random_start

    def is_valid(self, x, y):
        """
        Checks if given (x,y) is valid
        """
        if x < 0 or x >= self.n or y < 0 or y >= self.n:
            return False
        return True

    def reset(self):
        """
        Reset env
        """
        if self.random_start:
            self.x = random.randint(0, self.n-1)
            self.y = random.randint(0, self.n-1)
        else:
            self.x = self.start_x
            self.y = self.start_y
        return (self.x, self.y)

    def get_actions(self):
        """
        Get all possible actions at current (x,y)
        """
        actions = []
        for i in range(len(self.index_to_direction)):
            if self.is_valid(self.x + self.index_to_direction[i][0], self.y + self.index_to_direction[i][1]):
                actions.append(i)
        return actions

    def get_all_actions(self):
        """
        get all possible actions in env irrespective of state
        """
        return [i for i in range(len(self.index_to_direction))]

    def get_actions_at_state(self, state):
        """
            get valid actions at state
        """
        actions = []
        for i in range(len(self.index_to_direction)):
            if self.is_valid(state[0] + self.index_to_direction[i][0], state[1] + self.index_to_direction[i][1]):
                actions.append(i)
        return actions

    def get_random_action_at_state(self, state):
        """
            get some random action at state
        """
        return random.choice(self.get_actions_at_state(state))

    def get_random_action(self):
        """
            get random action at current x,y
        """
        return random.choice(self.get_actions())

    def step(self, action):
        """
            take a step from action
        """
        self.x = self.x + self.index_to_direction[action][0]
        self.y = self.y + self.index_to_direction[action][1]

        reward = -1
        done = False
        if (self.x, self.y) in self.end_states:
            reward = 0
            done = True

        return (self.x, self.y), reward, done, None

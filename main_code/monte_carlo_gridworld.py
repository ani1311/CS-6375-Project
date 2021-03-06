import random
import gym
from gridworld import *
from monte_carlo_util import *
from log_util import *

"""
class that performs monte carlo
"""


class MonteCarloGridworld:
    """
    monte carlo for RL
    """

    def __init__(self,  env, exploration):
        self.state_action = {}
        self.env = env
        self.exploration = exploration

    def get_action(self, observation):
        t = random.random()

        if t < self.exploration:
            return self.env.get_random_action()
        else:
            action = self.env.get_random_action()
            best_score = 0
            for k, v in self.state_action.items():
                if k[0] == observation and v[0] > best_score:
                    action = k[1]
                    best_score = v[0]
            return action

    def get_episode(self, no_of_steps):
        # (state,action,reward)
        episode = []
        observation = self.env.reset()

        if observation in self.env.end_states:
            return episode

        for _ in range(no_of_steps):
            action = self.get_action(observation)
            new_observation, reward, done, _ = self.env.step(action)

            episode.append([observation, action, reward])
            observation = new_observation
            if done:
                break
        return episode

    def eval_episode(self, episode):
        episode.reverse()
        reward_so_far = 0

        for state, action, reward in episode:
            reward_so_far = reward_so_far + reward
            if (state, action) in self.state_action:
                value, count = self.state_action[(state, action)]
                new_val = (value * count + reward_so_far) / (count + 1)
                self.state_action[(state, action)] = [new_val, count + 1]

            else:
                self.state_action[(state, action)] = [reward_so_far, 1]

    def train(self, no_of_eps, episode_length):
        for i in range(1, no_of_eps):
            ep = self.get_episode(episode_length)
    #         print(ep)
            self.eval_episode(ep)
            self.exploration = self.exploration * 0.9999


params = {
    'n': 15,
    'learning_rate': 0.7,
    'end_states': {(1, 7), (9, 12)}
}


def train():
    global params
    env = Gridworld(15, 3, 3, params['end_states'], True)
    mc = MonteCarloGridworld(env, params['learning_rate'])
    mc.train(1000, 100)

    save_heat_map_mc("mc_gridworld", params, mc.state_action, params['n'])
    #  print_heat_map(mc.state_action, env.n)

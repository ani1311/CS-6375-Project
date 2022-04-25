import random
import gym
from log_util import *


class TD0_Cartpole:
    neg_inf = 0

    def __init__(self, env, exploration, learning_rate, discount_rate, scale):
        self.state_action = {}
        self.env = env
        self.exploration = exploration
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.scale = scale

    def get_discrete_obs(self, observation):
        t = (observation * self.scale).astype(int)
        return tuple(t)

    def train(self, no_episodes, episode_length):
        avg_ep_len = 0
        for ep in range(no_episodes):
            obs = self.get_discrete_obs(self.env.reset())
            if ep % 500 == 0:
                print(ep, avg_ep_len/500)
                avg_ep_len = 0
            ep_len = 0
            for i in range(episode_length):
                action = self.get_action(obs)
                new_obs, reward, done, _ = self.env.step(action)
                new_obs = self.get_discrete_obs(new_obs)
                self.update_state_action(obs, new_obs, action, reward)
                obs = new_obs
                ep_len += 1
                if done:
                    break
            avg_ep_len += ep_len
            write_entry(ep, ep_len)
            self.exploration = self.exploration * 0.9995

    def set_state_action(self, sa):
        if sa not in self.state_action:
            self.state_action[sa] = self.neg_inf

    def get_action(self, obs):
        t = random.random()
        if t < self.exploration:
            return self.env.action_space.sample()
        else:
            act = self.env.action_space.sample()
            val = 0

            for action in range(self.env.action_space.n):
                sa = (obs, action)
                self.set_state_action(sa)
                if self.state_action[sa] > val:
                    act = action
                    val = self.state_action[sa]

            return act

    ## This is SARSA
    def update_state_action(self, obs, new_obs, action, reward):
        current_sa = (obs, action)
        self.set_state_action(current_sa)

        next_action = self.get_action(new_obs)
        next_sa = (new_obs, next_action)
        self.set_state_action(next_sa)

        self.state_action[current_sa] = self.state_action[current_sa] + self.learning_rate * \
            (reward + self.discount_rate *
             self.state_action[next_sa] - self.state_action[current_sa])

    def run(self, no_of_runs):
        for _ in range(no_of_runs):
            observation = self.get_discrete_obs(self.env.reset())

            while True:
                action = self.get_action(observation)
                new_observation, _, done, _ = self.env.step(action)
                new_observation = self.get_discrete_obs(new_observation)
                self.env.render()

                observation = new_observation

                if done:
                    break


def train():
    params = {
        'epsilon': 0.7,
        'learning_rate': 1,
        'decay': 1,
        'scale': 12
    }

    create_file("TD0_cartpole_exponential_decay", params)
    e = gym.make('CartPole-v1')
    tc = TD0_Cartpole(e, params['epsilon'], params['learning_rate'],
                      params['decay'], params['scale'])
    tc.train(10000, 502)
    save_file()

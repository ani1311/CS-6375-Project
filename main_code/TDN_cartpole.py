import random
from TD_util import *
import gym
from model_util import *
from log_util import *


class TDN_Cartpole:
    neg_inf = 0
    retardation = 0.999

    def __init__(self, env, n, exploration, learning_rate, discount_rate, scale):
        self.state_action = {}
        self.env = env
        self.n = n
        self.exploration = exploration
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.scale = scale

    def get_discrete_obs(self, observation):
        t = (observation * self.scale).astype(int)
        return tuple(t)

    def train(self, no_episodes, episode_length):
        global params
        avg_reward = 0
        for ep in range(1, no_episodes):
            states = []
            actions = []
            rewards = []
            #  self.new_states = 0
            #  self.old_states = 0

            obs = self.get_discrete_obs(self.env.reset())
            ep_reward = 0

            for i in range(episode_length):
                action = self.get_action(obs)
                new_obs, reward, done, _ = self.env.step(action)
                new_obs = self.get_discrete_obs(new_obs)

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = new_obs
                ep_reward += reward
                if len(states) > self.n:
                    states = states[1:]
                    actions = actions[1:]
                    rewards = rewards[1:]

                if done:
                    while len(states) > 1:
                        self.update_state_action(states, actions, rewards)
                        states = states[1:]
                        actions = actions[1:]
                        rewards = rewards[1:]
                    break

                if len(states) < self.n:
                    continue

                self.update_state_action(states, actions, rewards)

            #  print("Ep over")
            avg_reward += ep_reward
            write_entry(ep, ep_reward)

            if ep % 500 == 0:
                print(ep, avg_reward/500)
                #  print("% new states = ", (self.new_states/(self.new_states +
                #                                             self.old_states)))
                #
                #  print("% old states = ", (self.old_states/(self.new_states +
                #                                             self.old_states)))
                avg_reward = 0

            if params['epi_decay']:
                self.exploration = self.exploration * self.retardation

    def set_state_action(self, sa):
        if sa not in self.state_action:
            self.state_action[sa] = 0

    def get_best_action(self, obs):
        act = self.env.action_space.sample()
        val = self.neg_inf
        for action in range(self.env.action_space.n):
            sa = (obs, action)
            self.set_state_action(sa)
            if self.state_action[sa] > val:
                act = action
                val = self.state_action[sa]

        return act

    def get_action(self, obs):
        t = random.random()
        if t < self.exploration:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(obs)

    def run(self, episode_length):
        total_reward = 0
        obs = self.get_discrete_obs(self.env.reset())

        for i in range(episode_length):
            action = self.get_best_action(obs)
            new_obs, reward, done, _ = self.env.step(action)
            self.env.render()
            new_obs = self.get_discrete_obs(new_obs)
            total_reward += reward
            obs = new_obs

            if done:
                break
        return total_reward

    def update_state_action(self, states, actions, rewards):
        current_sa = (states[0], actions[0])
        self.set_state_action(current_sa)

        total_rewards = 0
        for i in range(len(rewards)):
            total_rewards += pow(self.discount_rate, i+1) * rewards[i]

        next_action = self.get_action(states[-1])
        next_sa = (states[-1], next_action)
        self.set_state_action(next_sa)

        self.state_action[current_sa] = self.state_action[current_sa] + self.learning_rate * (total_rewards + pow(
            self.discount_rate, self.n) * self.state_action[next_sa] - self.state_action[current_sa])


params = {
    'n': 20,
    'epsilon': 0.0001,
    'learning_rate': 0.8,
    'decay': 0.7,
    'scale': 13,
    'epi_decay': False,
    'load_model': True
}


def train():
    create_file("TD0_cartpole_exponential_decay", params)
    e = gym.make('CartPole-v0')

    td = TDN_Cartpole(e, params['n'], params['epsilon'], params['learning_rate'],
                      params['decay'], params['scale'])
    model_name = "saved-models/TDN-40-0.9-1-1-13.out"

    if params['load_model']:
        td.state_action = load_model(model_name)

    for i in range(3):
        td.train(10000, 500)
        save_model(td.state_action, model_name)
        print("Model saved")

    save_file()

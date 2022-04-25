import random
from gridworld import *
from TD_util import *
from log_util import *


class TDN_Gridworld:
    neg_inf = -99

    def __init__(self, env, n, exploration, learning_rate, discount_rate):
        self.state_action = {}
        self.env = env
        self.n = n
        self.exploration = exploration
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    def train(self, no_episodes=10, episode_length=10):
        for ep in range(no_episodes):
            states = []
            actions = []
            rewards = []

            obs = self.env.reset()
            if obs in self.env.end_states:
                continue

            for i in range(episode_length):
                action = self.get_action(obs)
                new_obs, reward, done, _ = self.env.step(action)

                states.append(obs)
                actions.append(action)
                rewards.append(reward)
                obs = new_obs
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
            self.exploration = self.exploration * 0.999

    def set_state_action(self, sa):
        if sa not in self.state_action:
            self.state_action[sa] = 0

    def get_best_action(self, obs):
        act = self.env.get_random_action_at_state(obs)
        val = self.neg_inf
        for action in self.env.get_actions_at_state(obs):
            sa = (obs, action)
            self.set_state_action(sa)
            if self.state_action[sa] > val:
                act = action
                val = self.state_action[sa]

        return act

    def get_action(self, obs):
        t = random.random()
        if t < self.exploration:
            return self.env.get_random_action_at_state(obs)
        else:
            return self.get_best_action(obs)

    def run(self, episode_length):
        total_reward = 0
        obs = self.env.reset()
        if obs in self.env.end_states:
            return

        for i in range(episode_length):
            print(obs)
            action = self.get_best_action(obs)
            new_obs, reward, done, _ = env.step(action)
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
        #  print("updating")
        #  print(current_sa, self.state_action[current_sa])

        self.state_action[current_sa] = self.state_action[current_sa] + self.learning_rate * (total_rewards + pow(
            self.discount_rate, self.n) * self.state_action[next_sa] - self.state_action[current_sa])

        #  print(current_sa, self.state_action[current_sa])


params = {
    'no_of_states': 30,
    'n': 4,
    'epsilon': 0.1,
    'learning_rate': 1,
    'decay': 1,
    'end_states': {(6, 9), (24, 29)}
}


def train():
    global params
    env = Gridworld(params['no_of_states'], 2, 2, params['end_states'], True)
    td = TDN_Gridworld(env, params['n'], params['epsilon'], params['learning_rate'],
                       params['decay'])
    td.train(10000, 90)

    save_heat_map_td("TDN_Gridworld", params, td.state_action, env.n)

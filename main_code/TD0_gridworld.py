import random
from gridworld import *
from TD_util import *


class TD0_Gridworld:
    neg_inf = -99

    def __init__(self, env, exploration, learning_rate, discount_rate):
        self.state_action = {}
        self.env = env
        self.exploration = exploration
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate

    def train(self, no_episodes, episode_length):
        for ep in range(no_episodes):
            obs = self.env.reset()
            if obs in self.env.end_states:
                continue
            for i in range(episode_length):
                action = self.get_action(obs)
                new_obs, reward, done, _ = env.step(action)
                self.update_state_action(obs, new_obs, action, reward)
                obs = new_obs
                if done:
                    break
            self.exploration = self.exploration * 0.999

    def set_state_action(self, sa):
        if sa not in self.state_action:
            self.state_action[sa] = self.neg_inf

    def get_action(self, obs):
        t = random.random()
        if t < self.exploration:
            return self.env.get_random_action()
        else:
            act = self.env.get_random_action()
            val = self.neg_inf
            for action in self.env.get_actions():
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

        #  print("current sa = ", current_sa,
        #        "next sa = ", next_sa,
        #        "current sa val = ", self.state_action[current_sa],
        #        "next sa value = ", self.state_action[next_sa],
        #        "reward = ", reward,
        #        "to_update = ", reward + self.discount_rate *
        #        self.state_action[next_sa] - self.state_action[current_sa],
        #        "updated val = ", self.state_action[current_sa] + self.learning_rate * (reward + self.discount_rate * self.state_action[next_sa] - self.state_action[current_sa]))
        self.state_action[current_sa] = self.state_action[current_sa] + self.learning_rate * (
            reward + self.discount_rate * self.state_action[next_sa] - self.state_action[current_sa])


env = Gridworld(15, 1, 1, [(0, 0)], True)
td = TD0_Gridworld(env, 0.1, 1, 1)
td.train(1000, 200)

print_heat_map(td.state_action, env.n)

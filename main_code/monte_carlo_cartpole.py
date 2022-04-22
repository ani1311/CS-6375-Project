import random
import gym

"""
class that performs monte carlo
"""


class MonteCarloCartpole:
    """
    monte carlo for RL
    """

    def __init__(self,  env):
        self.state_action = {}
        self.env = env
        self.scale = 12
        self.explore = 13

    def get_discrete_obs(self, observation):
        t = (observation * self.scale).astype(int)
        return tuple(t)

    def get_action(self, observation):
        t = random.random()

        action = self.env.action_space.sample()
        best_score = 0
        for k, v in self.state_action.items():
            if k[0] == observation and v[0] > best_score:
                action = k[1]
                best_score = v[0]

        if best_score > 0 and best_score/self.explore > t:
            return action
        else:
            return self.env.action_space.sample()

    def get_episode(self, ep_len):
        # (state,action,reward)
        episode = []
        observation = self.env.reset()
        observation = self.get_discrete_obs(observation)

        for _ in range(ep_len):
            action = self.get_action(observation)
            new_observation, reward, done, _ = env.step(action)
            new_observation = self.get_discrete_obs(new_observation)

            episode.append([observation, action, reward])
            observation = new_observation

            if done:
                break

        return episode

    def eval_episode(self, episode):
        episode.reverse()
        reward_so_far = 0

        for state, action, reward in episode:
            #         print(state,action,reward)
            reward_so_far = reward_so_far + reward

            if (state, action) in self.state_action:
                value, count = self.state_action[(state, action)]
                new_val = (value * count + reward_so_far) / (count + 1)
                self.state_action[(state, action)] = [new_val, count + 1]

            else:
                self.state_action[(state, action)] = [reward_so_far, 1]

    def train(self, no_of_eps, ep_len):
        avg_len = 0
        max_len = 100
        for i in range(1, no_of_eps):
            ep = None
            ep = self.get_episode(ep_len)
            self.eval_episode(ep)
            avg_len = avg_len + len(ep)

            if i % max_len == 0:
                print("Avg is", avg_len/max_len)
                avg_len = 0

    def run(self, no_of_runs):
        for _ in range(no_of_runs):
            observation = self.env.reset()
            while True:
                action = self.get_action(observation)
                new_observation, _, done, _ = self.env.step(action)
                new_observation = self.get_discrete_obs(new_observation)
                self.env.render()

                observation = new_observation

                if done:
                    break


env = gym.make('CartPole-v1')
mc = MonteCarloCartpole(env)
mc.train(1000, 502)

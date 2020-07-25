import gym


class Environment(object):
    def __init__(self, env_name):

        self.env = gym.make(env_name)

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        observation = self.env.reset()
        return observation

    def render(self):
        self.env.render()

    def step(self, action):
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, info = self.env.step(action)

        return observation, reward, done, info

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()

import numpy as np
import seaborn
import matplotlib.pyplot as plt


class Env():
    def __init__(self, num_server=10, p=0.06):
        self.request = [1, 2, 4, 8]
        self.num_server = num_server
        self.p = p
        self.num_left = num_server
        self.now_request = 0

    def reset(self):
        self.now_request = np.random.choice(self.request)
        return 0, self.num_left, self.now_request

    def step(self, action):
        # action 1: accept , action 0: reject
        if action == 1 and self.num_left:
            reward = self.now_request
            self.num_left = self.num_left - 1 + np.random.binomial(self.num_server-self.num_left, self.p)
            self.now_request = np.random.choice(self.request)
            return reward, self.num_left, self.now_request
        self.now_request = np.random.choice(self.request)
        self.num_left += np.random.binomial(self.num_server - self.num_left, self.p)
        return 0, self.num_left, self.now_request


class valuef():
    def __init__(self, learning_rate=0.01, reward_update_rate=0.01):
        self.lr = learning_rate
        self.rur = reward_update_rate
        self.w = np.zeros(200)
        self.l = []
        self.r_mean = 0

    def value(self, state):
        if state not in self.l:
            self.l.append(state)
        return self.w[self.l.index(state)]

    def learn(self, now_state, next_state, reward):
        delta = reward - self.r_mean + self.value(next_state) - self.value(now_state)
        self.r_mean += self.rur * delta
        update_factor = self.lr * delta
        self.w[self.l.index(now_state)] += update_factor


def make_action(valuef, num_left, request, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1])
    value = []
    for i in range(2):
        value.append(valuef.value([num_left, request, i]))
    return value.index(max(value))


def print_policy(valuef):

    policy = np.zeros((4, 11))
    requests = [1, 2, 4, 8]
    for index, request in enumerate(requests):
        for servers in range(1, 11):
            policy[index][servers] = make_action(valuef, servers, request, epsilon=0)

    fig = seaborn.heatmap(policy, cmap="YlGnBu", xticklabels=range(1, 11), yticklabels=requests)
    fig.set_title('Policy (0 Reject, 1 Accept)')
    fig.set_xlabel('Number of free servers')
    fig.set_ylabel('Priority')
    plt.show()


T = 1000000
t = 0
env = Env()
q = valuef()
_, num_left, request = env.reset()
action = make_action(q, num_left, request)

while t < T:
    t += 1
    reward, next_num, next_request = env.step(action)
    next_action = make_action(q, next_num, next_request)
    q.learn([num_left, request, action], [next_num, next_request, next_action], reward)
    action = next_action
    num_left = next_num
    request = next_request

print_policy(q)

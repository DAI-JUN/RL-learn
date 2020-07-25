import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def prepro(x):
    o = torch.tensor(x[35:195]).permute(2, 0, 1)
    output = o[0]*0.2126 + o[1]*0.7152 + o[2]*0.0722
    return output


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 3, 5, padding=2),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(3, 6, 5, padding=2),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            Flatten(),
            nn.Linear(80 * 80 * 6, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, state):
        state = state.view(1, 1, 160, 160).cuda()
        p = self.cnn(state)
        return p


class Agent_PG():
    def __init__(self, env):
        self.env = env
        self.p, self.l, self.r = [], [], []
        self.net = net()
        self.net.to('cuda')
        self.opt = torch.optim.Adam(self.net.parameters())
        print('loading trained model')
        load = True
        if load:
            state_dic = torch.load('model.dat')
            self.net.load_state_dict(state_dic['net'])
            self.opt.load_state_dict(state_dic['opt'])

    def init_game_setting(self):
        pass

    def store(self, p, l, r):
        self.p.append(p)
        self.l.append(l)
        self.r.append(r)

    def train(self):
        for episode in range(1, 1000):
            pre_state = None
            state = prepro(self.env.reset())
            while True:
                self.env.render()
                s_delta = pre_state - state if pre_state is not None else torch.zeros(160, 160)
                pre_state = state
                p = self.make_action(s_delta)
                action = 2 if p > np.random.uniform() else 3
                state, reward, done, info = self.env.step(action)
                state = prepro(state)
                self.store(p, 0 if action == 3 else 1, reward)
                if done:

                    discount_factor = 0.99
                    r = []
                    dis_r = 0
                    for i in list(reversed(self.r)):
                        dis_r = dis_r + i
                        r.append(dis_r)
                        dis_r = dis_r * discount_factor
                    r = torch.tensor(list(reversed(r)), device='cuda')
                    r = (r - r.mean()) / r.std()
                    p_list = torch.cat(tuple(self.p), 0).view(-1).cuda()

                    loss = F.binary_cross_entropy(p_list, torch.tensor(self.l, dtype=torch.float32, device='cuda'), weight=r)
                    print(loss.item())
                    loss.backward()
                    self.p.clear()
                    self.l.clear()
                    self.r.clear()
                    break
            if episode % 1 == 0:
                print('episode = ' + str(episode))
                self.opt.step()
                self.opt.zero_grad()
                state_dic = {
                    'net': self.net.state_dict(),
                    'opt': self.opt.state_dict()
                }
                torch.save(state_dic, 'model.dat')

    def make_action(self, observation):
        return self.net(observation)


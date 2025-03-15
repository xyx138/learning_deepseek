import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import time
import gym
import numpy as np

learning_rate = 0.002
gamma = 0.8
episode = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
in_features, hidden_features, out_features = 4, 128, 2


# 策略函数
class PGPolicy(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(PGPolicy, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(p=0.6)

        # 记录对数概率和奖励
        self.save_log_probs = []
        self.rewards = []
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.softmax(x, dim=-1)
        return out


def choose_action(state, policy):
    # 选择动作
    # state = state.reshape(1, 4)
    state = torch.from_numpy(state).float().unsqueeze(0).to(device) # 在索引0对应位置增加一个维度

    probs = policy(state)
    m = Categorical(probs)
    # 按照m抽样
    action = m.sample()
    policy.save_log_probs.append(m.log_prob(action))
    return action.item()


def learn(optimizer, policy):
    # 计算回报
    R = 0
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R) # 前插


    # 归一化处理
    eps = np.finfo(np.float64).eps.item()        
    returns = torch.tensor(returns).to(device)
    returns = (returns - returns.mean()) / (returns.std() + eps)


    # 计算回报对数概率的加权和
    policy_loss = []
    for log_prob, R in zip(policy.save_log_probs, returns):
        policy_loss.append(-log_prob * R)

    # 优化参数
    optimizer.zero_grad()
    # policy_loss = torch.tensor(policy_loss, requires_grad=True).sum()
    policy_loss = torch.cat(policy_loss).sum()

    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.save_log_probs[::]

def train(episodes_num):
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)

    policy = PGPolicy(in_features, hidden_features, out_features).to(device)

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    avg_reward = 0

    for i in range(episodes_num):
        state = env.reset()
        # state = torch.tensor(state).to(device).float()
        total_reward = 0
        for t in range(10000):
            action = choose_action(state, policy)   
            state, reward, done, _ = env.step(action)
            # state = torch.tensor(state).to(device).float()
            policy.rewards.append(reward) # 记录该时间步奖励
            total_reward += reward
            if done:
                break
        # 优化参数
        avg_reward = 0.05 * total_reward + (1 - 0.05) * avg_reward
        learn(optimizer, policy)
        if i % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i, total_reward, avg_reward))

    torch.save(policy.state_dict(), 'PGPolicy_gpu.pt')

def test():
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)
    policy = PGPolicy(in_features, hidden_features, out_features).to(device)
    policy.load_state_dict(torch.load('PGPolicy_gpu.pt'))

    with torch.no_grad():
        state = env.reset()
        for i in range(10000):
            # state = torch.tensor(state).to(device)
            action = choose_action(state, policy)
            state, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.1)
            
            if done:
                print(f"done is true, {i}")
                break
# 
train(episode)
# test()





            
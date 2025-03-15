import torch
import gym
import numpy as np
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time

lr = 0.002
gamma = 0.8

class PGPolicy(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, output_size=2): #输入状态，输出动作
        super(PGPolicy, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.6)
        
        self.saved_log_probs = []# 记录每一步的动作概率的对数
        self.rewards = []#记录每一步的r
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = F.softmax(x, dim=1)
        return out

def choose_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0) # 在索引0对应位置增加一个维度
    probs = policy(state) 
    m = Categorical(probs) #创建以参数probs为标准的类别分布,之后的m.sampe就会按此概率选择动作
    action = m.sample() # 按照概率分布m选择动作
    policy.saved_log_probs.append(m.log_prob(action)) # 保存当前动作的对数概率
    return action.item()#返回的就是int 即当前选择的动作

def learn(policy, optimizer):
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma*R # 递归计算每一步的回报，每一步的回报等于从这一步开始到未来所有步的加权奖励和
        returns.insert(0,R)#从头部插入，即反着插入
    returns = torch.tensor(returns)
    # 归一化（均值方差），eps是一个非常小的数，避免除数为0
    eps = np.finfo(np.float64).eps.item()
    returns = (returns - returns.mean()) / (returns.std() + eps)  
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob*R)

    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]  # 清空数据
    del policy.saved_log_probs[:]

def train(episode_num):
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)
    policy = PGPolicy()
    # policy.load_state_dict(torch.load('save_model.pt'))  # 模型导入
    optimizer = optim.Adam(policy.parameters(), lr)
    average_r = 0

    for i in range(1, episode_num+1): #采这么多轨迹
        obs = env.reset() # 重置环境
        ep_r = 0 # 当前局的总回报
        for t in range(1, 10000): # 采集10000步
            action = choose_action(obs, policy)
            obs, reward, done, _ = env.step(action)
            policy.rewards.append(reward) 
            ep_r += reward
            if done:
                break
        average_r = 0.05 * ep_r + (1-0.05) * average_r # 计算平均回报
        learn(policy, optimizer)
        if i % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(i, ep_r, average_r))

    torch.save(policy.state_dict(), 'PGPolicy.pt')


def test():
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)
    policy = PGPolicy()
    policy.load_state_dict(torch.load('PGPolicy.pt'))  # 模型导入
    average_r = 0
    with torch.no_grad():
        obs = env.reset()
        ep_r = 0
        for t in range(1, 10000):
            action = choose_action(obs, policy)
            obs, reward, done, _ = env.step(action) # 动作喂给环境，环境状态改变，同时给出奖励
            policy.rewards.append(reward)
            env.render() # 显示画面
            time.sleep(0.1)
            ep_r += reward # 计算总奖励
            if done:
                print(f'done is true, {t}')
                break

# train(1000)

test()

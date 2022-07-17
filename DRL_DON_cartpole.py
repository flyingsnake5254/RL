import random

import numpy as np
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'nextState', 'reward'))

ENV = 'CartPole-v1'
GAMMA = 0.9
MAX_STEP = 250
MAX_EPISODE = 500

# 儲存經驗 ( Transition ) 的記憶體類別
#     方法：
        # 1. 存入 Trainsition
        # 2. 根據 batch size 取出 Trainstions
        # 3. 回傳目前 memory length
class ReplayMemory:

    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0 # 儲存資料的 index

    def push(self, state, action, nextState, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, nextState, reward)
        self.index = (self.index + 1) % self.capacity

        # 根據 batchSize 取出 Transition
    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)

# 定義 Brain
#     方法：
#         replay - 取出批次資料
#         getAction - 根據 eta-greedy 法，回傳 Q value 最大的 action
#     過程：
#         取出資料後，進行權重訓練
#         訓練完畢後，更新 Q 函數
#         最後利用 Q 決定 action

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

BATCH_SIZE = 32
CAPACITY = 10000

class Brain:
    def __init__(self, numState, numAction):
        self.numState = numState
        self.numAction = numAction

        # 建立記憶經驗的物件
        self.memory = ReplayMemory(CAPACITY)

        # 建立神經網路
        self.model = nn.Sequential(
            nn.Linear(numState, 32)
            , nn.ReLU()
            , nn.Linear(32, 32)
            , nn.ReLU()
            , nn.Linear(32, numAction)
        )

        # 設定最佳化方法
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)


    # 利用 Experience Replay 學習權重
    def replay(self):
        # 確認 memory 裡的資料量，如果小於指定的 Batch Size，就啥都不做，直到大小為 batchsize
        if len(self.memory) < BATCH_SIZE:
            return

        # 取得批次資料
        transitions = self.memory.sample(BATCH_SIZE)

        # 轉換 transitions
        # 原本的 transitions : Transition(state, action, nextState, reward) X BATCH_SIZE
        # 轉換成：Transition(state x BATCH_SIZE, action x BATCH_SIZE, nextState x BATCH_SIZE, reward x BATCH_SIZE)
        batchData = Transition(*zip(*transitions))

        # 轉換 transitions 內的元素
        # 舉例：
        #     以 state (cartPos, cartV, poleAngle, poleV) 為例，
        #     原本的資料大小：BATCH_SIZE x state(Tensor) -> 在 transition 內，有 BATCH_SIZE 個 state(Tensor)
        #     現在轉換成一個 tensor，裡面包含(BATCH_SIZE x 4)的資料 // 4 為 state 內 cartPos, cartV...等四個元素
        stateBatch = torch.cat(batchData.state)
        actionBatch = torch.cat(batchData.action)

            # 只有具備下個狀態的資料才可以被加入
        nextStateBatch = torch.cat([s for s in batchData.nextState if s is not None])
        rewardBatch = torch.cat(batchData.reward)

        # 為了計算 Q(s, a)的數值，先將模式轉推論狀態
        self.model.eval()

        # 計算 Q(s, a)：
        # model(stateBatch) 會輸出此批次資料中，各 state 的 action value ( 包含左、右 )
        # ex :
        # model 輸出的值將如下
        # batch 1 : (0.1, 0.24) // 0.1 為向左的 action Q值，0.24為向右的 Q值
        # batch 2 : (0.33, 0.98)
        # batch 3 : (0.22, 0.52)
        # ...
        #
        # 接下來要取出每個 batch 對應的 action 的 Q值
        # ex :
        # batchAction 資料如下：
        # batch 1 : 0
        # batch 2 : 1
        # batch 3 : 1
        # ...
        #
        # 擇要取出的 Q值 為 : 0.1、0.98、0.52


        stateActionValue = self.model(stateBatch).gather(1, actionBatch)

        # 計算 maxQ(s+1, a)
        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s is not None, batchData.nextState)))
            # 初始化
        nextStateValue = torch.zeros(BATCH_SIZE)
        nextStateValue[nonFinalMask] = self.model(nextStateBatch).max(1)[0].detach()

        expStateActionValue = rewardBatch + GAMMA * nextStateValue

        self.model.train()

        print('stav:\n',stateActionValue.grad_fn)
        loss = F.smooth_l1_loss(stateActionValue, expStateActionValue.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def getAction(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))
        if epsilon <= np.random.uniform(0,1):
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)

        else:
            action = torch.LongTensor([[random.randrange(self.numAction)]])
        return action


class Agent:
    def __init__(self, numState, numAction):
        self.brain = Brain(numState, numAction)

    def updateQ(self):
        self.brain.replay()

    def getAction(self, state, episode):
        action = self.brain.getAction(state, episode)
        return action

    def memorize(self, state, action, nextState, reward):
        self.brain.memory.push(state, action, nextState, reward)

import gym
class Environment:
    def __init__(self):
        self.env = gym.make(ENV)
        numState = self.env.observation_space.shape[0]
        numAction = self.env.action_space.n
        self.agent = Agent(numState, numAction)

    def run(self):
        episode_10_list = np.zeros(10)
        completeEpisode = 0
        episodeFinal = False

        for episode in range(MAX_EPISODE):
            observation = self.env.reset()
            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)
            for step in range(MAX_STEP):
                action = self.agent.getAction(state, episode)
                observationNext, _, done, _ = self.env.step(action.item())

                if done:
                    stateNext = None
                    episode_10_list = np.hstack((episode_10_list[1:], step + 1))
                    if step < 195:
                        reward = torch.FloatTensor([-1.0])
                        completeEpisode = 0
                    else:
                        reward = torch.FloatTensor([1.0])
                        completeEpisode += 1
                else:
                    reward = torch.FloatTensor([0.0])
                    stateNext = observationNext
                    stateNext = torch.from_numpy(stateNext).type(torch.FloatTensor)
                    stateNext = torch.unsqueeze(stateNext, 0)

                self.agent.memorize(state, action, stateNext, reward)
                self.agent.updateQ()
                state = stateNext
                if done:
                    print("{0} Episode: Finished after {1} steps: 10 回合的平均 step = {2}".format(episode, step + 1, episode_10_list.mean()))
                    break
                if completeEpisode >= 10:
                    print("連續10次成功")
                    episodeFinal = True

cartPoleEnv = Environment()
cartPoleEnv.run()

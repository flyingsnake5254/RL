import random
import gym
import torch
from torch import nn
from torch import optim
from collections import namedtuple
import torch.nn.functional as F

NAME = 'CartPole-v1'
MAX_EPISODE = 1000 # 執行回合數
MAX_STEP = 250
GAMMA = 0.9
Data = namedtuple('Data', ('state', 'action', 'nextState', 'reward'))
BATCH_SIZE = 64
MEM_SIZE = 10000
memory = [[],[],[],[]] # state, action, nextState, reward
index = 0

def buildModel(inputNum, outputNum):
    model = nn.Sequential(
        nn.Linear(inputNum, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, outputNum)
    )



    return model

def zip(data):
    global stateNum
    dataSize = len(data)
    pkg = []
    for i in range(stateNum):
        l = []
        pkg.append(l)

    for i in data:
        for j in range(stateNum):
            pkg[j].append(i[j])

    return pkg

def getAction(state, episode):
    global model
    epsilon = 0.5 * (1 / (episode + 1))
    if random.random() < epsilon:
        return random.randint(0,1)
    else:
        model.eval()
        result = model(state)
        if result[0] > result[1]:
            return 0
        else:
            return 1

def addDataToMem(state, action, nextState, reward):
    global memory
    global MEM_SIZE
    global index
    if len(memory[0]) < MEM_SIZE:
        memory[0].append(None)
        memory[1].append(None)
        memory[2].append(None)
        memory[3].append(None)
    memory[0][index] = state
    memory[1][index] = action
    memory[2][index] = nextState
    memory[3][index] = reward

    index = (index + 1) % MEM_SIZE

def calExpStateActionValue(rewards, GAMMA, nextStateActionValue):

    expStateActionValue = []


    for i in range(len(rewards)):
        expStateActionValue.append([rewards[i] + GAMMA * nextStateActionValue[i]])


    return expStateActionValue

    # expStateActionValue = rewards + GAMMA * nextStateActionValue

def unsqueeze_(data):
    newData = []
    for i in data:
        newData.append([i])
    return newData

# 建立環境
env = gym.make(NAME)
stateNum = env.observation_space.shape[0] # 取得 state 數量
actionNum = env.action_space.n # 取得 action 數量

# 建立模型
model = buildModel(stateNum, actionNum)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
epData = []
for episode in range(MAX_EPISODE):
    # 一回合 = 一次遊戲
    # 環境初始化
    state = env.reset()
    for step in range(MAX_STEP):
        # 先將 state 轉為 Tensor，才能透過 model 算出 action
        stateTensor = torch.FloatTensor(state)
        action = getAction(stateTensor, episode)
        nextState, _, done, _ = env.step(action)

        if done:
            nextState = [None]
            if step < 180:
                reward = -1.0
            else:
                reward = 1.0
        else:
            reward = 0.0

        addDataToMem(state, [action], nextState, reward)

        
        if len(memory[0]) >= BATCH_SIZE:
            # 隨機選出要訓練的資料 index
            trainDataIndex = random.sample(range(0, len(memory[0])), BATCH_SIZE)
            trainState = []
            actions = []
            nextStates = []
            rewards = []

            for k in trainDataIndex:
                trainState.append(memory[0][k])
                actions.append(memory[1][k])
                nextStates.append(memory[2][k])
                rewards.append(memory[3][k])



            # 將挑出來的 states 轉成 Tensor
            stateTensor = torch.Tensor(trainState)
            actions = torch.LongTensor(actions)

            # 會算出 a = 0 / a = 1 時的 value，所以要根據 action，留下對應的 action value
                    # 這裡一定要使用 gather 篩選，原因跟 pytorch autograd 有關，但目前不清楚
            stateActionValueTensor = model(stateTensor).gather(1, actions)

                # 先將資料轉 list 再篩選

            # 計算期望的 stateActionValue
            #     expStateActionValue = reward + gamma * nextStateActionValue
            #     nextStateActionValue 是將 nextState 丟進 model ，並取結果兩個數值當中，較大的那個當 value
            #       %% 計算全部都用 list 進行，最後再將結果轉成 tensor
                # 找出 None 資料所在的 index
            noneDataIndex = []
            for ii in range(len(nextStates)):
                if len(nextStates[ii]) == 1:
                    noneDataIndex.append(ii)

                # 刪除 None 資料
            nextStates2 = []
            for ii in nextStates:
                if len(ii) != 1:
                    nextStates2.append(ii)

            nextStateTensor = torch.FloatTensor(nextStates2)

            # 若 nextState 為 None，則無法丟進 model，所以先將 nextState = None 的資料先提出來，
            # 剩下丟進 model 計算
            # 最後將那些被提出來的資料，以 maxQ = 0 放回去
            nextStateActionValueTensor = model(nextStateTensor)


                # 先將資料轉 list，在把之前 nextState = None 的資料，以 value = 0 放回
            nextStateActionValue = []
            for ii in range(len(nextStates2)):
                nextStateActionValue.append(max(nextStateActionValueTensor[ii]))

            for ii in noneDataIndex:
                nextStateActionValue.insert(ii, 0)



            expStateActionValue = calExpStateActionValue(rewards, GAMMA, nextStateActionValue)
            expStateActionValueTensor = torch.FloatTensor(expStateActionValue)

            # 計算誤差
            model.train()

            loss = F.smooth_l1_loss(stateActionValueTensor, expStateActionValueTensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            epData.append(step + 1)
            if len(epData) == 10:
                print('連續10回平均 step : {0}'.format(sum(epData) // 10))
                epData = []
            #print("episode {0} : step {1}".format(episode, step + 1))
            break
        else:
            state = nextState






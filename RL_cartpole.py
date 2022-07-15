import random
import numpy as np
import gym


NAME = 'CartPole-v1'
CUT_SIZE = 6
ETA = 0.5
GAMMA = 0.9
MAX_STEP = 250
epsilon = 0.5
episode = 10000


env = gym.make(NAME)
obs = env.reset()
ACTION_NUM = env.action_space.n
print(obs)

def cut(minValue, maxValue, cutNum, target):
    countDelta = cutNum -  2
    delta = abs(maxValue - minValue) / countDelta
    cmp = minValue
    for i in range(cutNum - 1):
        if target < cmp:
            return i
        cmp = cmp + delta

    return cutNum - 1

def buildQ(cutSize, stateSize, actionSize):
    Q = np.random.rand(cutSize**stateSize , actionSize)
    return Q



def getStateMaxQAction(Q,state):
    return np.argmax(Q[state,:]), np.max(Q[state,:])

def getState(observation):
    cartPos , cartV , poleAngle, poleV = observation
    cartPos = cut(-2.4,2.4,CUT_SIZE,cartPos)
    cartV = cut(-3.0,3.0,CUT_SIZE,cartV)
    poleAngle = cut(-0.5,0.5,CUT_SIZE,poleAngle)
    poleV = cut(-2.0,2.0,CUT_SIZE,poleV)

    l = [cartPos, cartV, poleAngle, poleV]
    count = 0
    for i in range(4):
        count += l[i] * (CUT_SIZE ** i)
    return count

def getAction(Q, state, epsilon):

    if np.random.rand() < epsilon:
        return random.randint(0,1)
    else:
        action, p = getStateMaxQAction(Q,state)
        return action

def updateQ(Q, state, action, nextState, reward):
    global ETA
    global GAMMA
    newQ = Q.copy()
    _, maxQ = getStateMaxQAction(Q, nextState)
    newQ[state, action] = Q[state, action] + ETA * (reward + GAMMA * max(Q[nextState][:]) - Q[state, action])
    return newQ


Q = buildQ(CUT_SIZE, 4, 2)
env = gym.make(NAME)
countSuccess = 0
for e in range(episode):
    obs = env.reset()

    for step in range(MAX_STEP):
        reward = 0
        state = getState(obs)
        action = getAction(Q, state, epsilon)
        nextObs, _, done, _ = env.step(action)
        nextState = getState(nextObs)

        if done:
            if step < 180:
                reward = -1
                countSuccess = 0
            else:
                reward = 1
                countSuccess += 1
        else:
            reward = 0

        Q = updateQ(Q, state, action, nextState, reward)
        obs = nextObs

        if done:
            print("第{0}次：finish {1} times.".format(e + 1 , step + 1))
            break

    if countSuccess > 10:
        print("訓練完成")
        break

    epsilon /= 2


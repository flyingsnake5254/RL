import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.random


fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)



def makeQ():
    global theta0
    x , y = theta0.shape
    Q = np.random.rand(x , y) * theta0
    return Q

def makePi():
    global theta0
    x , y = theta0.shape
    pi = np.zeros((x , y))
    for i in range(x):
        pi[i,:] = theta0[i,:] / np.nansum(theta0[i,:])
    pi = np.nan_to_num(pi)
    return pi

def getAction(s , Q , pi , epsilon):
    # 0 : up  , 1 : down , 2 : left , 3 : right
    directions = [0 , 1 , 2 , 3]

    # random action
    if np.random.rand() < epsilon:
        direction = np.random.choice(directions , p = pi[s,:])
    else:
        direction = np.nanargmax(Q[s,:])

    return direction

def getNextState(s , a):
    if a == 0: # up
        return s - 3
    elif a == 1: # down
        return s + 3
    elif a == 2: # left
        return s - 1
    elif a == 3: # right
        return s + 1

def QLearning(s , a , nextS  , Q , gamma , eta , r):
    newQ = Q.copy()
    if nextS == 8:
        newQ[s , a] = Q[s , a] + eta * (r - Q[s , a])
    else:
        newQ[s , a] = Q[s , a] + eta * (r + gamma * np.nanmax(Q[nextS,:]) - Q[s , a])
    return newQ

def init():
    global line
    line.set_data([], [])
    return line,


def animate(i):
    global results
    s = results[i][0]
    x = (s % 3) + 0.5
    y = 2.5 - int(s / 3)
    line.set_data(x, y)
    return (line,)

def makeFig():
    global fig
    global ax
    global result

    plt.plot([1, 1], [0, 1], color='red', linewidth=2)
    plt.plot([2, 2], [2, 1], color='red', linewidth=2)
    plt.plot([1, 2], [2, 2], color='red', linewidth=2)
    plt.plot([2, 3], [1, 1], color='red', linewidth=2)

    plt.text(0.5, 2.5, 'S0', size=14, ha='center')
    plt.text(1.5, 2.5, 'S1', size=14, ha='center')
    plt.text(2.5, 2.5, 'S2', size=14, ha='center')
    plt.text(0.5, 1.5, 'S3', size=14, ha='center')
    plt.text(1.5, 1.5, 'S4', size=14, ha='center')
    plt.text(2.5, 1.5, 'S5', size=14, ha='center')
    plt.text(0.5, 0.5, 'S6', size=14, ha='center')
    plt.text(1.5, 0.5, 'S7', size=14, ha='center')
    plt.text(2.5, 0.5, 'S8', size=14, ha='center')
    plt.text(0.5, 2.3, 'START', ha='center')
    plt.text(2.5, 0.3, 'GOAL', ha='center')

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off',
                    labelleft='off')

    # interval 參數可調整動畫速度
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(results), interval=200, repeat=False)

    plt.show()


def getCirclePos(s):
    p1 = 0.5
    p2 = 2.5
    if s == 0:
        p1 = 0.5
        p2 = 2.5
    elif s == 1:
        p1 = 2.5
        p2 = 2.5
    elif s == 2:
        p1 = 0.5
        p2 = 1.5
    elif s == 4:
        p1 = 1.5
        p2 = 1.5
    elif s == 5:
        p1 = 2.5
        p2 = 1.5
    elif s == 6:
        p1 = 0.5
        p2 = 0.5
    elif s == 7:
        p1 = 1.5
        p2 = 0.5
    elif s == 8:
        p1 = 2.5
        p2 = 0.5

    return p1, p2

theta0 = np.array([
        [np.nan, 1, np.nan, 1],
        [np.nan, np.nan, 1, 1],
        [np.nan, 1, 1, np.nan],
        [1, 1, np.nan, 1],
        [np.nan, 1, 1, np.nan],
        [1, np.nan, np.nan, np.nan],
        [1, np.nan, np.nan, np.nan],
        [1, np.nan, np.nan, 1]
    ])

# 建立行動價值函數 ( action value )
Q = makeQ()
pi = makePi()

epsilon = 0.5
gamma = 0.9 # 時間折扣率
eta = 0.1 # 學習率

results = []
count = 0

while count < 100:
    s = 0
    result = []
    while True:
        r = 0
        nextAction = getAction(s, Q, pi, epsilon)
        nextState = getNextState(s, nextAction)
        nextStateAction = np.nan
        mem = [s, nextAction]
        result.append(mem)

        if nextState == 8:
            r = 1
        else:
            r = 0


        Q = QLearning(s, nextAction , nextState , Q , eta , gamma , r)

        if nextState == 8:
            mem = [8 , nextStateAction]
            result.append(mem)
            break
        else:
            s = nextState
    results += result
    print("step : ", len(result) - 1)

    count += 1
    epsilon = epsilon / 2
makeFig()

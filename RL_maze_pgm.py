import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy.random
from IPython.display import HTML

fig = plt.figure(figsize=(5, 5))
ax = plt.gca()
line, = ax.plot([0.5], [2.5], marker="o", color='g', markersize=60)


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
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(results), interval=100, repeat=False)
    HTML(anim.to_jshtml())

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


def makePI(theta0):
    [m, n] = theta0.shape
    pi = np.zeros((m, n))

    expTheta = np.exp(theta0)

    for i in range(0, m):
        pi[i, :] = expTheta[i, :] / np.nansum(expTheta[i, :])
    pi = np.nan_to_num(pi)

    return pi


def nextS(pi, s):
    direction = ['up', 'down', 'left', 'right']
    nextDir = numpy.random.choice(direction, p=pi[s])

    a = np.nan
    if nextDir == 'up':
        a = 0
        s -= 3
    elif nextDir == 'down':
        a = 1
        s += 3
    elif nextDir == 'left':
        a = 2
        s -= 1
    elif nextDir == 'right':
        a = 3
        s += 1
    return a, s


def run(pi, s):
    mem = []
    while 1:
        a, nexts = nextS(pi, s)
        l = [s, a]
        mem.append(l)
        s = nexts

        if s == 8:
            l = [8, np.nan]
            mem.append(l)
            break
    return mem


def updateTheta(result, pi, theta):
    eta = 0.3
    thetaX, thetaY = theta.shape
    Nsa = np.zeros((thetaX, thetaY))
    Nsia = np.zeros((thetaX, thetaY))
    T = len(result) - 1
    for i in range(thetaX):
        for j in range(thetaY):
            count = 0
            countNsia = 0
            for data in result:
                if data[0] == i:
                    countNsia += 1
                    if data[1] == j:
                        count += 1

            Nsa[i][j] = count
            Nsia[i][j] = countNsia

    deltaTheta = (Nsa - pi * Nsia) / T
    newTheta = theta + eta * deltaTheta

    return newTheta


results = []
# 上、下、左、右
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
pi = makePI(theta0)
stop = 10 ** -4
while True:
    result = run(pi, 0)
    results += result
    print("步數：", len(result) - 1)
    # makeFig()
    theta0 = updateTheta(result, pi, theta0)
    newPi = makePI(theta0)
    if np.sum(np.abs(newPi - pi)) < stop:
        print(newPi)
        break
    else:
        pi = newPi
makeFig()




import matplotlib.pyplot as plt
import numpy as np


def print_state_action(state_action, n):
    g = []
    for i in range(n):
        t = []
        for j in range(n):
            t.append(-9999.0)
        g.append(t)

    for k, v in state_action.items():
        g[k[0][0]][k[0][1]] = max(g[k[0][0]][k[0][1]], v[0])

    for i in range(n):
        for j in range(n):
            print("%-3.2f " % (g[i][j]), end="")
        print()


def print_heat_map(state_action, n):
    g = []
    for i in range(n):
        t = []
        for j in range(n):
            t.append(-9999.0)
        g.append(t)

    for k, v in state_action.items():
        g[k[0][0]][k[0][1]] = max(g[k[0][0]][k[0][1]], v[0])

    for i in range(n):
        for j in range(n):
            if(g[i][j] == -9999.0):
                g[i][j] = 0

    a = np.array(g)
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.show()

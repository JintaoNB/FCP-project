import numpy as np
import matplotlib.pyplot as plt
import random
grid_length = 100
T = 0.5  # threshold
beta = 0.5  # coupling
timestep = 100

def initialize_opinions(size):
    opinions = np.random.random(size)
    return opinions


def change_opinions(opinions, T, beta, timestep):
    opinions_list = [opinions.copy()]
    for a in range(timestep):
        i = np.random.randint(grid_length)
        if i == 0:
            j = i + 1
        elif i == grid_length - 1:
            j = i - 1
        else:
            j = i + np.random.choice([-1, 1])
        # math the opinions
        if abs(opinions[i] - opinions[j]) < T:
            opinions[i] += beta * (opinions[j] - opinions[i])
            opinions[j] += beta * (opinions[i] - opinions[j])
        opinions_list.append(opinions.copy())
    return opinions_list


opinions_list = change_opinions(initialize_opinions(grid_length), T, beta, timestep)


plt.plot(opinions_list)
plt.xlabel('timestep')
plt.ylabel('opinion')
plt.grid(True)
plt.show()
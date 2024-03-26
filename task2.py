import numpy as np
import matplotlib.pyplot as plt
import random
# 初始化参数
grid_length = 100
T = 0.5  # threshold
beta = 0.5  # coupling
steps = 100

def initialize_opinions(size):
    return np.random.random(size)


def change_opinions(opinions, T, beta, steps):
    for i in range(steps):
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

    return opinions


final_opinions = change_opinions(opinions, T, beta, steps)

plt.figure(figsize=(10, 6))
plt.plot(final_opinions, 'o')
plt.title('Final Opinion Distribution')
plt.xlabel('Individual')
plt.ylabel('Opinion')
plt.grid(True)
plt.show()
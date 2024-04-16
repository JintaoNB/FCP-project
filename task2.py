import numpy as np
import matplotlib.pyplot as plt
import argparse

grid_length = 100
beta = 0.1  # coupling
T = 0.2 # threshold
timestep = 10000


# Function to initialize opinions randomly
def initialize_opinions(size):
    opinions = np.random.rand(size)
    return opinions


# Function to change opinions according to the model
def change_opinions(opinions, T, beta, timestep):
    opinions_list = []
    for a in range(timestep):
        i = np.random.randint(grid_length)
        if i == 0:
            j = i + 1
        elif i == grid_length - 1:
            j = i - 1
        else:
            j = i + np.random.choice([-1, 1])

        if abs(opinions[i] - opinions[j]) < T:
            opinions[i] += beta * (opinions[j] - opinions[i])
            opinions[j] += beta * (opinions[i] - opinions[j])

        opinions_list.append(opinions.copy())

    return opinions_list


# Initialize opinions
opinions = initialize_opinions(grid_length)

# Change opinions over time
opinions_list1 = change_opinions(opinions, T, beta, timestep)

# Plotting function
def plot_opinion_dynamics(opinions_list, grid_length, timestep, step=100):
    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(opinions_list[-1], bins=10, color='blue')
    ax1.set_xlabel('Opinion')
    ax1.set_ylabel('Number')
    ax1.set_title('Opinion Distribution at Last Time Step')
    ax1.set_xticks(np.arange(0, 1.1, 0.1))


    ax2 = fig.add_subplot(1, 2, 2)
    for t, opinions in enumerate(opinions_list):
        if t % step == 0:
            ax2.plot([t/step] * grid_length, opinions, 'r.', markersize=18, alpha=0.5)
    ax2.set_xticks(np.arange(0, timestep/step+1, 10))
    ax2.set_xticklabels([str(int(x)) for x in np.arange(0, timestep/step+1, 10)])
    ax2.set_xlabel('Time Step (in hundreds)')
    ax2.set_ylabel('Opinion')
    ax2.set_title('Opinion Dynamics Over Time')
    ax2.set_xlim([0, timestep/step])

    plt.tight_layout()
    plt.show()

def main():
    parser.add_argument()
    parser.add_argument()
    parser.add_argument()
    args = parser.parer_args()
    plot_opinion_dynamics(opinions_list1, grid_length, timestep, step=100)

if _name_ == "_main_"
    main()
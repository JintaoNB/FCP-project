import numpy as np
import matplotlib.pyplot as plt
import argparse

grid_length = 100 # 100 neighbours
timestep = 10000 # It is 10000 because 100 neighbour runs 100 times

# make random opinions
def initialize_opinions(size):
    return np.random.rand(size)

# calculate the opinions
def change_opinions(opinions, T, beta, timestep):
    opinions_list = []
    for a in range(timestep):
        # choose random neighbour
        i = np.random.randint(grid_length)
        # made them like a circle , first one could talk with last one
        j = (i + np.random.choice([-1, 1])) % grid_length
        # compare opinions and change
        if abs(opinions[i] - opinions[j]) < T:
            opinions[i] += beta * (opinions[j] - opinions[i])
            opinions[j] += beta * (opinions[i] - opinions[j])
        opinions_list.append(opinions.copy())
    return opinions_list

# make pragh
def plot_opinion_dynamics(opinions_list, grid_length, timestep, step=100):
    # the 1st graph
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(opinions_list[-1], bins=10, color='blue')
    ax1.set_xlabel('Opinion')
    ax1.set_ylabel('Number')
    ax1.set_title('Opinion Distribution at Last Time Step')
    ax1.set_xticks(np.arange(0, 1.1, 0.1))

    # the 2nd graph
    ax2 = fig.add_subplot(1, 2, 2)
    for t, opinions in enumerate(opinions_list):
        # made 10000 steps shows 100 step in xlabel
        if t % step == 0:
            ax2.plot([t / step] * grid_length, opinions, 'r.', markersize=18, alpha=0.5)
    ax2.set_xticks(np.arange(0, timestep / step + 1, 10))
    ax2.set_xticklabels([str(int(x)) for x in np.arange(0, timestep / step + 1, 10)])
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Opinion')
    ax2.set_title('Opinion Dynamics Over Time')
    ax2.set_xlim([0, timestep / step])
    plt.tight_layout()
    plt.show()
# the total function for last 3 functions
def defuant_main(T, beta):
    opinions = initialize_opinions(grid_length)
    opinions_list = change_opinions(opinions, T, beta, timestep)
    plot_opinion_dynamics(opinions_list, grid_length, timestep, step=100)

# compare with 4 graphs in paper
def test_default():
    defuant_main(0.5,0.5)
    defuant_main(0.1,0.5)
    defuant_main(0.5,0.1)
    defuant_main(0.1,0.2)

# how to use this in terminal
def main():
    parser = argparse.ArgumentParser(description='Run the Defuault model with defuault parameters.')
    parser.add_argument('-defuant', action='store_true', help='Run defuault model.')
    parser.add_argument('-beta', default = 0.2, type=float, help='Coupling beta.')
    parser.add_argument('-threshold', default = 0.2, type=float, help='Threshold T.')
    parser.add_argument('-test_defuant', action='store_true', help='Run test model.')
    args = parser.parse_args()
    if args.defuant:
        defuant_main(args.beta, args.threshold)
    elif args.test_defuant:
        test_default()

if __name__ == '__main__':
    main()

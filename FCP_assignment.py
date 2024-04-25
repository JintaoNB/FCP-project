import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import argparse
import sys


class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

        def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def get_mean_degree(self):
		#Your code  for task 3 goes here

	def get_mean_clustering(self):
		#Your code for task 3 goes here

	def get_mean_path_length(self):
		#Your code for task 3 goes here

	def make_random_network(self, N, connection_probability):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
	    self.nodes = []
            for node_number in range(N):
                value = np.random.random()
                connections = [0 for _ in range(N)]
                self.nodes.append(Node(value, node_number, connections))

		
	    for (index, node) in enumerate(self.nodes):
                for offset in range(-neighbour_range, neighbour_range + 1):
                    if offset != 0:  # Skip connecting a node to itself
                        neighbour_index = (index + offset) % N
                        node.connections[neighbour_index] = 1
                        self.nodes[neighbour_index].connections[index] = 1

	def make_small_world_network(self, N, re_wire_prob=0.2):
		
	    self.make_ring_network(N, neighbour_range=2)

            for node_index in range(len(self.nodes)):
                node = self.nodes[node_index]
                connection_inds = [ind for ind in range(N) if node.connections[ind] == 1]
                for connection_ind in connection_inds:
                    if np.random.random() < re_wire_prob:
                        # this removes the connection
                        node.connections[connection_ind] = 0
                        self.nodes[connection_ind].connections[node_index] = 0

                        random_index = np.random.choice(
                            [idx for idx in range(N) if idx != node_index and idx not in connection_inds])
                        self.nodes[random_index].connections[node_index] = 1
                        node.connections[random_index] = 1

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_clustering()
	assert(network.get_path_length()==1), network.get_path_length()

	print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''



#This function gathers the opinions of the neighbours of one person. It checks the nighbour above, below, left and right of this person and loops round if this person is on the edge
def get_neighbours_opinions(population, i, j):
	n, m = population.shape
	neighbours = []
	neighbours.append(population[i-1,j])
	neighbours.append(population[(i+1)%n ,j])
	neighbours.append(population[i,j-1])
	neighbours.append(population[i,(j+1)%m])
	return neighbours


def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''
	#sets original agreement value to 0 and initialises the 'person' randomly picked as a value according to the row and column placement
	agreement = 0
	value = population[row, col]
	# makes a list of neighbours and ignores the neighbours that are out of bounds
	neighbour_ops = get_neighbours_opinions(population, row, col)
	# loops over these neighbours and adds value onto the agreement from equation
	for opinion in neighbour_ops:	
		agreement += value *opinion  
	agreement +=  external * value	
	#return np.random * population
	return agreement



def ising_step(population, external=0.0, alpha = 1):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)
	agreement = calculate_agreement(population, row, col, external=0.0)

	#using probability part

	if agreement < 0:
		#flips if disagreement
		population[row, col] *= -1
		#uses probability, if random number is less than p then flip back to original opinion, this is the stubborness part 
	elif alpha:
		random_number = random.random()
		e = math.e	
		p = e ** (-agreement/alpha)
		if random_number < p:
			population[row, col] *= -1


def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''

	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.01)
	

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''
	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1)==4), "Test 1"

	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"

	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"

	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"

	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"

	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
	assert(calculate_agreement(population,1,1,-10)==14), "Test 10"
	print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
	for step in range(1000):
		ising_step(population, external, alpha)
	plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

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
'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
	parser = argparse.ArgumentParser()

	#adding the flags using argparse
	parser.add_argument("-ising_model", action ='store_true')
	parser.add_argument("-external", type = float, default = 0)
	parser.add_argument("-alpha", type = float, default = 1)
	parser.add_argument("-test_ising", action ='store_true')

	#this will define the variables
	args = parser.parse_args()
	alpha = args.alpha
	external = args.external 

	if args.ising_model:
		pop = np.random.choice([-1,1],size=(100,100))
		ising_main(pop, alpha, external)
	if args.test_ising:
		test_ising()


	

if __name__=="__main__":
	main()

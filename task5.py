import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import argparse
import math



def average_opinion(network):
    """
    Calculate the mean opinion of all the nodes in the network at each step.
    """
    total = 0
    opinions = [node.value for node in network.nodes]
    for opinion in opinions:
        total += opinion
    mean_opinion = total/len(opinions)

    return mean_opinion





def network_calculate_agreement(network, node_number, external):
    '''
    This calculates the agreement of each node with the indexed node
    '''
    primary_node = network.nodes[node_number]

    #sets the agreement value to 0 and creates an empty list of neighbours opinions 
    agreement = 0
    neighbour_opinions = []

    # Finds the neigbouring nodes
    neighbours = primary_node.get_neighbours()
    for node in neighbours:
        node = network.nodes[node]
        neighbour_opinions.append(node.value)

    #calculates the agreement between the indexed node and its neighbours
    for opinion in neighbour_opinions:
        #increases agreement if opinions are the same but will decrease if they oppose
        agreement += node.value * opinion

    #increases agreement using the external pull value
    agreement += external * node.value

    return agreement  


def ising_network_iteration(network, external=0.0, alpha=1):
    '''
    This function causes a single update of the Ising model

    '''
    # Picks a random node
    node_number = np.random.randint(0, len(network.nodes))
    node = network.nodes[node_number]
    
    # Calls the calculate agreement function and sets this as the agreement variable
    agreement = network_calculate_agreement(network, node_number, external)
    
    # if the agreement is less than zero the node value will flip
    if agreement < 0:
        node.value *= -1
    
    # runs a random chance that the node's opinion will flip 
    elif alpha:
        # produces a random number between 0 and 1
        random_number = random.random()
        # calculates the probability that this nodes opinion will flip back to its original value
        p = math.e**(-agreement/alpha)
        #checking if random number is less than p
        if random_number < p:
            #changes the opinion
            node.value *= -1




def ising_network_opinions(ising_network_size, alpha=None, external=0):
    '''
    This function is the main plot for the ising model based on networks:
    
    '''

    network = Network()
    average_opinions = []
    steps = np.arange(1,11,1)
    small_world_network = network.make_small_world_network(ising_network_size)
    # set a random value for opinions of each node, either -1 or 1
    for node in small_world_network.nodes:
        node.value = random.uniform(-1, 1)
    
    # This updates the network for a set amount of times
    for frame in range(len(steps)):
        # Iterating a single step 1000 times to form one update
        for step in range(1000):
            #This calls the iteration function
            ising_network_iteration(network, external, alpha)
        #plotting each one the frames
        average_opinions.append(average_opinion(network))
        network.plot()
        plt.show()
        #plots the mean opinion of the network over time
    plt.plot(steps, average_opinions)
    plt.title('Mean opinions of the Network over time')
    plt.xlabel('Iterations')
    plt.ylabel('Mean Opinion of the Network')
    plt.show()



def main():
	parser = argparse.ArgumentParser()

	#adding the flags using argparse for exercise 1
	parser.add_argument("-ising_model", action ='store_true')
	parser.add_argument("-external", type = float, default = 0)
	parser.add_argument("-alpha", type = float, default = 1)


	#ex 5 flags
	parser.add_argument("-use_network", type=int, help="Size of the network for ex 5")	

	#this will define the variables
	args = parser.parse_args()

	alpha = args.alpha
	external = args.external 
	re_wire_prob = args.re_wire


	#ex 1
	if args.ising_model and args.use_network:
			ising_network(ising_network_size, alpha, external)
	
    #ex 4
	if args.ring_network:
		network = Network()  # Create an instance of the Network class for ring network and plots the graph
		network.make_ring_network(args.ring_network, neighbour_range=1)
		network.plot()
		plt.show() 

	if args.small_world:
		network = Network() # Create an instance of the Network class for small world network and plots the graph
		network.make_small_world_network(args.small_world, re_wire_prob = args.re_wire)
		network.plot()
		plt.show()


if __name__ == "__main__":
	main()


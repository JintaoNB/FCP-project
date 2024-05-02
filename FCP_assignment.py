import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
import argparse
import math


class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

    def get_neighbours(self):
        return np.where(np.array(self.connections) == 1)[0]

class Network: 

    def __init__(self, nodes=None):

        if nodes is None: 
            self.nodes = []
        else:
            self.nodes = nodes 

    def add_node(self, node):
        self.nodes.append(node) 

    def get_mean_degree(self):
        total_degree = sum(sum(node.connections) for node in self.nodes)
        mean_degree = total_degree / len(self.nodes)
        return mean_degree

    def clustering_coefficient(self, node):
        neighbors = [self.nodes[i] for i, connected in enumerate(node.connections) if connected]
        num_possible_connections = len(neighbors) * (len(neighbors) - 1) / 2
        if num_possible_connections == 0:
            return 0.0
        num_actual_connections = 0
        for i, neighbor1 in enumerate(neighbors):
            for neighbor2 in neighbors[i + 1:]:
                if neighbor1.connections[neighbor2.index] == 1:
                    num_actual_connections += 1
        return num_actual_connections / num_possible_connections

    def get_clustering(self):
        mean_coefficient = sum(self.clustering_coefficient(node) for node in self.nodes) / len(self.nodes)
        return mean_coefficient


    def get_path_length(self):
        total_path_length = 0
        total_paths = 0
        for node in self.nodes:
            for other_node in self.nodes:
                if node != other_node:
                    path_length = self.bfs_path_length(node, other_node)
                    if path_length is not None:
                        total_path_length += path_length
                        total_paths += 1
        if total_paths == 0:
            return 0
        return round(total_path_length / total_paths, 15)
    '''
    def get_neighbors(self, node):
        return [self.nodes[i] for i, connected in enumerate(node.connections) if connected]
    '''
    def bfs_path_length(self, start_node, end_node):
        visited = set()
        queue = [(start_node, 0)]
        while queue:
            current_node, distance = queue.pop(0)
            if current_node == end_node:
                return distance
            visited.add(current_node)
            for neighbor in self.get_neighbors(current_node):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        return None


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
        # This function creates a ring network of nodes
        # This initialises an epty list for the nodes to be added to, the network is created using the same methods as the random network 
        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))
        # Iterates through each node in the list of nodes
        for (index, node) in enumerate(self.nodes):
            # Iterates through a range centred around the current nodes index
            for offset in range(-neighbour_range, neighbour_range + 1):
                if offset != 0:  # Skip connecting a node to itself
                    neighbour_index = (index + offset) % N # Calculates the index of the neighbouring node in the network
                    node.connections[neighbour_index] = 1 #  This creates a connection between the current node and its neighbour
                    self.nodes[neighbour_index].connections[index] = 1 # This creates a connection between the nieghbour and the current index

    def make_small_world_network(self, N, re_wire_prob=0.2):
        # This creates a ring network of size N and range 2
        self.make_ring_network(N, neighbour_range = 2)
        #This iterates over each node in the node list
        for node_index in range(len(self.nodes)):
            node = self.nodes[node_index]
            # This creates a list of indexes that the current node is connected to by looping through the list of connections and seperations
            connection_inds = [ind for ind in range(N) if node.connections[ind]==1]
            for connection_ind in connection_inds: # This iterates over each connection
                if np.random.random() < re_wire_prob: # Implements the re-wire probability  by choosing a random number between 0 and 1, if this number is less than prob, carry out behaviour
                    #this removes the connection
                    node.connections[connection_ind] = 0 # sets the current connection between the index and neighbour to 0
                    self.nodes[connection_ind].connections[node_index] = 0 # sets the current connection between the nighbour and index to 0
                    #This then chooses a random node and creates connection between both of them
                    random_index = np.random.choice([idx for idx in range(N) if idx != node_index and idx not in connection_inds]) 
                    self.nodes[random_index].connections[node_index] = 1
                    node.connections[random_index] = 1
        return self

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
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number - 1) % num_nodes] = 1
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert (network.get_mean_degree() == 2), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number + 1) % num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert (network.get_mean_degree() == 1), network.get_mean_degree()
    assert (network.get_clustering() == 0), network.get_clustering()
    assert (network.get_path_length() == 5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert (network.get_mean_degree() == num_nodes - 1), network.get_mean_degree()
    assert (network.get_clustering() == 1), network.get_clustering()
    assert (network.get_path_length() == 1), network.get_path_length()

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
    ax2.set_xlabel('Time Step (in hundreds)')
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
def test_defaunt():
    defuant_main(0.5,0.5)
    defuant_main(0.1,0.5)
    defuant_main(0.5,0.1)
    defuant_main(0.1,0.2)


#Task five

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



'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
    parser = argparse.ArgumentParser()

    #adding the flags using argparse for exercise 1
    parser.add_argument("-ising_model", action ='store_true')
    parser.add_argument("-external", type = float, default = 0)
    parser.add_argument("-alpha", type = float, default = 1)
    parser.add_argument("-test_ising", action ='store_true')

    #adding flags for ex 2
    parser.add_argument('-defuant', action='store_true', help='Run defuault model.')
    parser.add_argument('-beta', default = 0.2, type=float, help='Coupling beta.')
    parser.add_argument('-threshold', default = 0.2, type=float, help='Threshold T.')
    parser.add_argument('-test_defuant', action='store_true', help='Run test model.')

    #ex 5 flags
    parser.add_argument("-use_network", type=int, help="Size of the network for ex 5")

    #adding the flags for exercise 3
    parser.add_argument("-test_networks", action="store_true", help="Test networks")
    parser.add_argument("-network", "--network_size", type=int, help="Size of the network")


    #adding the flags using argparse for exercise 4
    parser.add_argument("-re_wire", type = float, default = 0.2)
    parser.add_argument("-ring_network", type = int, default = 0)
    parser.add_argument("-small_world", type = int, default = 0)
    

    #this will define the variables
    args = parser.parse_args()

    alpha = args.alpha
    external = args.external 
    re_wire_prob = args.re_wire
    ising_network_size = args.use_network

    #ex 1
    if args.ising_model:
        if args.use_network:
            ising_network_opinions(ising_network_size, alpha, external)
           
        else:
            pop = np.random.choice([-1,1],size=(100,100))
            ising_main(pop, alpha, external) # runs the ising model
    if args.test_ising:
        test_ising() # runs the tests for ex 1

    # ex 2
    if args.defuant:
        defuant_main(args.beta, args.threshold)
    if args.test_defuant:
        test_defaunt()
    
    # ex 3
    if args.test_networks:
        test_networks()
    if args.network_size:
        network = Network()
        network.make_random_network(args.network_size, 0.5)
        mean_degree = network.get_mean_degree()
        average_path_length = network.get_path_length()
        mean_clustering_coefficient = network.get_clustering()
        print(f"Mean degree: {mean_degree}")
        print(f"Average path length: {average_path_length}")
        print(f"Clustering coefficient: {mean_clustering_coefficient}")
    # Uncomment below to plot the network
    # network
    # ex 4
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



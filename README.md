Exercise 1: 

Firstly, I created a get_neighbours_opinions function. This treats the array like a sphere by using modulos to collect the neighbours opinions for people on the edge of the array. The calculate agreement function implements the equations from the question in order to find the value of agreement of each neighbour node pair. The ising_step function defines the population. It also tests if the agreement values are less than 0, if they are the opinion of that node flips. To account for the stubborness, a proportion of these flipped nodes will flip back. This is implemented using  random.random(), if this value is less than the probability (p = exp(-agreement/alpha), the node flips back to its original value. This model passes all  the tests and produces an animation. This model is used by using the argument –ising and the value of alpha can be changed using –alpha (N), where N is the value of alpha to be set. 


Exercise 2:

In task2, I created a function called initialize_opinions at first to make random opinions for every person in the grid. Then, I use  change_opinions function to random choose people and their neighbor, then compare opinions because big different of  them can not communicate with each other, and calculate the opinions. It could simulates their interactions over a specified number of time steps. After this I draw graph of them. In the test defuant part, I use 4 default numbers of  coupling and Threshold to make 4 original graph.


Exercise 4: 

Using part of the make_random_network given to us in the skeleton code, I created a random network. This has a list of nodes within the network. By iterating through the list of nodes and the node forms connections with neighbouring nodes in the network, this neighbouring node creates a connection back to the index node. These connections are formed within a range set. N is the amount of nodes in the system, this is passed through using argparse. To make the small world network, first a ring network with range 2 must e created. Then by iterating through the list of nodes, for each node a list of connections is produced. This list consists of indexes that correlate to the nodes this index node is attached to. For each of these connections, if a random number is less than the re wire probability, the connection is set to 0. This is the connection from the index node to the neighbouring node and from the neigbouring node to the index node. Then using the random.choice function within the range of number of nodes, if the number is not the index of the index node a connection will be made to this new random node. A connection from the new random node to the index node is also produced. By calling –ring_network  (number) or –small_world (number), a network with number amount of nodes will be formed. 

 

 

Exercise 5: 

The average opinion function calculates the mean opinion of all nodes in the network at each step by iterating over each node, retrieving their opinions and calculating the mean opinion. The network calculate agreement function collects the agreement of each node with the indexed node by iterating over the neighbouring nodes, collecting their opinions and calculating the agreement , this is also influenced by an external pull value. The ising network iteration causes a single update of the ising model by randomly selecting a node from the network and calculating the agreement of this selected node with its neighbours. If this agreement value is negative the node’s value will flip and if it is positive or zero the node goes through a probability that the node’s opinion will change anyway. The ising network opinions function is the main plot for the ising model when based on networks. It creates a network and sets each node with a random value between 1 and –1, representing their opinion. It updates the network and plots each update and the mean opinion of the network over time. This can be called by using –ising paired with –use_network (N), where N represents the amount of nodes in the network. 

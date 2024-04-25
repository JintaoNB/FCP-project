import numpy as np
import argparse

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

	def add_node(self, node):
		self.nodes.append(node)

	def create_random_network(self, size, prob_connection):
		# Create nodes
		for i in range(size):
			self.add_node(Node(value=i, number=i))

		for node in self.nodes:
			for other_node in self.nodes:
				if node != other_node and np.random.rand() < prob_connection:
					if node.connections is None:
						node.connections = [other_node]
					else:
						node.connections.append(other_node)

	def get_mean_degree(self):
		mean_degree = sum(len(node.connections) for node in self.nodes) / len(self.nodes)
		return mean_degree

	def get_mean_path_length(self):
		total_path_length = 0
		total_paths = 0
		for node in self.nodes:
			for other_node in self.nodes:
				if node != other_node:
					path_length = self.bfs_path_length(node, other_node)
					if path_length is not None:
						total_path_length += path_length
						total_paths += 1
		mean_path_length = total_path_length / total_paths
		return mean_path_length

	def get_mean_clustering(self):
		mean_coefficient = sum(self.clustering_coefficient(node) for node in self.nodes) / len(self.nodes)
		return mean_coefficient

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


def main():
	if args.test_networks:
		test_networks()
	elif args.network_size:
		network = Network()
		network.create_random_network(args.network_size, 0.5)
		mean_degree = network.mean_degree()
		average_path_length = network.mean_path_length()
		clustering_coefficient = network.mean_clustering_coefficient()
		print(f"Mean degree: {mean_degree}")
		print(f"Average path length: {average_path_length}")
		print(f"Clustering coefficient: {clustering_coefficient}")
	# Uncomment below to plot the network
	# network.plot_network()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Network simulation with flag handling")
	parser.add_argument("-test_networks", action="store_true", help="Test networks")
	parser.add_argument("-network", "--network_size", type=int, help="Size of the network")
	args = parser.parse_args()
	main()

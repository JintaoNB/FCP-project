import numpy as np
import argparse

class Node:

	def __init__(self, value, number, connections=None): # Define the initial nodes

		self.index = number
		self.connections = connections
		self.value = value

class Network: 

	def __init__(self, nodes=None): # Define the initial network

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

	def clustering_coefficient(self, node): # Get the number of triangles formed by its neighbours divided by the maximum possible number of triangles.
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

	def bfs_path_length(self, start_node, end_node): # Using breadth-first search
		visited = set()
		queue = [(start_node, 0)]
		while queue:
			current_node, distance = queue.pop(0)
			if current_node == end_node:
				return distance
			visited.add(current_node)
			for neighbor in self.get_neighbors(current_node): # For each node, calculate the average shortest path length to all other nodes.
				if neighbor not in visited:
					queue.append((neighbor, distance + 1))
		return None

	def get_neighbors(self, node):
		return [self.nodes[i] for i, connected in enumerate(node.connections) if connected]

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

	def make_random_network(self, N, connection_probability): # Create a random network
		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index + 1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

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


def main(args):
	if args.test_networks:
		test_networks()
	elif args.network_size:
		network = Network()
		network.make_random_network(args.network_size, 0.5)
		mean_degree = network.get_mean_degree()
		average_path_length = network.get_path_length()
		mean_clustering_coefficient = network.get_clustering()
		print(f"Mean degree: {mean_degree}")
		print(f"Average path length: {average_path_length}")
		print(f"Clustering coefficient: {mean_clustering_coefficient}")
	# Uncomment below to plot the network
	# network.plot_network()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Network simulation with flag handling")
	parser.add_argument("-test_networks", action="store_true", help="Test networks")
	parser.add_argument("-network", "--network_size", type=int, help="Size of the network")
	args = parser.parse_args()
	main(args)

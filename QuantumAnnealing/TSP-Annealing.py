#travelling salesman problem with quantum annealing
#Only works through the D-Wave Leap cloud service
#Robert Vila, 2023
#Universitat de Barcelona

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dwave_networkx as dnx
from dwave.system import DWaveSampler, EmbeddingComposite
import dimod

# number of cities
n = 5

#Set one decimal random coordinates for the cities in a 100x100 map
cities = np.random.randint(0,1000,(n,2))/10.

#Calculate the distance between cities
distance = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        distance[i][j] = np.sqrt((cities[i][0]-cities[j][0])**2 + (cities[i][1]-cities[j][1])**2)
        

llista = []

for i in range(n):
    for j in range(n):
        if i != j:
            llista.append((i,j,distance[i][j]))

G = nx.Graph()
print(llista)
G.add_weighted_edges_from(llista)

path_nodes = dnx.traveling_salesperson(G, dimod.ExactSolver(), start=0)

print(path_nodes)
# Create a new graph with only the nodes in the optimal path
path_edges = [(path_nodes[i], path_nodes[i+1]) for i in range(len(path_nodes)-1)]
path_graph = nx.Graph()
path_graph.add_nodes_from(path_nodes)
path_graph.add_edges_from(path_edges)

# Make the figure of the cities and the path
plt.figure()
nx.draw(G, pos=nx.circular_layout(G), node_color='r', node_size=50)
nx.draw_networkx_edges(path_graph, pos=nx.circular_layout(G), edge_color='b', width=2)
plt.savefig('path.png')

# figure of the optimal path in a 100x100 map
plt.figure()
path_nodes.append(path_nodes[0])
data = np.array([cities[i] for i in path_nodes])
plt.plot(data[:,0],data[:,1],'-o')
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig('path_map.png')

# Figure with the cities in a 100x100 map
plt.figure()
plt.plot(cities[:,0],cities[:,1],'o')
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig('cities.png')

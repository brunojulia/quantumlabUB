import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import dwave_networkx as dnx
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import BinaryQuadraticModel

#Number of cities
n = 10

#Set one decimal random coordinates for the cities in a 100x100 map
cities = np.random.randint(0,1000,(n,2))/10.
cities_list = list(range(n))
print(cities_list)
x = [[f'x_{i}_{j}' for j in cities_list] for i in cities_list]

#Calculate the distance between cities
distance = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        distance[i][j] = np.sqrt((cities[i][0]-cities[j][0])**2 + (cities[i][1]-cities[j][1])**2)
        
#Give the diagonal a big value
for i in range(n):
    distance[i][i] = 10000


#Represent the cities in a map and save it to png file
plt.plot(cities[:,0],cities[:,1],'o')
plt.xlim(0,100)
plt.ylim(0,100)
plt.savefig('cities.png')

bqm = BinaryQuadraticModel('BINARY')

#Solve tsp using dwave_networkx.algorithms.tsp.traveling_salesperson
G = nx.from_numpy_matrix(distance)

#Find the optimal path
path = dnx.traveling_salesperson(G, sampler=EmbeddingComposite(DWaveSampler()), start=0)
print('Optimal path:', path)

#Plot the optimal path
plt.figure()
nx.draw(G, pos=cities, node_color='r', node_size=50)
nx.draw_networkx_edges(G, pos=cities, edgelist=path.edges, edge_color='b', width=2)
plt.savefig('path.png')

#Calculate the total distance
total_distance = 0
for i in range(n-1):
    total_distance += distance[path[i]][path[i+1]]
total_distance += distance[path[n-1]][path[0]]
print('Total distance:', total_distance)


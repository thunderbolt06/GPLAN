import numpy as np
import networkx as nx

def biconnectivity_transformation(graph,edge,biconnected_vertices):
	vertices = []
	for triangle in graph.triangles:
		# print(triangle)
		if(edge[0] in triangle and edge[1] in triangle):
			if(len([x for x in triangle if x not in edge])!=0 and [x for x in triangle if x not in edge][0] not in vertices):
				vertices.append([x for x in triangle if x not in edge][0])
	graph.node_count +=1
	new_adjacency_matrix = np.zeros([graph.node_count, graph.node_count], int)
	new_adjacency_matrix[0:graph.matrix.shape[0],0:graph.matrix.shape[1]] = graph.matrix
	biconnected_vertices.append(graph.node_count-1)
	#extra edges being added and shortcut being deleted
	new_adjacency_matrix[edge[0]][edge[1]] = 0
	new_adjacency_matrix[edge[1]][edge[0]] = 0
	new_adjacency_matrix[graph.node_count-1][edge[0]] = 1
	new_adjacency_matrix[graph.node_count-1][edge[1]] = 1
	new_adjacency_matrix[edge[0]][graph.node_count-1] = 1
	new_adjacency_matrix[edge[1]][graph.node_count-1] = 1
	graph.edge_count += 1
	graph.north +=1
	graph.east +=1
	graph.west +=1
	graph.south +=1
	for j in vertices:
		new_adjacency_matrix[graph.node_count-1][j] = 1
		new_adjacency_matrix[j][graph.node_count-1] = 1
		graph.edge_count +=1
	graph.matrix = new_adjacency_matrix

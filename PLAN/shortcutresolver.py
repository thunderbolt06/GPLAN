import numpy as np
import networkx as nx 

# returns all shortcuts present in a graph
def get_shortcut(graph):
    shortcut =[]
    for i in range(0,len(graph.outer_vertices)):
        for j in range(0,len(graph.outer_vertices)):
            if(graph.matrix[graph.outer_vertices[i]][graph.outer_vertices[j]] == 1 and (graph.outer_vertices[i],graph.outer_vertices[j]) not in graph.outer_boundary and [graph.outer_vertices[j],graph.outer_vertices[i]] not in shortcut):
                shortcut.append([graph.outer_vertices[i],graph.outer_vertices[j]])
    return shortcut

#remove a particular shortcut from a graph
def remove_shortcut(shortcut_to_be_removed,graph,rdg_vertices,rdg_vertices2,to_be_merged_vertices):
    neighbour_vertices =[]
    triangles = graph.triangles.copy()
    #identifies the common neighbour vertices of both the vertices
    for i in triangles:
        if(shortcut_to_be_removed[0] in i and shortcut_to_be_removed[1] in i):
            for a in i:
                if(a not in shortcut_to_be_removed and a not in neighbour_vertices):
                    neighbour_vertices.append(a)
    graph.node_count +=1		#extra vertex added
    new_adjacency_matrix = np.zeros([graph.node_count, graph.node_count], int)
    new_adjacency_matrix[0:graph.matrix.shape[0],0:graph.matrix.shape[1]] = graph.matrix
    rdg_vertices.append(shortcut_to_be_removed[0])
    rdg_vertices2.append(shortcut_to_be_removed[1])
    to_be_merged_vertices.append(graph.node_count-1)
    #extra edges being added and shortcut being deleted
    new_adjacency_matrix[shortcut_to_be_removed[0]][shortcut_to_be_removed[1]] = 0
    new_adjacency_matrix[shortcut_to_be_removed[1]][shortcut_to_be_removed[0]] = 0
    new_adjacency_matrix[graph.node_count-1][shortcut_to_be_removed[0]] = 1
    new_adjacency_matrix[graph.node_count-1][shortcut_to_be_removed[1]] = 1
    new_adjacency_matrix[graph.node_count-1][neighbour_vertices[0]] = 1
    new_adjacency_matrix[graph.node_count-1][neighbour_vertices[1]] = 1
    new_adjacency_matrix[shortcut_to_be_removed[0]][graph.node_count-1] = 1
    new_adjacency_matrix[shortcut_to_be_removed[1]][graph.node_count-1] = 1
    new_adjacency_matrix[neighbour_vertices[0]][graph.node_count-1] = 1
    new_adjacency_matrix[neighbour_vertices[1]][graph.node_count-1] = 1
    graph.edge_count += 3
    graph.matrix = new_adjacency_matrix
    graph.north +=1
    graph.east +=1
    graph.west +=1
    graph.south +=1
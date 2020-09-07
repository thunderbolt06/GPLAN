import numpy as np
import networkx as nx 

#Returns intersection between two lists as a new list
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3
 
# Checks if there exist an element in 'cln' whose intersection with 'lst1' is of length 'size'
def list_comparer(lst1,cln,size):
    for temp in cln:
        if(len(intersection(lst1,temp)) == size):
            return True
    return False

#Returns directed graph of a given graph
def get_directed(graph):
	H = nx.from_numpy_matrix(graph.matrix,create_using=nx.DiGraph)
	return H

# Returns all triangles of the input graph
def get_all_triangles(graph):
    #H = get_directed(graph)
    H = nx.from_numpy_matrix(graph.matrix,create_using=nx.DiGraph)
    all_cycles = list(nx.simple_cycles(H))
    all_triangles = []
    for cycle in all_cycles:
        if len(cycle) == 3:
            all_triangles.append(cycle)
    return all_triangles

#Returns outer boundary and outer vertices of a graph
def get_outer_boundary_vertices(graph):
    
    all_triangles = get_all_triangles(graph)
    H = get_directed(graph)         
    # H = graph.directed.copy()
    outer_boundary = []
    for edge in H.edges:
        count = 0
        for triangle in all_triangles:
            if edge[0] in triangle and edge[1] in triangle:
                count += 1
        if count <= 2:
            outer_boundary.append(edge)
    outer_vertices = []
    for edge in outer_boundary:
        if edge[0] not in outer_vertices:
            outer_vertices.append(edge[0])
        if edge[1] not in outer_vertices:
            outer_vertices.append(edge[1])
    return outer_vertices,outer_boundary




def get_ordered_neighbour_label(graph, centre, y, clockwise=False):
    next = get_ordered_neighbour(graph,centre, y, clockwise)
    if graph.matrix[centre][next] == 2 or graph.matrix[next][centre] == 2:
        return 2
    else:
        return 3

def get_ordered_neighbour(graph, centre, y, clockwise=False):
    ordered_neighbours = order_neighbours(graph,centre, clockwise)
    return ordered_neighbours[(ordered_neighbours.index(y) + 1) % len(ordered_neighbours)]

def order_neighbours(graph, centre, clockwise=False):
    vertex_set = np.concatenate([np.where(np.logical_or(graph.matrix[centre] == 2, graph.matrix[centre] == 3))[0],
                                     np.where(np.logical_or(graph.matrix[:, centre] == 2, graph.matrix[:, centre] == 3))[0]]).tolist()
    ordered_set = [vertex_set.pop(0)]
    while len(vertex_set) != 0:
        for i in vertex_set:
            if graph.matrix[ordered_set[len(ordered_set) - 1]][i] != 0 \
                    or graph.matrix[i][ordered_set[len(ordered_set) - 1]] != 0:
                ordered_set.append(i)
                vertex_set.remove(i)
                break
            elif graph.matrix[ordered_set[0]][i] != 0 or graph.matrix[i][ordered_set[0]] != 0:
                ordered_set.insert(0, i)
                vertex_set.remove(i)
                break

    current = 0
    # case: centre is the South vertex
    if centre == graph.south:
        if graph.matrix[graph.west][ordered_set[0]] != 0:
            ordered_set.reverse()

    # case: centre is the West vertex
    elif centre == graph.west:
        if graph.matrix[ordered_set[0]][graph.north] != 0:
            ordered_set.reverse()

    # case: first vertex is in t1_leaving
    elif graph.matrix[centre][ordered_set[0]] == 2:
        while graph.matrix[centre][ordered_set[current]] == 2:
            current += 1
        if graph.matrix[centre][ordered_set[current]] == 3:
            ordered_set.reverse()

    # case: first vertex is in t2_entering
    elif graph.matrix[ordered_set[0]][centre] == 3:
        while graph.matrix[ordered_set[current]][centre] == 3:
            current += 1
        if graph.matrix[centre][ordered_set[current]] == 2:
            ordered_set.reverse()

    # case: first vertex is in t1_entering
    elif graph.matrix[ordered_set[0]][centre] == 2:
        while graph.matrix[ordered_set[current]][centre] == 2:
            current += 1
        if graph.matrix[ordered_set[current]][centre] == 3:
            ordered_set.reverse()

    # case: first vertex is in t2_leaving
    elif graph.matrix[centre][ordered_set[0]] == 3:
        while graph.matrix[centre][ordered_set[current]] == 3:
            current += 1
        if graph.matrix[ordered_set[current]][centre] == 2:
            ordered_set.reverse()

    if clockwise:
        ordered_set.reverse()
    return ordered_set


def get_encoded_matrix(graph):
    encoded_matrix =  np.zeros((graph.t2_matrix.shape[0],graph.t1_matrix.shape[1]), int)
    room_width = np.array(graph.room_width, dtype='int')
    room_height = np.array(graph.room_height, dtype='int')
    room_x = np.array(graph.room_x, dtype='int')
    room_y = np.array(graph.room_y, dtype='int')
    for node in range(graph.matrix.shape[0]-4):
        for width in range(room_width[node]):
            for height in range(room_height[node]):
                encoded_matrix[room_y[node]+height][room_x[node]+width] = node
    return encoded_matrix

def is_complex_triangle(graph):
    for node in range(0,graph.original_node_count):
        value = np.count_nonzero(graph.matrix[node])
        if(value <4):
            return True
    H = nx.from_numpy_matrix(graph.matrix,create_using=nx.DiGraph)
    all_cycles = list(nx.simple_cycles(H))
    all_triangles = 0
    for cycle in all_cycles:
        if len(cycle) == 3:
            all_triangles+=1
    vertices = graph.matrix.shape[0]
    edges = int(np.count_nonzero(graph.matrix)/2)
    if(int(all_triangles/2) == (edges-vertices + 1)):
        return False
    else:
        return True

def ordered_outer_boundary(graph):
    vertices = get_outer_boundary_vertices(graph)[0]
    edges = get_outer_boundary_vertices(graph)[1]

    # print(vertices,edges)
    ordered_vertices = [vertices[0]]
    while(len(ordered_vertices) != len(vertices)):
        temp = ordered_vertices[len(ordered_vertices)-1]
        # print(temp)
        for vertex in vertices:
            # print(vertex)
            if((temp,vertex) in edges and vertex not in ordered_vertices):
                ordered_vertices.append(vertex)
                break
        # if(len(ordered_vertices) > 2):
        #     break
    # print(ordered_vertices)
    return ordered_vertices

def find_possible_boundary(boundary):
    list_of_boundaries = []
    for i in boundary:
        index = boundary.index(i)
        temp1 = boundary[0:index]
        temp2 = boundary[index:len(boundary)]
        temp = temp2 + temp1
        temp.append(temp[0])
        list_of_boundaries.append(temp)
        # print(list_of_boundaries)
    return list_of_boundaries

def calculate_area(graph,to_be_merged_vertices,rdg_vertices):
    for i in range(graph.room_x.shape[0]):
        if graph.room_width[i] == 0 or i in graph.biconnected_vertices or i in to_be_merged_vertices:
            continue
        area = graph.room_width[i]*graph.room_height[i]
        if(i in rdg_vertices):
            area+= graph.room_width[to_be_merged_vertices[rdg_vertices.index(i)]]*graph.room_height[to_be_merged_vertices[rdg_vertices.index(i)]]
        graph.area.append(round(area,3))


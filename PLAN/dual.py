import networkx as nx 
import numpy as np 
import operations as opr

def populate_t1_matrix(graph):
    get_n_s_paths(graph,graph.south, [graph.south])
    graph.t1_matrix = np.empty((0, graph.t1_longest_distance_value + 1), int)
    row_index = 0
    for path in graph.n_s_paths:
        is_valid_path = True
        row = [-1] * (graph.t1_longest_distance_value + 1)
        path_index = 0
        current_vertex = path[path_index]
        for distance in range(graph.t1_longest_distance_value + 1):
            if path_index + 1 < len(path) and graph.t1_longest_distance[path[path_index + 1]] <= distance:
                path_index += 1
                current_vertex = path[path_index]
            if row_index != 0 and graph.t1_matrix[row_index - 1][distance] != current_vertex \
                    and current_vertex in graph.t1_matrix[:, distance]:
                is_valid_path = False
                break
            row[distance] = current_vertex
        if is_valid_path:
            graph.t1_matrix = np.append(graph.t1_matrix, [row], axis=0)
            row_index += 1
    graph.t1_matrix = graph.t1_matrix.transpose()

# while populating the t1_matrix we need N-S paths such that they are obtained in a DFS ordered manner with children
# obtained in anticlockwise direction..... but in the REL we have S-N paths... so we construct the S-N path with
# children obtained in clockwise direction and reverse the path when we reach N.
def get_n_s_paths(graph, source, path):
    if source == graph.north: # base case of this recursive function as every S-N ends at N

        # making a deep copy of the path array as it changes during the recursive calls and wew want o save the
        # current state of this array
        path_deep_copy = [i for i in path]

        path_deep_copy.reverse() # reversing the array to get N-S path from the S-N path

        #iterating over the nodes in path and updating their longest distance from north
        for i in range(len(path_deep_copy)):
            node = path_deep_copy[i]
            graph.t1_longest_distance[node] = max(graph.t1_longest_distance[node], i) # index i represent the distance of node from north
            # updating the length of the longest N-S path
            graph.t1_longest_distance_value = max(graph.t1_longest_distance_value, graph.t1_longest_distance[node])

        # adding this path in the n_s_paths
        graph.n_s_paths.append(path_deep_copy)
        return

    # if we have not reached north yet then we get the children of the current source node and continue this DFS
    # to reach N from each children
    ordered_children = get_t1_ordered_children(graph,source)
    for child in ordered_children:
        path.append(child)
        get_n_s_paths(graph,child, path)
        path.remove(child)

def get_t1_ordered_children(graph, centre):
    ordered_neighbours = opr.order_neighbours(graph,centre, clockwise=True)
    index = 0
    ordered_children = []
    if centre == graph.south:
        return ordered_neighbours
    while graph.matrix[ordered_neighbours[index]][centre] != 3:
        index = (index + 1) % len(ordered_neighbours)
    while graph.matrix[ordered_neighbours[index]][centre] == 3:
        index = (index + 1) % len(ordered_neighbours)
    while graph.matrix[centre][ordered_neighbours[index]] == 2:
        ordered_children.append(ordered_neighbours[index])
        index = (index + 1) % len(ordered_neighbours)
    return ordered_children

def populate_t2_matrix(graph):
    get_w_e_paths(graph,graph.west, [graph.west])
    graph.t2_matrix = np.empty((0, graph.t2_longest_distance_value + 1), int)
    row_index = 0
    for path in graph.w_e_paths:
        is_valid_path = True
        row = [-1] * (graph.t2_longest_distance_value + 1)
        path_index = 0
        current_vertex = path[path_index]
        for distance in range(graph.t2_longest_distance_value + 1):
            if path_index + 1 < len(path) and graph.t2_longest_distance[path[path_index + 1]] <= distance:
                path_index += 1
                current_vertex = path[path_index]
            if row_index != 0 and graph.t2_matrix[row_index - 1][distance] != current_vertex \
                    and current_vertex in graph.t2_matrix[:, distance]:
                is_valid_path = False
                break
            row[distance] = current_vertex
        if is_valid_path:
            graph.t2_matrix = np.append(graph.t2_matrix, [row], axis=0)
            row_index += 1

def get_w_e_paths(graph, source, path):
    graph.t2_longest_distance[source] = max(graph.t2_longest_distance[source], len(path) - 1)
    graph.t2_longest_distance_value = max(graph.t2_longest_distance_value, graph.t2_longest_distance[source])
    if source == graph.east:
        path_deep_copy = [i for i in path]
        graph.w_e_paths.append(path_deep_copy)
        return
    ordered_children = get_t2_ordered_children(graph,source)
    for child in ordered_children:
        path.append(child)
        get_w_e_paths(graph,child, path)
        path.remove(child)

def get_t2_ordered_children(graph, centre):
    ordered_neighbours = opr.order_neighbours(graph,centre, clockwise=True)
    index = 0
    ordered_children = []
    if centre == graph.west:
        return ordered_neighbours
    while graph.matrix[centre][ordered_neighbours[index]] != 2:
        index = (index + 1) % len(ordered_neighbours)
    while graph.matrix[centre][ordered_neighbours[index]] == 2:
        index = (index + 1) % len(ordered_neighbours)
    while graph.matrix[centre][ordered_neighbours[index]] == 3:
        ordered_children.append(ordered_neighbours[index])
        index = (index + 1) % len(ordered_neighbours)
    return ordered_children

def get_coordinates(graph,hor_dgph):
        
        def ismember(d, k):
            return [1 if (i == k) else 0 for i in d]

        def any(A):
            for i in A:
                if i != 0:
                    return 1
            return 0

        def find_sp(arr):
            for i in range(0,len(arr)):
                if arr[i]==1:
                    return [i+1]
            return [0]

        def find(arr):
            for i in range(0,len(arr)):
                if arr[i]==1:
                    return [i]
            return [0]

        hor_dgph=np.array(hor_dgph)
        hor_dgph=hor_dgph.transpose()
        xmin=float(0)
        ymin=float(0)
        B=np.array(graph.encoded_matrix)
        m=len(B[0])
        n=len(B)
        N=np.amax(B)+1
        rect_drawn=[]
        # C = np.zeros((n,m))
        # for i in range(0,m):
        #     temp = len(np.unique(np.transpose(B[i])))
        #     for j in range(0,temp):
        #         C[j][i] = np.unique(np.transpose(B[i]))[j]
        # print(C)


        j=0
        C=[[-1 for i in range(0,len(B[0]))] for i in range(0,len(B))]
        # print(C)
        while j<len(B[0]):
            rows=[]
            for i in range(0,len(B)):
                if B[i][j] not in rows:
                    rows.append(B[i][j])
            k=0
            for k in range(0,len(rows)):
                C[k][j]=rows[k]
            j+=1
        # print(C)


        # for i in range(0,len(C)):
        #     for j in range(0,len(C[0])):
        #         C[i][j] +=1
        xR=np.zeros((N),float)
        for i in range(0,m):
            xmax=np.zeros((N),float)
            ymin=0
            for j in range(0,n):
                if C[j][i]==-1:
                    break
                else:
                    if any(ismember(rect_drawn,C[j][i])):
                        ymin = ymin + graph.room_height[C[j][i]]
                        xmax=np.zeros((N),float)
                        xmax[0]=xR[C[j][i]]
                        continue
                    else:
                        if not any(find_sp(hor_dgph[C[j][i]])):
                            ymin=ymin
                        else:
                            l=find(hor_dgph[C[j][i]])
                            xmin=xR[l]
                    graph.room_x[C[j][i]],graph.room_y[C[j][i]]=xmin,ymin #-graph.room_height[C[j][i]]  #not subtracting height because code requires top left corner
                    rect_drawn.append(C[j][i])
                    xmax[C[j][i]]=xmin+graph.room_width[C[j][i]]
                    xR[C[j][i]]=xmax[C[j][i]]
                    ymin = ymin + graph.room_height[C[j][i]]
            xmax=xmax[xmax!=0]
            xmin=min(xmax)

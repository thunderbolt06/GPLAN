import sys
from types import ModuleType

class MockModule(ModuleType):
    def __init__(self, module_name, module_doc=None):
        ModuleType.__init__(self, module_name, module_doc)
        if '.' in module_name:
            package, module = module_name.rsplit('.', 1)
            get_mock_module(package).__path__ = []
            setattr(get_mock_module(package), module, self)

    def _initialize_(self, module_code):
        self.__dict__.update(module_code(self.__name__))
        self.__doc__ = module_code.__doc__

def get_mock_module(module_name):
    if module_name not in sys.modules:
        sys.modules[module_name] = MockModule(module_name)
    return sys.modules[module_name]

def modulize(module_name, dependencies=[]):
    for d in dependencies: get_mock_module(d)
    return get_mock_module(module_name)._initialize_

##===========================================================================##

@modulize('operations')
def _operations(__name__):
    ##----- Begin operations.py --------------------------------------------------##
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
        
        #all_triangles = get_all_triangles(graph)
        all_triangles = get_all_triangles(graph)
        if graph.node_count <3:
            print("hi")
            return [i for i in graph.graph.nodes()],[]
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
        print(vertices)
        print(edges)
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
    
    
    ##----- End operations.py ----------------------------------------------------##
    return locals()

@modulize('shortcutresolver')
def _shortcutresolver(__name__):
    ##----- Begin shortcutresolver.py --------------------------------------------##
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
    ##----- End shortcutresolver.py ----------------------------------------------##
    return locals()

@modulize('news')
def _news(__name__):
    ##----- Begin news.py --------------------------------------------------------##
    import networkx as nx 
    import numpy as np 
    import operations as opr
    import shortcutresolver as sr 
    import time
    from random import randint
    import copy
    
    # Get top,left,right and bottom boundaries of graph        
    def find_cip(graph):
        H = opr.get_directed(graph)
        civ = []
        outer_vertices = graph.outer_vertices
        outer_boundary = graph.outer_boundary
        # print(outer_vertices)
    # Finds all corner implying paths in the graph
        while len(outer_vertices) > 1:
            cip_store = [outer_vertices[0]] #stores the corner implying paths
            outer_vertices.pop(0)
            for vertices in cip_store:
                for vertex in outer_vertices:
                    cip_store_copy = cip_store.copy()
                    cip_store_copy.pop(len(cip_store) - 1)
                    if (cip_store[len(cip_store) - 1], vertex) in outer_boundary:
                        cip_store.append(vertex)
                        outer_vertices.remove(vertex)
                        if cip_store_copy is not None:  #checks for existence of shortcut
                            for vertex1 in cip_store_copy:
                                if (vertex1, vertex) in H.edges:
                                    cip_store.remove(vertex)
                                    outer_vertices.append(vertex)
                                    break
            civ.append(cip_store)       #adds the corner implying path to civ
            outer_vertices.insert(0, cip_store[len(cip_store) - 1]) #handles the last vertex of the corner implying path added
            if len(outer_vertices) == 1:        #works for the last vertex left in the boundary
                last_cip=0
                first_cip=0
                merge_possible =0
                for test in civ[len(civ)-1]:            #checks last corner implying path
                    if((test,civ[0][0]) in H.edges and (test,civ[0][0]) not in outer_boundary ):
                        last_cip = 1
                        first_cip = 0
                        break
                for test in civ[0]:             #checks first corner implying path
                    if((test,outer_vertices[0]) in H.edges and (test,outer_vertices[0]) not in outer_boundary):
                        last_cip = 1
                        first_cip = 1
                        break
                if last_cip == 0 and len(civ)!=2:       #if merge is possible as well as both civs are available for last vertex
                    for test in civ[len(civ)-1]:
                        for test1 in civ[0]:
                            if ((test,test1) in H.edges and (test,test1) not in H.edges):
                                merge_possible = 1
                    if(merge_possible == 1):                  #adding last vertex to last civ
                        civ[len(civ)-1].append(civ[0][0])
                    else:                                     #merging first and last civ
                        civ[0] = civ[len(civ)-1] + list(set(civ[0]) - set(civ[len(civ)-1]))
                        civ.pop()
                elif(last_cip == 0 and len(civ)==2):          #if there are only 2 civs
                    civ[len(civ)-1].append(civ[0][0])
                elif (last_cip ==1 and first_cip == 0):      #adding last vertex to first civ
                    civ[0].insert(0,outer_vertices[0])
                elif (last_cip ==0 and first_cip == 1):      #adding last vertex to last civ
                    civ[len(civ)-1].append(civ[0][0])
                elif (last_cip == 1 and first_cip == 1):     #making a new corner implying path
                    civ.append([outer_vertices[0],civ[0][0]])
    
        print(civ)
    
        if(len(sr.get_shortcut(graph))==0):
            civ.append(civ[0]+civ[1])
            civ[len(civ)-1].pop(len(civ[0]))
            civ.pop(0)
            civ.pop(0)
        if(len(civ)<4):
            for i in range(4-len(civ)):
                index = civ.index(max(civ,key =len))
                # print(index)
                create_cip(civ,index)
        return civ
    
    def find_cip_single(graph):
        boundary_vertices = opr.ordered_outer_boundary(graph)
        boundary_vertices.append(boundary_vertices[0])
        outer_boundary = opr.get_outer_boundary_vertices(graph)[1]
        print("ver" , boundary_vertices)
        print("bound",outer_boundary)
        civ = []
        temp = []
        for i in range(0,len(boundary_vertices)):
            breakpoint = 0
            if(len(temp) == 0):
                temp.append(boundary_vertices[i])
                continue
            else:
                for j in temp:
                    if(boundary_vertices[i],j) not in outer_boundary and graph.matrix[boundary_vertices[i]][j] == 1:
                        breakpoint =1
                        break
            if breakpoint == 1:
                value = temp[len(temp)-1]
                civ.append(temp)
                temp = []
                temp.append(value)
                temp.append(boundary_vertices[i])
            else:
                temp.append(boundary_vertices[i])
        civ.append(temp)
        merge = 0
        if(len(civ)>1):
            for i in civ[len(civ)-1]:
                for j in civ[0]:
                    if (i,j) not in outer_boundary and graph.matrix[i][j] == 1:
                        merge = 1
                        break
                if merge == 1:
                    break
            if merge == 0:
                for i in range(1,len(civ[0])):
                    civ[len(civ)-1].append(civ[0][i])
                civ.pop(0)
        print(civ)
        if(len(civ)<4):
            for i in range(4-len(civ)):
                index = civ.index(max(civ,key =len))
                # print(index)
                create_cip(civ,index)
        return civ
    
    # get four corner implying paths in case there are less than 4 civs
    def create_cip(civ,index):
        civ.insert(index + 1, civ[index])
        length = int(len(civ[index])/2)
        civ[index] = civ[index][0:2]
        del civ[index + 1][0:1]
    
    # connect civs to north, east, west and south vertices
    def news_edges(graph,matrix,civ, source_vertex):
        for vertex in civ:
            graph.edge_count += 1
            matrix[source_vertex][vertex] = 1
            matrix[vertex][source_vertex] = 1
    
    
    def populate_cip_list(graph):
        new_list_of_cips =[]
        list_of_cips =[]
        shortcuts = graph.shortcuts
        for i in range(0,len(graph.boundaries)):
            temp = copy.deepcopy(graph.boundaries[i])
            cip_list = [[temp]]
            while(len(cip_list[0])<4):
                length = len(cip_list)
                for k in range(0,length):
                    abc = copy.deepcopy(cip_list[k])
                    for j in abc:
                        l = copy.deepcopy(j)
                        index = abc.index(j)
                        for i in range(1,len(j)):
                            temp = copy.deepcopy(abc)
                            a = l[0:i]
                            b = l[i-1:len(l)]
                            temp.insert(index,a)
                            temp.insert(index+1,b)
                            temp.remove(j)
                            if(temp not in cip_list):
                                cip_list.append(temp)
                cip_list = cip_list[length:len(cip_list)]
            for i in cip_list:
                count = 0
                for j in i:
                    for shortcut in shortcuts:
                        if(shortcut[0] in j and shortcut[1] in j):
                            count = 1
                            break
                    if(count == 1):
                        break
                if(count != 1):
                    list_of_cips.append(i)
        for i in list_of_cips:
            if([i[3],i[0],i[1],i[2]] not in new_list_of_cips and [i[2],i[3],i[0],i[1]] not in new_list_of_cips and [i[1],i[2],i[3],i[0]] not in new_list_of_cips):
                new_list_of_cips.append(i)
    
        
        return new_list_of_cips
    
    #connect north, west, south and east vertics to each other:
    def connect_news(matrix,graph):
        matrix[graph.north][graph.west] = 1
        matrix[graph.west][graph.north] = 1
        matrix[graph.west][graph.south] = 1
        matrix[graph.south][graph.west] = 1
        matrix[graph.south][graph.east] = 1
        matrix[graph.east][graph.south] = 1
        matrix[graph.north][graph.east] = 1
        matrix[graph.east][graph.north] = 1
        
    
    
    #Add north,east, west and south vertices
    def add_news_vertices(graph):
        civ = graph.civ
        # print(civ)
        if(len(civ)>4):
            shortcut = sr.get_shortcut(graph)
            print('Shortcut:')
            print(shortcut)
            while(len(shortcut)>4):
                index = randint(0,len(shortcut)-1)
                sr.remove_shortcut(shortcut[index],graph,graph.rdg_vertices,graph.rdg_vertices2,graph.to_be_merged_vertices)
                shortcut.pop(index)
            print(shortcut)
            civ = find_cip_single(graph)
        # if(len(civ)<4):
        #     for i in range(4-len(civ)):
        #         index = civ.index(max(civ,key =len))
        #         # print(index)
        #         create_cip(civ,index)
                # print(civ)
        # print(civ)
        # civ = [[6,0,4],[4,1,5],[5,7],[7,2,6]]
        graph.node_count += 4
        new_adjacency_matrix = np.zeros([graph.node_count, graph.node_count], int)
        new_adjacency_matrix[0:graph.matrix.shape[0],0:graph.matrix.shape[1]] = graph.matrix
        news_edges(graph,new_adjacency_matrix,civ[0], graph.north)
        news_edges(graph,new_adjacency_matrix,civ[1], graph.east)
        news_edges(graph,new_adjacency_matrix,civ[2], graph.south)
        news_edges(graph,new_adjacency_matrix,civ[3], graph.west)
        graph.edge_count += 4
        connect_news(new_adjacency_matrix,graph)
        graph.matrix = new_adjacency_matrix.copy()
        graph.user_matrix = new_adjacency_matrix
    
    
    
    ##----- End news.py ----------------------------------------------------------##
    return locals()

@modulize('dual')
def _dual(__name__):
    ##----- Begin dual.py --------------------------------------------------------##
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
    
    ##----- End dual.py ----------------------------------------------------------##
    return locals()

@modulize('drawing', dependencies=['ptpg'])
def _drawing(__name__):
    ##----- Begin drawing.py -----------------------------------------------------##
    import networkx as nx 
    import numpy as np 
    import turtle
    import ptpg
    import dual
    import math
    
    scale = 300
    origin = {'x': 300, 'y': -150}
    
    #Draw undirected graph 
    def draw_undirected_graph(graph,pen):
        pen.clear()
        pen.pencolor('black')
        pen.penup()
        for from_node in range(graph.matrix.shape[0]):
            pen.setposition(graph.node_position[from_node][0] * scale + origin['x'],
                            graph.node_position[from_node][1] * scale + origin['y'])
            if from_node == graph.north:
                pen.write("N")
            elif from_node == graph.south:
                pen.write("S")
            elif from_node == graph.east:
                pen.write("E")
            elif from_node == graph.west:
                pen.write("W")
            else:
                pen.write(from_node)
            for to_node in range(from_node):
                if graph.matrix[from_node][to_node] == 1:
                    pen.setposition(graph.node_position[from_node][0] * scale + origin['x'],
                                    graph.node_position[from_node][1] * scale + origin['y'])
                    pen.pendown()
                    pen.setposition(graph.node_position[to_node][0] * scale + origin['x'],
                                    graph.node_position[to_node][1] * scale + origin['y'])
                    pen.penup()
    
    #Draw directed graph
    def draw_directed_graph(graph,pen):
        pen.clear()
        pen.width(1)
        pen.penup()
        for from_node in range(graph.matrix.shape[0]):
            pen.setposition(graph.node_position[from_node][0] * scale + origin['x'],
                            graph.node_position[from_node][1] * scale + origin['y'])
            if from_node == graph.north:
                pen.write("N")
            elif from_node == graph.south:
                pen.write("S")
            elif from_node == graph.east:
                pen.write("E")
            elif from_node == graph.west:
                pen.write("W")
            else:
                pen.write(from_node)
            for to_node in range(graph.matrix.shape[0]):
                if graph.matrix[from_node][to_node] == 0:
                    continue
                else:
                    if graph.matrix[from_node][to_node] == 2:
                        pen.color('blue')
                    elif graph.matrix[from_node][to_node] == 3:
                        pen.color('red')
                    pen.setposition(graph.node_position[from_node][0] * scale + origin['x'],
                                    graph.node_position[from_node][1] * scale + origin['y'])
                    pen.pendown()
                    pen.setposition(((graph.node_position[from_node][0] + graph.node_position[to_node][0]) * scale / 2) + origin['x'],
                                    ((graph.node_position[from_node][1] + graph.node_position[to_node][1]) * scale / 2) + origin['y'])
                    if graph.matrix[from_node][to_node] != 1:
                        pen.width(2)
                    pen.setposition(graph.node_position[to_node][0] * scale + origin['x'],
                                    graph.node_position[to_node][1] * scale + origin['y'])
                    pen.penup()
                    pen.color('black')
                    pen.width(1)
    
    # Draw rectangular dual of graph
    def draw_rdg(graph,count,pen,to_be_merged_vertices):
        pen.width(1.5)
        pen.color('white')
        pen.hideturtle()
        pen.penup()
        width= np.amax(graph.room_width)
        height = np.amax(graph.room_height)
        if(width == 0):
            width = 1
        if(height == 0):
            height = 1
        if(width < height):
            width = height
        print(width)
        print(height)
        scale = 100*(math.exp(-0.30*width+math.log(0.8)) + 0.1)
        print(scale)
        # origin = {'x': graph.origin, 'y': -550}
        dim =[0,0]
        origin = {'x': graph.origin - 400, 'y': -100}
        for i in range(graph.room_x.shape[0]):
            if i not in to_be_merged_vertices and graph.node_color[i] != "#FF4C4C":
                pen.color('white')
                # pen.color(graph.node_color[i])
            else:
                pen.color(graph.node_color[i])
            if graph.room_width[i] == 0 or i in graph.biconnected_vertices:
                continue
            # print(graph.node_color[i])
            pen.fillcolor(graph.node_color[i])
            pen.begin_fill()
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])
    
            pen.pendown()
            
            pen.setposition((graph.room_x_bottom_left[i]) * scale + origin['x'],
                                graph.room_y[i] * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x_bottom_right[i]) * scale + origin['x'],
                                graph.room_y[i] * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y[i]) * scale + origin['y'])
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y_right_bottom[i]) * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y_right_top[i]) * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            pen.setposition((graph.room_x_top_right[i]) * scale + origin['x'],
                                (graph.room_y[i] + graph.room_height[i]) * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x_top_left[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y_left_top[i]) * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y_left_bottom[i]) * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])
            pen.penup()
            pen.end_fill()
            pen.color('black')
            if(i not in to_be_merged_vertices):
                pen.setposition(((2 * graph.room_x[i] ) * scale / 2) + origin['x'] + 5,
                                ((2 * graph.room_y[i] + graph.room_height[i]) * scale / 2) + origin['y'])
                pen.write(graph.room_names[i])
                pen.penup()
            x_index = int(np.where(graph.room_x == np.min(graph.room_x))[0][0])
            y_index = int(np.where(graph.room_y == np.max(graph.room_y))[0][0])
            pen.setposition((graph.room_x[x_index]) * scale + origin['x'],(graph.room_y[y_index] + graph.room_height[y_index]) * scale + origin['y'] + 200)
            
            
            pen.write(count,font=("Arial", 20, "normal"))
            pen.penup()
            if(graph.room_x[i] + graph.room_width[i]> dim[0] ):
                dim[0] = graph.room_x[i] + graph.room_width[i]
            if(graph.room_y[i] + graph.room_height[i]> dim[1] ):
                dim[1] = graph.room_y[i] + graph.room_height[i]
            graph.origin+= np.amax(graph.room_width)
        
        pen.setposition(0* scale + origin['x'], 0 * scale + origin['y'])
        pen.pendown()
        pen.setposition(dim[0]* scale + origin['x'], 0 * scale + origin['y'])
        pen.setposition(dim[0]* scale + origin['x'], dim[1]* scale + origin['y'])
        pen.setposition(0* scale + origin['x'], dim[1]* scale + origin['y'])
        pen.setposition(0* scale + origin['x'], 0 * scale + origin['y'])
        pen.penup()
        
        value = 1
        if(len(graph.area) != 0):
            pen.setposition(origin['x'], origin['y']-100)
            pen.write('      Area' ,font=("Arial", 24, "normal"))
            for i in range(0,len(graph.area)):
                
                pen.setposition(origin['x'], origin['y']-100-value*30)
                pen.write('Room '+ str(i) + ': '+ str(graph.area[i]),font=("Arial", 24, "normal"))
                pen.penup()
                value+=1
    
    def draw_rfp(graph,pen,count):
        pen.width(4)
        pen.color('black')
        pen.hideturtle()
        pen.penup()
        # scale = 75
        scale = 20
        origin = {'x': graph.origin, 'y': -400}
        # print(graph.room_x)
        # print(graph.room_y)
        for i in range(graph.room_x.shape[0]):
            if graph.room_width[i] == 0:
                continue
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])
            pen.pendown()
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                            graph.room_y[i] * scale + origin['y'])
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                            (graph.room_y[i] + graph.room_height[i]) * scale + origin['y'])
            pen.setposition(graph.room_x[i] * scale + origin['x'],
                            (graph.room_y[i] + graph.room_height[i]) * scale + origin['y'])
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])
            pen.penup()
            pen.setposition(((2 * graph.room_x[i] ) * scale / 2) + origin['x'] + 5,
                                ((2 * graph.room_y[i] + graph.room_height[i]) * scale / 2) + origin['y'])
            pen.write(graph.room_names[i].get())
            pen.setposition(((2 * graph.room_x[i] + graph.room_width[i]) * scale / 2 - scale/2) + origin['x'],
                            ((2 * graph.room_y[i] + graph.room_height[i]) * scale / 2 - scale/2) + origin['y'])
            pen.write('( ' + str(round(graph.room_height[i],2)) + ' x ' + str(round(graph.room_width[i],2)) + ' )',font = ('Times',7))
            pen.penup()
            x_index = int(np.where(graph.room_x == np.min(graph.room_x))[0][0])
            y_index = int(np.where(graph.room_y == np.max(graph.room_y))[0][0])
            pen.setposition((graph.room_x[x_index]) * scale + origin['x'],(graph.room_y[y_index] + graph.room_height[y_index]) * scale + origin['y'] + 200)
            pen.write(count,font=("Arial", 20, "normal"))
            pen.penup() 
    
    def get_rectangle_coordinates(graph,to_be_merged_vertices,rdg_vertices):
        for i in range(0,graph.north):
            graph.room_x_bottom_left[i] = graph.room_x[i]
            graph.room_x_bottom_right[i] = graph.room_x[i]
            graph.room_x_top_left[i] = graph.room_x[i]+graph.room_width[i]
            graph.room_x_top_right[i] = graph.room_x[i]+graph.room_width[i]
            graph.room_y_right_bottom[i] = graph.room_y[i]
            graph.room_y_right_top[i] = graph.room_y[i]
            graph.room_y_left_bottom[i] = graph.room_y[i] + graph.room_height[i]
            graph.room_y_left_top[i] = graph.room_y[i] + graph.room_height[i]
        
        for i in range(0,len(to_be_merged_vertices)):
            vertices = [rdg_vertices[i],to_be_merged_vertices[i]]
            get_direction(graph,vertices)
    
    
    def get_direction(graph,vertices):
        if(graph.room_y[vertices[0]] + graph.room_height[vertices[0]] - graph.room_y[vertices[1]] < 0.000001):
            if graph.room_x[vertices[0]]>graph.room_x[vertices[1]]:
                graph.room_x_top_left[vertices[0]]= graph.room_x[vertices[0]]
                graph.room_x_bottom_left[vertices[1]]= graph.room_x[vertices[0]]
            else:
                graph.room_x_top_left[vertices[0]] = graph.room_x[vertices[1]]
                graph.room_x_bottom_left[vertices[1]]=graph.room_x[vertices[1]]
            if graph.room_x[vertices[0]]+graph.room_width[vertices[0]]<graph.room_x[vertices[1]]+graph.room_width[vertices[1]]:
                graph.room_x_top_right[vertices[0]] = graph.room_x[vertices[0]] + graph.room_width[vertices[0]]
                graph.room_x_bottom_right[vertices[1]] = graph.room_x[vertices[0]] + graph.room_width[vertices[0]]
            else:
                graph.room_x_top_right[vertices[0]] = graph.room_x[vertices[1]]+ graph.room_width[vertices[1]]
                graph.room_x_bottom_right[vertices[1]]=graph.room_x[vertices[1]]+ graph.room_width[vertices[1]]
        elif(graph.room_y[vertices[0]] - graph.room_y[vertices[1]] - graph.room_height[vertices[1]] < 0.000001):
            if graph.room_x[vertices[0]]>graph.room_x[vertices[1]]:
                graph.room_x_bottom_left[vertices[0]]= graph.room_x[vertices[0]]
                graph.room_x_top_left[vertices[1]] = graph.room_x[vertices[0]]
            else:
                graph.room_x_bottom_left[vertices[0]]= graph.room_x[vertices[1]]
                graph.room_x_top_left[vertices[1]] = graph.room_x[vertices[1]]
            if graph.room_x[vertices[0]]+graph.room_width[vertices[0]]<graph.room_x[vertices[1]]+graph.room_width[vertices[1]]:
                graph.room_x_bottom_right[vertices[0]]= graph.room_x[vertices[0]] + graph.room_width[vertices[0]]
                graph.room_x_top_right[vertices[1]] = graph.room_x[vertices[0]] + graph.room_width[vertices[0]]
            else:
                graph.room_x_bottom_right[vertices[0]]= graph.room_x[vertices[1]]+ graph.room_width[vertices[1]]
                graph.room_x_top_right[vertices[1]] = graph.room_x[vertices[1]]+ graph.room_width[vertices[1]]
        elif(graph.room_x[vertices[0]] + graph.room_width[vertices[0]] - graph.room_x[vertices[1]]<  0.000001):
            if graph.room_y[vertices[0]]>graph.room_y[vertices[1]]:
                graph.room_y_right_bottom[vertices[0]]=graph.room_y[vertices[0]]
                graph.room_y_left_bottom[vertices[1]]=graph.room_y[vertices[0]]
            else:
                graph.room_y_right_bottom[vertices[0]]= graph.room_y[vertices[1]]
                graph.room_y_left_bottom[vertices[1]]= graph.room_y[vertices[1]]
            if graph.room_y[vertices[0]]+graph.room_height[vertices[0]]<graph.room_y[vertices[1]]+graph.room_height[vertices[1]]:
                graph.room_y_right_top[vertices[0]]=graph.room_y[vertices[0]] + graph.room_height[vertices[0]]
                graph.room_y_left_top[vertices[1]]=graph.room_y[vertices[0]] + graph.room_height[vertices[0]]
            else:
                graph.room_y_right_top[vertices[0]]=graph.room_y[vertices[1]]+ graph.room_height[vertices[1]]
                graph.room_y_left_top[vertices[1]]= graph.room_y[vertices[1]]+ graph.room_height[vertices[1]]
        elif(graph.room_x[vertices[0]] - graph.room_x[vertices[1]] - graph.room_width[vertices[1]]< 0.000001):
            if graph.room_y[vertices[0]]>graph.room_y[vertices[1]]:
                graph.room_y_left_bottom[vertices[0]]= graph.room_y[vertices[0]]
                graph.room_y_right_bottom[vertices[1]]= graph.room_y[vertices[0]]
            else:
                graph.room_y_left_bottom[vertices[0]]=graph.room_y[vertices[1]]
                graph.room_y_right_bottom[vertices[1]]=graph.room_y[vertices[1]]
            if graph.room_y[vertices[0]]+graph.room_height[vertices[0]]<graph.room_y[vertices[1]]+graph.room_height[vertices[1]]:
                graph.room_y_left_top[vertices[0]]=graph.room_y[vertices[0]] + graph.room_height[vertices[0]]
                graph.room_y_right_top[vertices[1]]=graph.room_y[vertices[0]] + graph.room_height[vertices[0]]
            else:
                graph.room_y_left_top[vertices[0]]=graph.room_y[vertices[1]]+ graph.room_height[vertices[1]]
                graph.room_y_right_top[vertices[1]]=graph.room_y[vertices[1]]+ graph.room_height[vertices[1]]
    
    def construct_rdg(graph,to_be_merged_vertices,rdg_vertices):
        graph.t1_matrix = None
        graph.t2_matrix = None
        graph.t1_longest_distance = [-1] * (graph.west + 1)
        graph.t2_longest_distance = [-1] * (graph.west + 1)
        graph.t1_longest_distance_value = -1
        graph.t2_longest_distance_value = -1
        graph.n_s_paths = []
        graph.w_e_paths = []
    
        graph.room_x = np.zeros(graph.west - 3)
        graph.room_y = np.zeros(graph.west - 3)
        graph.room_height = np.zeros(graph.west - 3)
        graph.room_width = np.zeros(graph.west - 3)
        graph.room_x_bottom_right = np.zeros(graph.west - 3)
        graph.room_x_bottom_left = np.zeros(graph.west - 3)
        graph.room_x_top_right = np.zeros(graph.west - 3)
        graph.room_x_top_left = np.zeros(graph.west - 3)
        graph.room_y_right_top = np.zeros(graph.west - 3)
        graph.room_y_left_top = np.zeros(graph.west - 3)
        graph.room_y_right_bottom = np.zeros(graph.west - 3)
        graph.room_y_left_bottom = np.zeros(graph.west - 3)
        dual.populate_t1_matrix(graph)
        dual.populate_t2_matrix(graph)
        get_dimensions(graph)
        get_rectangle_coordinates(graph,to_be_merged_vertices,rdg_vertices)
    
    def construct_rfp(G,hor_dgph,to_be_merged_vertices,rdg_vertices):
        G.t1_matrix = None
        G.t2_matrix = None
        G.t1_longest_distance = [-1] * (G.west + 1)
        G.t2_longest_distance = [-1] * (G.west + 1)
        G.t1_longest_distance_value = -1
        G.t2_longest_distance_value = -1
        G.n_s_paths = []
        G.w_e_paths = []
    
        G.room_x = np.zeros(G.west - 3)
        G.room_y = np.zeros(G.west - 3)
        # G.room_height = np.zeros(G.west - 3)
        # G.room_width = np.zeros(G.west - 3)
        dual.populate_t1_matrix(G)
        dual.populate_t2_matrix(G)
        dual.get_coordinates(G,hor_dgph)
        get_rectangle_coordinates(G,to_be_merged_vertices,rdg_vertices)
    
    def get_dimensions(graph):
        for node in range(graph.matrix.shape[0]):
            if node in [graph.north, graph.east, graph.south, graph.west]:
                continue
            row, col = np.where(graph.t1_matrix[1:-1] == node)
            if row.shape[0] == 0:#remove this later
                continue
            counts = np.bincount(row)
            max_row = np.argmax(counts)
            indexes, = np.where(row == max_row)
            graph.room_x[node] = col[indexes[0]]
            graph.room_width[node] = col[indexes[-1]] - col[indexes[0]] + 1
    
    
            row, col = np.where(graph.t2_matrix[:, 1:-1] == node)
            counts = np.bincount(col)
            max_col = np.argmax(counts)
            indexes, = np.where(col == max_col)
            graph.room_y[node] = row[indexes[0]]
            graph.room_height[node] = row[indexes[-1]] - row[indexes[0]] + 1
    
    def draw_rdg_circulation(graph,count,pen,to_be_merged_vertices,orig):
        pen.width(1.5)
        # pen.color('white')
        pen.hideturtle()
        pen.penup()
        width= np.amax(graph.room_width)
        height = np.amax(graph.room_height)
        if(width == 0):
            width = 1
        if(height == 0):
            height = 1
        if(width < height):
            width = height
        print(width)
        print(height)
        scale = 70*(math.exp(-0.30*width+math.log(0.8)) + 0.1)
        print(scale)
        # origin = {'x': graph.origin, 'y': -550}
        dim =[0,0]
        origin = {'x': graph.origin - 400, 'y': -100}
        for i in range(graph.room_x.shape[0]):
            if(i not in to_be_merged_vertices):
                pen.color('white')
            else:
                pen.color(graph.node_color[i])
            if graph.room_width[i] == 0 or i in graph.biconnected_vertices:
                continue
            pen.fillcolor(graph.node_color[i])
            pen.begin_fill()
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])
    
            pen.pendown()
            
            pen.setposition((graph.room_x_bottom_left[i]) * scale + origin['x'],
                                graph.room_y[i] * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x_bottom_right[i]) * scale + origin['x'],
                                graph.room_y[i] * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y[i]) * scale + origin['y'])
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y_right_bottom[i]) * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y_right_top[i]) * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            pen.setposition((graph.room_x_top_right[i]) * scale + origin['x'],
                                (graph.room_y[i] + graph.room_height[i]) * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x_top_left[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y_left_top[i]) * scale + origin['y'])
            
            pen.penup()
            
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y_left_bottom[i]) * scale + origin['y'])
            
            pen.pendown()
            
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])
            pen.penup()
            pen.end_fill()
            if(i not in to_be_merged_vertices):
                pen.setposition(((2 * graph.room_x[i] ) * scale / 2) + origin['x'] + 5,
                                ((2 * graph.room_y[i] + graph.room_height[i]) * scale / 2) + origin['y'])
                pen.write(graph.room_names[i].get())
                pen.penup()
            x_index = int(np.where(graph.room_x == np.min(graph.room_x))[0][0])
            y_index = int(np.where(graph.room_y == np.max(graph.room_y))[0][0])
            pen.setposition((graph.room_x[x_index]) * scale + origin['x'],(graph.room_y[y_index] + graph.room_height[y_index]) * scale + origin['y'] + 200)
            
            pen.color('black')
            pen.write(count,font=("Arial", 20, "normal"))
            pen.penup()
            if(graph.room_x[i] + graph.room_width[i]> dim[0] ):
                dim[0] = graph.room_x[i] + graph.room_width[i]
            if(graph.room_y[i] + graph.room_height[i]> dim[1] ):
                dim[1] = graph.room_y[i] + graph.room_height[i]
            graph.origin+= np.amax(graph.room_width)
        
        pen.setposition(0* scale + origin['x'], 0 * scale + origin['y'])
        pen.pendown()
        pen.setposition(dim[0]* scale + origin['x'], 0 * scale + origin['y'])
        pen.setposition(dim[0]* scale + origin['x'], dim[1]* scale + origin['y'])
        pen.setposition(0* scale + origin['x'], dim[1]* scale + origin['y'])
        pen.setposition(0* scale + origin['x'], 0 * scale + origin['y'])
        pen.penup()
        
        value = 1
        if(len(graph.area) != 0):
            pen.setposition(origin['x'], origin['y']-100)
            pen.write('      Area' ,font=("Arial", 24, "normal"))
            for i in range(0,len(graph.area)):
                
                pen.setposition(origin['x'], origin['y']-100-value*30)
                pen.write('Room '+ str(i) + ': '+ str(graph.area[i]),font=("Arial", 24, "normal"))
                pen.penup()
                value+=1
    ##----- End drawing.py -------------------------------------------------------##
    return locals()

@modulize('contraction')
def _contraction(__name__):
    ##----- Begin contraction.py -------------------------------------------------##
    import networkx as nx 
    import numpy as np 
    
    def initialize_degrees(graph):
        """initializes the degrees array from the adjacency matrix of the PTPG"""
        graph.degrees = [np.count_nonzero(graph.matrix[node]) for node in range(graph.node_count)]
    
    def initialize_good_vertices(graph):
        """initializes the good vertices array by checking each node if its good or not"""
        graph.good_vertices = []
        for node in range(graph.matrix.shape[0]):
            if is_good_vertex(graph,node):
                graph.good_vertices.append(node)
    
    def is_good_vertex(graph, node):
        """
        checks if the node is good vertex or not
        Definitions:
        1) Light vertex: vertex whose degree <= 19
        2) Heavy vertex: vertex whose degree >= 20
        3) Degree 5 good vertex: (vertex who has degree 5) and (has 0 or 1  heavy neighbours)
        4) Degree 4 good vertex: (vertex who has degree 4) and
                                 ((has 0 or 1 heavy neighbour) or (has 2 heavy neighbours which are not adjacent))
        5) Good vertex: Degree 4 good vertex or Degree 5 good vertex
        Note: We do not want any of the 4 boundary NESW vertices to be a good vertex since we never want to contract
            any edge connected to these vertices. LookUp: Assusmption 1 for detailed reason
        """
    
        if node not in [graph.north, graph.east, graph.south, graph.west]:
            if graph.degrees[node] == 5:
                heavy_neighbour_count = 0
                neighbours, = np.where(graph.matrix[node] == 1)
                for neighbour in neighbours:  # iterating over neighbours and checking if any of them is heavy vertex
                    if graph.degrees[neighbour] >= 20:
                        heavy_neighbour_count += 1
                if heavy_neighbour_count <= 1:
                    return True  # satisfies all conditions for degree 5 good vertex
    
            elif graph.degrees[node] == 4:
                heavy_neighbours = []
                neighbours, = np.where(graph.matrix[node] == 1)
                for neighbour in neighbours:  # iterating over neighbours and checking if any of them is heavy vertex
                    if graph.degrees[neighbour] >= 20:
                        heavy_neighbours.append(neighbour)
                if (len(heavy_neighbours) <= 1) or (
                        len(heavy_neighbours) == 2 and graph.matrix[heavy_neighbours[0]][heavy_neighbours[1]] != 1):
                    return True  # satisfies all conditions for degree 4 good ertex
        return False
    
    def get_contractible_neighbour(graph, v):
        v_nbr, = np.where(graph.matrix[v] == 1)
        # checking if any of neighbors of the good vertex v is contractible
        # by lemma we will find one but it can be one of nesw so we need to ignore this v
        for u in v_nbr:
            if u in [graph.north, graph.east, graph.south, graph.west]:
                continue
            contractible = True
            u_nbr, = np.where(graph.matrix[u] == 1)
            y_and_z = np.intersect1d(v_nbr, u_nbr, assume_unique=True)
            if len(y_and_z) != 2:
                print("Input graph might contain a complex triangle")
            for x in v_nbr:
                if x in y_and_z or x == u:
                    continue
                x_nbr, = np.where(graph.matrix[x] == 1)
                intersection = np.intersect1d(x_nbr, u_nbr, assume_unique=True)
                for node in intersection:
                    if node not in y_and_z and node != v:
                        contractible = False
                        break
                if not contractible:
                    break
            if contractible:
                return u, y_and_z
        return -1, []
    
    def update_adjacency_matrix(graph, v, u):
        graph.node_position[u][0] = (graph.node_position[u][0] + graph.node_position[v][0]) / 2
        graph.node_position[u][1] = (graph.node_position[u][1] + graph.node_position[v][1]) / 2
        v_nbr, = np.where(graph.matrix[v] == 1)
        for node in v_nbr:
            graph.matrix[v][node] = 0
            graph.matrix[node][v] = 0
            if node != u:
                graph.matrix[node][u] = 1
                graph.matrix[u][node] = 1
    
    def update_good_vertices(graph, v, u, y_and_z):
        graph.degrees[u] += graph.degrees[v] - 4
        graph.degrees[y_and_z[0]] -= 1
        graph.degrees[y_and_z[1]] -= 1
        graph.degrees[v] = 0
        check(graph,u)
        check(graph,y_and_z[0])
        check(graph,y_and_z[1])
    
    def check(graph,node):
        if is_good_vertex(graph,node) and (node not in graph.good_vertices):
            graph.good_vertices.append(node)
        elif (not is_good_vertex(graph,node)) and (node in graph.good_vertices):
            graph.good_vertices.remove(node)
        
    
    def contract(graph):
        attempts = len(graph.good_vertices)
        while attempts > 0:
            v = graph.good_vertices.pop(0)
            u, y_and_z = get_contractible_neighbour(graph,v)
            if u == -1:
                graph.good_vertices.append(v)
                attempts -= 1
                continue
            graph.contractions.append({'v': v, 'u': u, 'y_and_z': y_and_z, 'v_nbr': np.where(graph.matrix[v] == 1)[0]})
            update_adjacency_matrix(graph,v, u)
            update_good_vertices(graph,v, u, y_and_z)
            graph.node_count -= 1
            graph.edge_count -= 3
            return v, u
        return -1, -1
    
    
    
    
    
    
    ##----- End contraction.py ---------------------------------------------------##
    return locals()

@modulize('expansion')
def _expansion(__name__):
    ##----- Begin expansion.py ---------------------------------------------------##
    import networkx as nx 
    import numpy as np
    import operations as opr 
    
    def get_trivial_rel(graph):
        for node in range(graph.matrix.shape[0]):
            if graph.matrix[graph.north][node] == 1 and node not in [graph.east, graph.west]:
                graph.matrix[node][graph.north] = 2
                graph.matrix[graph.north][node] = 0
    
                graph.matrix[graph.south][node] = 2
                graph.matrix[node][graph.south] = 0
    
                graph.matrix[node][graph.east] = 3
                graph.matrix[graph.east][node] = 0
    
                graph.matrix[graph.west][node] = 3
                graph.matrix[node][graph.west] = 0
    
    def expand(graph):
        contraction = graph.contractions.pop()
        case = get_case(graph,contraction)
        o = contraction['u']
        v = contraction['v']
        case(graph,o, v, contraction['y_and_z'][0], contraction['y_and_z'][1], contraction['v_nbr'])
        graph.node_position[o][0] = 2 * graph.node_position[o][0] - graph.node_position[v][0]
        graph.node_position[o][1] = 2 * graph.node_position[o][1] - graph.node_position[v][1]
    
    def get_case(graph, contraction):
        o = contraction['u']
        y_and_z = contraction['y_and_z']
        y = y_and_z[0]
        z = y_and_z[1]
        if graph.matrix[o][y] == 2:
            if graph.matrix[o][z] == 3:
                # print("o->y : T1, o->z : T2, caseA")
                return caseA
            elif graph.matrix[o][z] == 2:
                temp = y
                while(temp!=z):
                    label = opr.get_ordered_neighbour_label(graph,o, temp, clockwise=False)
                    temp = opr.get_ordered_neighbour(graph,o,temp,False)
                    if(label == 3):
                        y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                        break
                # print("o->y : T1, o->z : T1, caseB")
                return caseB
            elif graph.matrix[z][o] == 3:
                # print("o->y : T1, z->o : T2, caseD")
                return caseD
            elif graph.matrix[z][o] == 2:
                # print("o->y : T1, z->o : T1, caseF")
                return caseF
            else:
                print("ERROR")
    
        if graph.matrix[y][o] == 2:
            if graph.matrix[o][z] == 3:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                # print("y->o : T1, o->z : T2, caseE")
                return caseE
            elif graph.matrix[o][z] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                # print("y->o : T1, o->z : T1, caseF")
                return caseF
            elif graph.matrix[z][o] == 3:
                # print("y->o : T1, z->0 : T2, caseH")
                return caseH
            elif graph.matrix[z][o] == 2:
                temp = y
                while(temp!=z):
                    label = opr.get_ordered_neighbour_label(graph,o, temp, clockwise=False)
                    temp = opr.get_ordered_neighbour(graph,o,temp,False)
                    if(label == 3):
                        y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                        break
                # print("y->o : T1, z->o : T1, caseI")
                return caseI
            else:
                print("ERROR")
                
        if graph.matrix[o][y] == 3:
            if graph.matrix[o][z] == 3:
                temp = y
                while(temp!=z):
                    label = opr.get_ordered_neighbour_label(graph,o, temp, clockwise=False)
                    temp = opr.get_ordered_neighbour(graph,o,temp,False)
                    if(label == 2):
                        y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                        break
                # print("o->y : T2, o->z : T2, caseC")
                return caseC
            elif graph.matrix[o][z] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                # print("o->y : T2,  o->z : T1, caseA swapped")
                return caseA
            elif graph.matrix[z][o] == 3:
                # print("o->y : T2, z->o : T2, caseG")
                return caseG
            elif graph.matrix[z][o] == 2:
                # print("o->y : T2, z->o : T1, caseE")
                return caseE
            else:
                print("ERROR")
    
        if graph.matrix[y][o] == 3:
            if graph.matrix[o][z] == 3:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                # print("y->o : T2, o->z : T2, caseG")
                return caseG
            elif graph.matrix[o][z] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                # print("y->o : T2,  o->z : T1, caseD")
                return caseD
            elif graph.matrix[z][o] == 3:
                temp = y
                while(temp!=z):
                    label = opr.get_ordered_neighbour_label(graph,o, temp, clockwise=False)
                    temp = opr.get_ordered_neighbour(graph,o,temp,False)
                    if(label == 2):
                        y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                        break
                # print("y->o : T2,  z->o : T2, caseJ")
                return caseJ
            elif graph.matrix[z][o] == 2:
                y_and_z[0], y_and_z[1] = y_and_z[1], y_and_z[0]
                # print("y->o : T2,  z->o : T1, caseH")
                return caseH
            else:
                print("ERROR")
    
    def handle_original_u_nbrs(graph, o, v, y, z, v_nbr):
        for alpha in v_nbr:
            if alpha != y and alpha != z and alpha != o:
                if graph.matrix[o][alpha] != 0:
                    graph.matrix[v][alpha] = graph.matrix[o][alpha]
                    graph.matrix[o][alpha] = 0
                if graph.matrix[alpha][o] != 0:
                    graph.matrix[alpha][v] = graph.matrix[alpha][o]
                    graph.matrix[alpha][o] = 0
    
    def caseA(graph, o, v, y, z, v_nbr):
        if opr.get_ordered_neighbour_label(graph,o, y, clockwise=True) == 2:
            if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[y][v] = 3
                graph.matrix[v][z] = 3
                graph.matrix[o][v] = 2
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)     
                graph.matrix[v][y] = 2
                graph.matrix[v][z] = 3
                graph.matrix[v][o] = 2
                graph.matrix[o][y] = 0
                graph.matrix[y][o] = 3
        else:
            if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[v][y] = 2
                graph.matrix[z][v] = 2
                graph.matrix[o][v] = 3
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr) 
                graph.matrix[o][z] = 0
                graph.matrix[z][o] = 2
                graph.matrix[v][o] = 3
                graph.matrix[v][y] = 2
                graph.matrix[v][z] = 3
    
    def caseB(graph, o, v, y, z, v_nbr):
        handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
        graph.matrix[z][v] = 3
        graph.matrix[v][y] = 3
        graph.matrix[o][v] = 2 
        
    
    def caseC(graph, o, v, y, z, v_nbr):
        handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
        graph.matrix[y][v] = 2
        graph.matrix[v][z] = 2
        graph.matrix[o][v] = 3
    
    def caseD(graph, o, v, y, z, v_nbr):
        if opr.get_ordered_neighbour_label(graph,o, y, clockwise=False) == 2:
            if opr.get_ordered_neighbour(graph,o, y, False) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[v][y] = 3
                graph.matrix[z][v] = 3
                graph.matrix[o][v] = 2
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[o][y] = 3
                graph.matrix[v][y] = 2
                graph.matrix[z][v] = 3
                graph.matrix[v][o] = 2
        else:
            if opr.get_ordered_neighbour(graph,o, y, False) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[v][y] = 2
                graph.matrix[z][v] = 2
                graph.matrix[v][o] = 3
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[z][o] = 2
                graph.matrix[z][v] = 3
                graph.matrix[v][y] = 2
                graph.matrix[o][v] = 3
    
    def caseE(graph, o, v, y, z, v_nbr):
        if opr.get_ordered_neighbour_label(graph,o, y, clockwise=True) == 2:
            if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[v][y] = 3
                graph.matrix[z][v] = 3
                graph.matrix[v][o] = 2
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[z][o] = 3
                graph.matrix[z][v] = 2
                graph.matrix[v][y] = 3
                graph.matrix[o][v] = 2
    
        else:
            if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[v][y] = 2
                graph.matrix[z][v] = 2
                graph.matrix[o][v] = 3
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[o][y] = 2
                graph.matrix[v][o] = 3
                graph.matrix[v][y] = 3
                graph.matrix[z][v] = 2
    
    def caseF(graph, o, v, y, z, v_nbr):
        if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
            handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
            graph.matrix[v][y] = 2
            graph.matrix[z][v] = 2
            graph.matrix[o][v] = 3
        else:
            handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
            graph.matrix[v][y] = 2
            graph.matrix[z][v] = 2
            graph.matrix[v][o] = 3
    
    def caseG(graph, o, v, y, z, v_nbr):
        if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
            handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
            graph.matrix[v][y] = 3
            graph.matrix[z][v] = 3
            graph.matrix[v][o] = 2
        else:
            handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
            graph.matrix[v][y] = 3
            graph.matrix[z][v] = 3
            graph.matrix[o][v] = 2
    
    def caseH(graph, o, v, y, z, v_nbr):
        if opr.get_ordered_neighbour_label(graph,o, y, clockwise=True) == 2:
            if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[v][y] = 3
                graph.matrix[z][v] = 3
                graph.matrix[v][o] = 2
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[y][o] = 0
                graph.matrix[o][y] = 3
                graph.matrix[y][v] = 2
                graph.matrix[z][v] = 3
                graph.matrix[o][v] = 2
        else:
            if opr.get_ordered_neighbour(graph,o, y, True) in v_nbr:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[y][v] = 2
                graph.matrix[v][z] = 2
                graph.matrix[v][o] = 3
            else:
                handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
                graph.matrix[z][o] = 0
                graph.matrix[o][z] = 2
                graph.matrix[y][v] = 2
                graph.matrix[z][v] = 3
                graph.matrix[o][v] = 3 
    
    def caseI(graph, o, v, y, z, v_nbr):
        handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
        graph.matrix[y][v] = 3
        graph.matrix[v][z] = 3
        graph.matrix[v][o] = 2
    
    def caseJ(graph, o, v, y, z, v_nbr):
        handle_original_u_nbrs(graph,o, v, y, z, v_nbr)
        graph.matrix[v][y] = 2
        graph.matrix[z][v] = 2
        graph.matrix[v][o] = 3
    
    ##----- End expansion.py -----------------------------------------------------##
    return locals()

@modulize('flippable')
def _flippable(__name__):
    ##----- Begin flippable.py ---------------------------------------------------##
    import numpy as np
    import operations as opr
    
    def get_flippable_edges(graph,matrix):
    	edges = []
    	for i in range(0,matrix.shape[0]):
    	    for j in range(0,matrix.shape[1]):
    	        if(matrix[i,j]!=0):
    	            if(matrix[i,j]==2):
    	                edges.append([i,j])
    	            elif(matrix[i,j]==3):
    	                edges.append([i,j])
    	flippable_edge = []
    	for edge in edges:
    	    if(edge[0]> graph.original_node_count or edge[1]> graph.original_node_count):
    	        continue
    	    x_nbr = np.where(graph.user_matrix[edge[0]] != 0)[0]
    	    y_nbr = np.where(graph.user_matrix[edge[1]] != 0)[0]
    	    intersection = np.intersect1d(x_nbr, y_nbr, assume_unique=True)
    	    if(matrix[edge[0],intersection[0]] == 3 or matrix[intersection[0],edge[0]] == 3 ):
    	        if(matrix[edge[0],intersection[1]] == 2 or matrix[intersection[1],edge[0]] == 2 ):
    	             if(matrix[edge[1],intersection[1]] == 3 or matrix[intersection[1],edge[1]] == 3 ):
    	                     if(matrix[edge[1],intersection[0]] == 2 or matrix[intersection[0],edge[1]] == 2 ):
    	                            flippable_edge.append(edge)
    	    elif(matrix[edge[0],intersection[0]] == 2 or matrix[intersection[0],edge[0]] == 2 ):
    	        if(matrix[edge[0],intersection[1]] == 3 or matrix[intersection[1],edge[0]] == 3 ):
    	             if(matrix[edge[1],intersection[1]] == 2 or matrix[intersection[1],edge[1]] == 2 ):
    	                     if(matrix[edge[1],intersection[0]] == 3 or matrix[intersection[0],edge[1]] == 3 ):
    	                            flippable_edge.append(edge)
    
    	return flippable_edge
    
    def get_flippable_vertices(graph,matrix):
    	degrees = np.zeros(matrix.shape[0],int)
    	outer_vertices = opr.get_outer_boundary_vertices(graph)
    	for i in range(0,matrix.shape[0]):
    	    for j in range(0,matrix.shape[1]):
    	        if(matrix[i,j]!=0):
    	            degrees[i]+=1
    	            degrees[j]+=1
    	flippable_vertex = []
    	flippable_vertex_neighbours =[]
    	four_degree_vertex = []
    	for i in range(0,len(degrees)):
    	    if(degrees[i] == 4 and i not in outer_vertices):
    	        four_degree_vertex.append(i)
    
    	for vertex in four_degree_vertex:
    	    neighbors = list(np.where(graph.user_matrix[vertex] != 0)[0])
    	    temp = []
    	    temp.append(neighbors.pop())
    	    while(len(neighbors)!=0):
    	        for vertices in neighbors:
    	            if(graph.user_matrix[temp[len(temp)-1],vertices]==1):
    	                temp.append(vertices)
    	                neighbors.remove(vertices)
    	                break
    	    if(matrix[temp[0],temp[1]] == 3 or matrix[temp[1],temp[0]] == 3 ):
    	        if(matrix[temp[1],temp[2]] == 2 or matrix[temp[2],temp[1]] == 2 ):
    	             if(matrix[temp[2],temp[3]] == 3 or matrix[temp[3],temp[2]] == 3 ):
    	                     if(matrix[temp[3],temp[0]] == 2 or matrix[temp[0],temp[3]] == 2 ):
    	                            flippable_vertex.append(vertex)
    	                            flippable_vertex_neighbours.append(temp)
    	    elif(matrix[temp[0],temp[1]] == 2 or matrix[temp[1],temp[0]] == 2 ):
    	        if(matrix[temp[1],temp[2]] == 3 or matrix[temp[2],temp[1]] == 3 ):
    	             if(matrix[temp[2],temp[3]] == 2 or matrix[temp[3],temp[2]] == 2 ):
    	                     if(matrix[temp[3],temp[0]] == 3 or matrix[temp[0],temp[3]] == 3 ):
    	                            flippable_vertex.append(vertex)
    	                            flippable_vertex_neighbours.append(temp)
    
    	return flippable_vertex,flippable_vertex_neighbours
    
    def resolve_flippable_edge(edge,graph,rel):
    	new_rel = rel.copy()
    	# print(opr.get_ordered_neighbour_label(graph, edge[0], edge[1], True))
    	if(new_rel[edge[0],edge[1]] == 2):
    		if(opr.get_ordered_neighbour_label(graph, edge[0], edge[1], True) == 3):
    			# print("Case A")
    			new_rel[edge[0],edge[1]] = 0
    			new_rel[edge[0],edge[1]] = 3
    		elif(opr.get_ordered_neighbour_label(graph, edge[0], edge[1], True) == 2):
    			# print("Case B")
    			new_rel[edge[0],edge[1]] = 0
    			new_rel[edge[1],edge[0]] = 3
    	elif(new_rel[edge[0],edge[1]] == 3):
    		if(opr.get_ordered_neighbour_label(graph, edge[0], edge[1], True) == 3):
    			# print("Case C")
    			new_rel[edge[0],edge[1]] = 0
    			new_rel[edge[0],edge[1]] = 2
    		elif(opr.get_ordered_neighbour_label(graph, edge[0], edge[1], True) == 2):
    			# print("Case D")
    			new_rel[edge[0],edge[1]] = 0
    			new_rel[edge[1],edge[0]] = 2
    	return new_rel
    
    def resolve_flippable_vertex(vertex,neighbours,graph,rel):
    	new_rel = rel.copy()
    	clockwise_neighbour = opr.get_ordered_neighbour(graph,vertex,neighbours[0],True)
    	if(neighbours[1] != clockwise_neighbour):
    		neighbours.reverse()
    	while(new_rel[vertex,neighbours[0]] != 3):
    		first_element = neighbours.pop(0)
    		neighbours.append(first_element)
    	if(opr.get_ordered_neighbour_label(graph, neighbours[0],vertex,True) == 3):
    		new_rel[vertex,neighbours[0]] = 2
    		new_rel[vertex,neighbours[1]] = 3
    		new_rel[neighbours[1],vertex] = 0
    		new_rel[neighbours[2],vertex] = 2
    		new_rel[vertex,neighbours[3]] = 0
    		new_rel[neighbours[3],vertex] = 3
    	elif(opr.get_ordered_neighbour_label(graph, neighbours[0],vertex,True) == 2):
    		new_rel[vertex,neighbours[0]] = 0
    		new_rel[neighbours[0],vertex] = 2
    		new_rel[neighbours[1],vertex] = 3
    		new_rel[neighbours[2],vertex] = 0
    		new_rel[vertex,neighbours[2]] = 2
    		new_rel[vertex,neighbours[3]] = 3
    	return new_rel
    ##----- End flippable.py -----------------------------------------------------##
    return locals()

@modulize('tablenoscroll')
def _tablenoscroll(__name__):
    ##----- Begin tablenoscroll.py -----------------------------------------------##
    # def _tablenoscroll(__name__):
    ##----- Begin tablenoscroll.py -----------------------------------------------##
    # Author: Miguel Martinez Lopez
    # Version: 0.14
    
    try:
        from Tkinter import Frame, Label, Message, StringVar , Entry
        import Tkconstants 
        import tkinter as tk
    except ImportError:
        from tkinter import Frame, Label, Message, StringVar
        import tkinter.constants
        import tkinter as tk
    
    class Cell(Frame):
        """Base class for cells"""
    
    class Data_Cell(Cell):
        def __init__(self, master, variable, anchor='w', bordercolor=None, borderwidth=1, padx=0, pady=0, background=None, foreground=None, font=None):
            Cell.__init__(self, master, background=background, highlightbackground=bordercolor, highlightcolor=bordercolor, highlightthickness=borderwidth, bd= 0)
    
            self._message_widget = tk.Entry(self, textvariable=variable, font=font, background=background, foreground=foreground,width=15)
            self._message_widget.pack(expand=True, padx=padx, pady=pady, anchor=anchor)
            # try:
            #     self.bind("<Configure>", self._on_configure)
            # except :
            #     pass
    
        def _on_configure(self, event):
            self._message_widget.configure(width=event.width)
    
    class Header_Cell(Cell):
        def __init__(self, master, text, bordercolor=None, borderwidth=1, padx=None, pady=None, background=None, foreground=None, font=None, anchor='c'):
            Cell.__init__(self, master, background=background, highlightbackground=bordercolor, highlightcolor=bordercolor, highlightthickness=borderwidth, bd= 0)
            self._header_label = Label(self, text=text, background=background, foreground=foreground, font=font,width=13)
            self._header_label.pack(padx=padx, pady=pady, expand=True)
            
            if bordercolor is not None:
                separator = Frame(self, height=2, background=bordercolor, bd=0, highlightthickness=0, class_="Separator")
                separator.pack(fill='x', anchor=anchor)
            
    class Table(Frame):
        def __init__(self, master, columns, column_weights=None, column_minwidths=None, height=None, minwidth=20, minheight=20, padx=5, pady=5, cell_font=None, cell_foreground="black", cell_background="white", cell_anchor='w', header_font=None, header_background="white", header_foreground="black", header_anchor='c', bordercolor = "#999999", innerborder=True, outerborder=True, stripped_rows=("#EEEEEE", "white"), on_change_data=None):
            outerborder_width = 1 if outerborder else 0
    
            Frame.__init__(self,master, highlightbackground=bordercolor, highlightcolor=bordercolor, highlightthickness=outerborder_width, bd= 0,width=15)
            self.master = master
            self._cell_background = cell_background
            self._cell_foreground = cell_foreground
            self._cell_font = cell_font
            self._cell_anchor = cell_anchor
            
            self._number_of_rows = 0
            self._number_of_columns = len(columns)
            
            self._stripped_rows = stripped_rows
    
            self._padx = padx
            self._pady = pady
            
            self._bordercolor = bordercolor
            self._innerborder_width= 1 if innerborder else 0
    
            self._data_vars = []
    
            self._columns = columns
    
            for j in range(len(columns)):
                column_name = columns[j]
    
                header_cell = Header_Cell(self, text=column_name, borderwidth=self._innerborder_width, font=header_font, background=header_background, foreground=header_foreground, padx=padx, pady=pady, bordercolor=bordercolor, anchor=header_anchor)
                header_cell.grid(row=0, column=j, sticky='news')
    
            if column_weights is None:
                for j in range(len(columns)):
                    self.grid_columnconfigure(j, weight=1)
            else:
                for j, weight in enumerate(column_weights):
                    self.grid_columnconfigure(j, weight=weight)
    
            if column_minwidths is not None:
                self.update_idletasks()
                for j, minwidth in enumerate(column_minwidths):
                    if minwidth is None:
                        header_cell = self.grid_slaves(row=0, column=j)[0]
                        minwidth = header_cell.winfo_reqwidth()
                    self.grid_columnconfigure(j, minsize=minwidth)
            
    
            if height is not None:
                self._append_n_rows(height)
    
            self._on_change_data = on_change_data
    
        def _append_n_rows(self, n):
            number_of_rows = self._number_of_rows
            number_of_columns = self._number_of_columns
    
            for i in range(number_of_rows+1, number_of_rows+n+1):
                list_of_vars = []
                for j in range(number_of_columns):
                    var = StringVar()
                    list_of_vars.append(var)
    
                    if self._stripped_rows:
                        cell = Data_Cell(self, borderwidth=self._innerborder_width, variable=var, bordercolor=self._bordercolor, padx=self._padx, pady=self._pady, background=self._stripped_rows[(i+1)%2], foreground=self._cell_foreground, font=self._cell_font, anchor=self._cell_anchor)
                    else:
                        cell = Data_Cell(self, borderwidth=self._innerborder_width, variable=var, bordercolor=self._bordercolor, padx=self._padx, pady=self._pady, background=self._cell_background, foreground=self._cell_foreground, font=self._cell_font, anchor=self._cell_anchor)
                    cell.grid(row=i, column=j, sticky='news')
    
                self._data_vars.append(list_of_vars)
    
            self._number_of_rows += n
    
        def _pop_n_rows(self, n):
            number_of_rows = self._number_of_rows
            number_of_columns = self._number_of_columns
            for i in range(number_of_rows-n+1, number_of_rows+1):
                for j in range(number_of_columns):
                    self.grid_slaves(row=i, column=j)[0].destroy()
                
                self._data_vars.pop()
            
            self._number_of_rows -= n
    
        def _pop_all(self):
            number_of_rows = self._number_of_rows
            number_of_columns = self._number_of_columns
            for i in range(1,number_of_rows+1):
                for j in range(number_of_columns):
                    self.grid_slaves(row=i, column=j)[0].destroy()
                
            self._data_vars.clear()
            
            self._number_of_rows =0
    
        def set_data(self, data):
            n = len(data)
            m = len(data[0])
    
            number_of_rows = self._number_of_rows
    
            if number_of_rows > n:
                self._pop_n_rows(number_of_rows-n)
            elif number_of_rows < n:
                self._append_n_rows(n-number_of_rows)
    
            for i in range(n):
                for j in range(m):
                    self._data_vars[i][j].set(data[i][j])
    
            if self._on_change_data is not None: self._on_change_data()
    
        def get_data(self):
            number_of_rows = self._number_of_rows
            number_of_columns = self._number_of_columns
            
            data = []
            for i in range(number_of_rows):
                row = []
                row_of_vars = self._data_vars[i]
                for j in range(number_of_columns):
                    cell_data = row_of_vars[j].get()
                    row.append(cell_data)
                
                data.append(row)
            return data
    
        @property
        def number_of_rows(self):
            return self._number_of_rows
    
        @property
        def number_of_columns(self):
            return self._number_of_columns
    
        def row(self, index, data=None):
            number_of_columns = self._number_of_columns
    
            if data is None:
                row = []
                row_of_vars = self._data_vars[index]
    
                for j in range(number_of_columns):
                    row.append(row_of_vars[j].get())
                    
                return row
            else:
                if len(data) != number_of_columns:
                    raise ValueError("data has no %d elements: %s"%(number_of_columns, data))
    
                row_of_vars = self._data_vars[index]
                for j in range(number_of_columns):
                    row_of_vars[index][j].set(data[j])
                    
                if self._on_change_data is not None: self._on_change_data()
    
        def column(self, index, data=None):
            number_of_rows = self._number_of_rows
    
            if data is None:
                column= []
    
                for i in range(number_of_rows):
                    column.append(self._data_vars[i][index].get())
                    
                return column
            else:
                
                if len(data) != number_of_rows:
                    raise ValueError("data has no %d elements: %s"%(number_of_rows, data))
    
                for i in range(self._number_of_columns):
                    self._data_vars[i][index].set(data[i])
    
                if self._on_change_data is not None: self._on_change_data()
    
        def clear(self):
            number_of_rows = self._number_of_rows
            number_of_columns = self._number_of_columns
    
            for i in range(number_of_rows):
                for j in range(number_of_columns):
                    self._data_vars[i][j].set("")
    
            if self._on_change_data is not None: self._on_change_data()
    
        def delete_row(self, index):
            i = index
            while i < self._number_of_rows:
                row_of_vars_1 = self._data_vars[i]
                row_of_vars_2 = self._data_vars[i+1]
    
                j = 0
                while j <self._number_of_columns:
                    row_of_vars_1[j].set(row_of_vars_2[j])
    
                i += 1
    
            self._pop_n_rows(1)
    
            if self._on_change_data is not None: self._on_change_data()
    
        def insert_row(self, data, index='end'):
            self._append_n_rows(1)
    
            if index == 'end':
                index = self._number_of_rows - 1
            
            i = self._number_of_rows-1
            while i > index:
                row_of_vars_1 = self._data_vars[i-1]
                row_of_vars_2 = self._data_vars[i]
    
                j = 0
                while j < self._number_of_columns:
                    row_of_vars_2[j].set(row_of_vars_1[j])
                    j += 1
                i -= 1
    
            list_of_cell_vars = self._data_vars[index]
            for cell_var, cell_data in zip(list_of_cell_vars, data):
                cell_var.set(cell_data)
    
            if self._on_change_data is not None: self._on_change_data()
    
        def cell(self, row, column, data=None):
            """Get the value of a table cell"""
            if data is None:
                return self._data_vars[row][column].get()
            else:
                self._data_vars[row][column].set(data)
                if self._on_change_data is not None: self._on_change_data()
    
        def __getitem__(self, index):
            if isinstance(index, tuple):
                row, column = index
                return self.cell(row, column)
            else:
                raise Exception("Row and column indices are required")
            
        def __setitem__(self, index, value):
            if isinstance(index, tuple):
                row, column = index
                self.cell(row, column, value)
            else:
                raise Exception("Row and column indices are required")
    
        def on_change_data(self, callback):
            self._on_change_data = callback
    
    if __name__ == "__main__":
        try:
            from Tkinter import Tk
        except ImportError:
            from tkinter import Tk
    
        root = Tk()
    
        table = Table(root, ["column A", "column B", "column C"], column_minwidths=[None, None, None])
        table.pack(expand=True, fill='x', padx=10,pady=10)
    
        table.set_data([[1,2,3],[4,5,6], [7,8,9], [10,11,12]])
        table.cell(0,0, " a fdas fasd fasdf asdf asdfasdf asdf asdfa sdfas asd sadf ")
        root.mainloop()
    
    ##----- End tablenoscroll.py -------------------------------------------------##
    
    ##----- End tablenoscroll.py -------------------------------------------------##
    return locals()

@modulize('gui')
def _gui(__name__):
    ##----- Begin gui.py ---------------------------------------------------------##
    import tkinter as tk
    from tkinter import Menu
    from tkinter import Label
    from tkinter import messagebox   
    import tkinter.ttk as ttk
    from tkinter import filedialog
    from tkinter import ALL, EventType
    import turtle
    import os
    import sys
    import json
    # import tablescroll
    import tablenoscroll
    import random 
    import ast
    import networkx as nx
    import matplotlib.pyplot as plt
    import warnings
    from PIL import Image, ImageTk
    import cv2
    done = True
    # col = ["#3b429f","#8043B1","#f5d7e3","#f4a5ae","#a8577e"]
    # col = ["#3b429f","#button","#tk","#cavas","#a8577e"]
    col = ["#788585","#9A8C98","#F2E9E4","#C9ADA7","#e1eaec"]
    font={'font' : ("lato bold",10,"")}
    # reloader = Reloader()
    warnings.filterwarnings("ignore") 
    class ScrollFrame(tk.Frame):
                def __init__(self, parent):
                    super().__init__(parent) # create a frame (self)
    
                    self.canvas = tk.Canvas(self, borderwidth=0, background="#ffffff")          #place canvas on self
                    self.viewPort = tk.Frame(self.canvas, background="#ffffff")                    #place a frame on the canvas, this frame will hold the child widgets 
                    self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview) #place a scrollbar on self 
                    self.canvas.configure(yscrollcommand=self.vsb.set)                          #attach scrollbar action to scroll of canvas
    
                    # self.vsb.pack(side="right", fill="y")                                       #pack scrollbar to right of self
                    # self.canvas.pack(side="left", fill="both", expand=True)                     #pack canvas to left of self and expand to fil
                    self.canvas_window = self.canvas.create_window((4,4), window=self.viewPort, anchor="nw",            #add view port frame to canvas
                                            tags="self.viewPort")
    
                    self.viewPort.bind("<Configure>", self.onFrameConfigure)                       #bind an event whenever the size of the viewPort frame changes.
                    self.canvas.bind("<Configure>", self.onCanvasConfigure)                       #bind an event whenever the size of the viewPort frame changes.
    
                    self.onFrameConfigure(None)                                                 #perform an initial stretch on render, otherwise the scroll region has a tiny border until the first resize
    
                def onFrameConfigure(self, event):                                              
                    '''Reset the scroll region to encompass the inner frame'''
                    self.canvas.configure(scrollregion=self.canvas.bbox("all"))                 #whenever the size of the frame changes, alter the scroll region respectively.
    
                def onCanvasConfigure(self, event):
                    '''Reset the canvas window to encompass inner frame when required'''
                    canvas_width = event.width
                    self.canvas.itemconfig(self.canvas_window, width = canvas_width)            #whenever the size of the canvas changes alter the window region respectively.
    
    class gui_class:
        def __init__(self):
            border_details = {'bg': col[2], 'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}        
            
            self.open = False
            self.command = "hi"
            self.value = []
            self.root =tk.Tk()
            self.open_ret = []
            # self.root.filename = 
            self.root.config(bg=col[2])
            self.textbox = tk.Text
            # self.pen = turtle.Screen()
            # self.pen = turtle.RawTurtle
            # self.pen.screen.bgcolor(col[2])
            self.end= tk.IntVar(self.root)
            self.frame2 = tk.Frame(self.root,bg=col[2])
            self.frame2.grid(row=0,column=1,rowspan=6,sticky='news')
            self.frame5 = tk.Frame(self.root,bg=col[2])
            self.frame5.grid(row=0,column=2,rowspan=3,sticky='news',padx=10)
            self.tablehead = tk.Label(self.frame5,text='Room Info',bg =col[2])
            self.tablehead.pack()
    
            self.app = self.PlotApp(self.frame2,self)
            self.root.state('zoomed')
            self.root.title('Input Graph')
            self.checkvar1 = tk.IntVar()
    
            self.tabledata = []
            self.frame1 = tk.Frame(self.root,bg=col[2])
            self.frame1.grid(row=0,column=0)
            label1 = tk.LabelFrame(self.frame1,text="tools")
            label1.grid(row=0,column=0,pady=10)
            self.frame3 = tk.Frame(self.root,bg=col[2])
            self.frame3.grid(row=1,column=0)
            self.Buttons(self.frame1,self)
            self.menu(self)
            print(self.app.rnames)
            self.root.protocol("WM_DELETE_WINDOW", self.exit)
            self.tbox = self.output_text(self.frame3)
            self.ocan = self.output_canvas(self.frame2)
            self.pen = self.ocan.getpen()
            self.root_window = self.ocan.getroot()   
            self.root.wait_variable(self.end)
            self.graph_ret()
            while((self.value[0] == 0) and done):
                # print(done)
                self.root.wait_variable(self.end)
                self.value=self.app.return_everything()
                tk.messagebox.showinfo("error","The graph is empty , please draw a graph")
    
        class Nodes():
            def __init__(self,id,x,y):
                self.circle_id=id
                self.pos_x=x
                self.pos_y=y
                self.radius=15
                self.adj_list=[]
        class PlotApp(object):
    
            def __init__(self, root,master):
                self.l1 = tk.Label(root,text='Draw a test graph here',bg=col[2])
                self.l1.grid(row=0,column=0)
                self._root = root
                self.radius_circle=15
                self.rnames = []
                self.master = master
                self.command = "hi"
                self.table = tablenoscroll.Table(self.master.frame5,["Index", "Room Name"], column_minwidths=[None, None])
                self.table.pack(padx=10,pady=10)
                self.table.config(bg="#F4A5AE")
                self.createCanvas()
                self.hex_list = []
    
                
            colors = ['#edf1fe','#c6e3f7','#e1eaec','#e5e8f2','#def7fe','#f1ebda','#f3e2c6','#fff2de','#ecdfd6','#f5e6d3','#e3e7c4','#efdbcd','#ebf5f0','#cae1d9','#c3ddd6','#cef0cc','#9ab8c2','#ddffdd','#fdfff5','#eae9e0','#e0dddd','#f5ece7','#f6e6c5','#f4dbdc','#f4daf1','#f7cee0','#f8d0e7','#efa6aa','#fad6e5','#f9e8e2','#c4adc9','#f6e5f6','#feedca','#f2efe1','#fff5be','#ffffdd']
            nodes_data=[]
            id_circle=[]
            name_circle= []
            edge_count=0
            multiple_rfp = 0
            cir=0
            edges=[]
            random_list = []
            connection=[]
    
            def return_everything(self):
                return [len(self.nodes_data),self.edge_count,self.edges,self.command,self.master.checkvar1.get(),[row[1].get() for row in self.table._data_vars],self.hex_list]
    
            def createCanvas(self):
                self.id_circle.clear()
                self.name_circle.clear()
                for i in range(0,100):
                    self.id_circle.append(i)
                for i in range(0,100):
                    self.name_circle.append(str(i))
                self.nodes_data.clear()
                self.edges.clear()
                self.table._pop_all()
                self.edge_count = 0
                # border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}
                self.canvas = tk.Canvas(self._root,bg=col[3], width=1000, height=400)
                self.canvas.grid(column=0,row =1, sticky='nwes')
                self.canvas.bind("<Button-3>",self.addH)
                self.connection=[]
                self.canvas.bind("<Button-1>",self.button_1_clicked) 
                self.ButtonReset = tk.Button(self._root, text="Reset",fg='white',width=10,height=2 ,**font,relief = 'flat',bg=col[1] ,command=self.reset)
                self.ButtonReset.grid(column=0 ,row=1,sticky='n',pady=20,padx=40)
                
                self.instru = tk.Button(self._root, text="Instructions",fg='white',width=10,height=2 , **font ,relief = 'flat', bg=col[1] ,command=self.instructions)
                self.instru.grid(column=0 ,row=1,sticky='wn',pady=22,padx=40)
    
            def instructions(self):
                tk.messagebox.showinfo("Instructions",
                "--------User Instructrions--------\n 1. Draw the input graph. \n 2. Use right mouse click to create a new room. \n 3. left click on one node then left click on another to create an edge between them. \n 4. You can give your own room names by clicking on the room name in the graph or the table on the right. \n 5. After creating a graph you can choose one of the option to create it's corresponding RFP or multiple RFPs with or without dimension. You can also get the corridor connecting all the rooms by selecting 'circultion' or click on 'RFPchecker' to check if RFP exists for the given graph. \n 6. You can also select multiple options .You can also add rooms after creating RFP and click on RFP to re-create a new RFP. \n 7.Reset button is used to clear the input graph. \n 8. Press 'Exit' if you want to close the application or Press 'Restart' if you want to restart the application")
    
            def addH(self, event):
                random_number = random.randint(0,35)
                while(random_number in self.random_list):
                    random_number = random.randint(0,35)
                self.random_list.append(random_number)
                hex_number = self.colors[random_number]
                # print(random_number)
                # print(hex_number)
                self.hex_list.append(hex_number)
                if(len(self.random_list) == 36):
                    self.random_list = []
                x, y = event.x, event.y
                id_node=self.id_circle[0]
                self.id_circle.pop(0)
                node=self.master.Nodes(id_node,x,y)
                self.nodes_data.append(node)
                self.rframe = tk.Frame(self._root,width=20,height=20)
                self.rname= tk.StringVar(self._root)
                self.rnames.append(self.rname)
                self.rname.set(self.name_circle[0])
                self.table.insert_row(list((id_node,self.rname.get())))
                self.name_circle.pop(0)
                # self.rframe.grid(row=0,column=1)
                self.canvas.create_oval(x-self.radius_circle,y-self.radius_circle,x+self.radius_circle,y+self.radius_circle,width=3, fill=hex_number,tag=str(id_node))
                # self.canvas.create_text(x,y-self.radius_circle-9,text=str(id_node),font=("Purisa",14))
                # self.buttonBG = self.canvas.create_rectangle(x-15,y-self.radius_circle-20, x+15,y-self.radius_circle, fill="light grey")
                # self.buttonTXT = self.canvas.create_text(x,y-self.radius_circle-9, text="click")
                # self.rcanframe = self.canvas.create_window(x,y-self.radius_circle-12, window=self.rframe)
                # self.canvas.tag_bind(self.buttonBG, "<Button-1>", self.room_name) ## when the square is clicked runs function "clicked".
                # self.canvas.tag_bind(self.buttonTXT, "<Button-1>", self.room_name) ## same, but for the text.
                # def _on_configure(self, event):
                #     self.entry.configure(width=event.width)
                self.canvas.create_text(x,y-self.radius_circle-9,text=str(id_node),font=("Purisa",14))
                # self.entry = tk.Entry(self.rframe,textvariable=self.table._data_vars[self.id_circle[0]-1][1],relief='flat',justify='c',width=15,bg=col[2])
                # self.entry.bind("<Configure>", _on_configure)
                
                # but =tk.Button(self.rframe)
                # but.grid()
                # self.entry.grid()
                # print(self.rname.get())
            def button_1_clicked(self,event):
                if len(self.connection)==2:
                    self.connection=[]
                if len(self.nodes_data)<=1:
                    tk.messagebox.showinfo("Connect Nodes","Please make 2 or more nodes")
                    return
                x, y = event.x, event.y
                value=self.get_id(x,y)
                if value == -1:
                    return
                else:
                    if value in self.connection:
                        tk.messagebox.showinfo("Connect Nodes","You have clicked on same node. Please try again")
                        return
                    self.connection.append(value)
    
                if len(self.connection)>1:
                    node1=self.connection[0]
                    node2=self.connection[1]
    
                    if node2 not in self.nodes_data[node1].adj_list:
                        self.nodes_data[node1].adj_list.append(node2)
                    if node1 not in self.nodes_data[node2].adj_list:
                        self.nodes_data[node2].adj_list.append(node1)
                        self.edge_count+=1
                    self.edges.append(self.connection)
                    self.connect_circles(self.connection)
    
                # for i in self.nodes_data:
                # 	print("id: ",i.circle_id)
                # 	print("x,y: ",i.pos_x,i.pos_y)
                # 	print("adj list: ",i.adj_list)
                
            def connect_circles(self,connections):
                node1_id=connections[0]
                node2_id=connections[1]
                node1_x=self.nodes_data[node1_id].pos_x
                node1_y=self.nodes_data[node1_id].pos_y
                node2_x=self.nodes_data[node2_id].pos_x
                node2_y=self.nodes_data[node2_id].pos_y
                self.canvas.create_line(node1_x,node1_y,node2_x,node2_y,width=3)
    
            def get_id(self,x,y):
                for i in self.nodes_data:
                    distance=((i.pos_x-x)**2 + (i.pos_y-y)**2)**(1/2)
                    if distance<=self.radius_circle:
                        return i.circle_id
                tk.messagebox.showinfo("Connect Nodes","You have clicked outside all the circles. Please try again")
                return -1
            
            # def pop_node(self,event)
            def reset(self):
                self.canvas.destroy()
                self.createCanvas()
    
        class Buttons:
            def __init__(self,root,master):
                
                button_details={'wraplength':'150','bg':col[1],'fg':'white','font':('lato','14') , 'padx':5 ,'pady':5,'activebackground' : col[2] }
                # b1 = tk.Button(master.frame1,width=15,text='Rectangular Floor Plan',relief='flat',**button_details,command=master.single_floorplan)
                # b1.grid(row=1,column=0,padx=5,pady=5)
                # b2 = tk.Button(master.frame1,width=15, text='Multiple Rectangular Floor Plans',relief='flat',**button_details,command=master.multiple_floorplan)
                # b2.grid(row=2,column=0,padx=5,pady=5)
                # c1 = tk.Checkbutton(master.frame1, text = "Dimensioned",relief='flat',**button_details,selectcolor='#4A4E69',width=13 ,variable = master.checkvar1,onvalue = 1, offvalue = 0)
                # c1.grid(row=3,column=0,padx=5,pady=5)
                # b3 = tk.Button(master.frame1,width=15, text='Circulation',relief='flat',**button_details,command=master.circulation)
                # b3.grid(row=4,column=0,padx=5,pady=5)
                b4 = tk.Button(master.frame1,width=15, text='Check RFP' ,relief='flat',**button_details,command=master.checker)
                b4.grid(row=5,column=0,padx=5,pady=5)
                # b6 = tk.Button(master.frame1,width=15, text='Restart',relief='flat', **button_details,command=master.restart)
                # b6.grid(row=6,column=0,padx=5,pady=5)
                b5 = tk.Button(master.frame1,width=15, text='EXIT',relief='flat', **button_details,command=master.exit)
                b5.grid(row=7,column=0,padx=5,pady=5)
    
        class menu:
            def __init__(self,master):
                root  = master.root
                menubar = tk.Menu(root,bg=col[3])
                menubar.config(background=col[3])
                filemenu = tk.Menu(menubar,bg=col[3], tearoff=2)
                filemenu.add_command(label="New",command=master.restart)
                filemenu.add_command(label="Open",command=master.open_file)
                filemenu.add_command(label="Save",command=master.save_file)
                filemenu.add_command(label="Save as...",command=master.save_file)
                filemenu.add_command(label="Close",command=master.exit)
    
                filemenu.add_separator()
    
                filemenu.add_command(label="Exit", command=master.exit)
                menubar.add_cascade(label="File", menu=filemenu)
                editmenu = tk.Menu(menubar,bg=col[3], tearoff=0)
                editmenu.add_command(label="Undo")
    
                editmenu.add_separator()
    
                editmenu.add_command(label="Cut")
                editmenu.add_command(label="Copy")
                editmenu.add_command(label="Paste")
                editmenu.add_command(label="Delete")
                editmenu.add_command(label="Select All")
    
                menubar.add_cascade(label="Edit", menu=editmenu)
                helpmenu = tk.Menu(menubar,bg=col[3], tearoff=0)
                helpmenu.add_command(label="About...")
                menubar.add_cascade(label="Help", menu=helpmenu)
                
                root.config(menu=menubar)
        
        class output_canvas:
            def __init__(self,root):
                self.root_window = tk.Frame(root,width=1000,height=400)
                self.l1 = tk.Label(self.root_window, text= 'Rectangular Dual')
                # root_window.geometry(str(1366) + 'x' + str(700))
                # self.root_window.resizable(0, 0)
                # self.root_window.grid_columnconfigure(0, weight=1, uniform=1)
                # self.root_window.grid_rowconfigure(0, weight=1)
                # self.root_window.grid(row=2,column=0)
                self.root_window.grid(row=2,column=0,pady=5)
                # border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}
    
                self.canvas = tk.Canvas(self.root_window,bg ='red',width = 1000, height =377)
                # self.canvas.pack()
    
                # self.canvas.scan_mark(0,0)
                # self.canvas.scan_dragto(500,500)
                self.canvas.bind("<MouseWheel>",  self.do_zoom)
                self.canvas.bind('<Button-1>', lambda event: self.canvas.scan_mark(event.x, event.y))
                self.canvas.bind("<B1-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
                self.canvas.grid(row=0, column=0, sticky='nsew', padx=4, pady=10)
    
    
                # self.scroll_x = tk.Scrollbar(self.root_window, orient="horizontal", command=self.canvas.xview)
                # self.scroll_x.grid(row=1, column=0, sticky="ew")
    
                # self.scroll_y = tk.Scrollbar(self.root_window, orient="vertical", command=self.canvas.yview)
                # self.scroll_y.grid(row=0, column=1, sticky="ns")
    
                # self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)
                # self.canvas.configure(scrollregion=self.canvas.bbox("all"))
                self.tscreen = turtle.TurtleScreen(self.canvas)
                self.tscreen.bgcolor(col[3])
                self.pen = turtle.RawTurtle(self.tscreen)
                self.pen.speed(100000)
                # self.canvas.create_rectangle(0,0,100,100,fill='red')
                
            
            def do_zoom(self,event):
                factor = 1.001 ** event.delta
                self.canvas.scale(ALL, event.x, event.y, factor, factor)
    
            def getpen(self):
                return self.pen
    
            def getroot(self):
                return self.root_window
        
        class output_text:
            def __init__(self,root):
                self.textbox = tk.Text(root,bg=col[3],fg='black',relief='flat',height=40,width=30,padx=5,pady=5,**font)
                self.textbox.grid(row=0,column=0 ,padx=10,pady=10)
    
                self.textbox.insert('insert',"\t         Output\n")
    
            def gettext(self):
                return self.textbox    
        
        def graph_ret(self):
            if not self.open:
                self.value = self.app.return_everything()
                self.textbox = self.tbox.gettext()
            else:
                self.value = self.open_ret.copy()
                self.textbox = self.tbox.gettext()
    
        def single_floorplan(self):
            self.app.command="single"
            self.command = "single"
            self.end.set(self.end.get()+1)
            self.root.state('zoomed')
            # root.destroy()
    
        def multiple_floorplan(self):
            self.app.command="multiple"
            self.command = "multiple"
            self.end.set(self.end.get()+1)
            # root.destroy()
        
        def circulation(self):
            self.app.command="circulation"
            self.command = "circulation"
            self.end.set(self.end.get()+1)
        
        def checker(self):
            self.app.command="checker"
            self.command = "checker"
            self.end.set(self.end.get()+1)
    
        def restart(self):
            os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
        
        def exit(self):
            global done
    
            self.app.command="end"
            self.command = "end"
            self.end.set(self.end.get()+1)
            done = False
            self.root.destroy()
    
            
            
            
            # return self.value , self.root , self.textbox , self.pen ,self.end
    
        def open_file(self):
            self.file = filedialog.askopenfile(mode='r',defaultextension=".txt",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")))
            f = self.file.read()
            print(f)
            # print("hjio")
            fname = self.file.name
            print(fname)
            fname = fname[:-3]
            fname+="png"
            print(fname)
            self.open_ret = ast.literal_eval(f)
            print(self.open_ret)
            self.graph = nx.Graph()
            self.graph.add_edges_from(self.open_ret[2])
            nx.draw_planar(self.graph)
            # plt.show()
    
            plt.savefig(fname)
            img = Image.open(fname)
            img = img.convert("RGBA")
            datas = img.getdata()
    
            newData = []
            for item in datas:
                if item[0] == 255 and item[1] == 255 and item[2] == 255:
                    newData.append((255, 255, 255, 0))
                else:
                    newData.append(item)
    
            img.putdata(newData)
    
            render = ImageTk.PhotoImage(img)
            load = tk.Label(self.frame2, image=render)
            load.image = render
            load.grid(row=1,column=0,sticky='news')
            self.root.state('zoomed')
            img.save("img2.png", "PNG")
            self.open = True
    
    
        def save_file(self):
            # self.root.filename = self.value
            f = filedialog.asksaveasfile( defaultextension=".txt",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")),initialfile="Rectangular Dual Graph.txt")
            if f is None:
                return
    
    
            f.write(str(self.value))
            f.close()
        # def copy_to_file(self):
    
    
                
    if __name__ == '__main__':
        value=gui_class()
        print(value.value)
    
    ##----- End gui.py -----------------------------------------------------------##
    return locals()

@modulize('Convert_adj_equ')
def _Convert_adj_equ(__name__):
    ##----- Begin Convert_adj_equ.py ---------------------------------------------##
    import numpy as np
    def Convert_adj_equ(DGPH,symm_rooms):
    	N=len(DGPH)
    	lineq_temp=np.zeros([N,N**2])
    	# '''sys rrooms
    	# code here
    	# symm rooms'''
    
    	#starting Liner equalities as a matrix
    	for i in range(0,N):
    		for j in range(0,N):
    			if DGPH[i][j]==1:
    				lineq_temp[i][N*i+j]=1
    			if DGPH[j][i]==1:
    				lineq_temp[i][N*j+i]= -1
    	 
    	#starting removing extra variables from matrix
    	lineq_temp_np=np.array(lineq_temp)
    	lineq_temp_np=lineq_temp_np.transpose()
    	LINEQ = []
    	for i in range(0,N):
    		for j in range(0,N):
    			if DGPH[i][j] == 1:
    				LINEQ.append(lineq_temp_np[N*(i)+j])
    	LINEQ=np.array(LINEQ)		
    	#Starting Objective function
    	LINEQ = np.transpose(LINEQ)
    	
    	n = len(LINEQ[0])
    
    	f = np.zeros([1,n])
    
    	z = np.sum(DGPH[0],dtype =int)
    
    	for i in range(0,z):
    		f[0][i] = 1
    	# print(f)
    
    	#Linear inequalities (Dimensional Constraints)
    	def ismember(d, k):
    		return [1 if (i == k) else 0 for i in d]
    
    	A = []
    	for i in range(0,N):
    		A.append(ismember(LINEQ[i],-1))
    	A = np.array(A)
    	A = np.dot(A,-1)
    	A = np.delete(A,0,0)
    	Aeq = []
    	# print(LINEQ)
    	
    	def any(A):
     		for i in A:
     			if i == 1:
     				return 1
     		return 0
    
    	for i in range(0,N):
    		if any(ismember(LINEQ[i],1)) != 0 and any(ismember(LINEQ[i],-1)) != 0:
    			Aeq.append(LINEQ[i])
    	Aeq = np.array(Aeq)
    	
    	Beq = np.zeros([1,len(Aeq)])
    
    	return [f,A,Aeq,Beq]
    
    # Convert_adj_equ([[0,1,1,1,0,0,0,0,0,0,0],
    # 				 [0,0,0,0,1,0,0,0,0,0,0],
    # 				 [0,0,0,0,0,0,0,1,0,0,0],
    # 				 [0,0,0,0,0,1,1,1,0,0,0],
    # 				 [0,0,0,0,0,0,0,0,0,0,0],
    # 				 [0,0,0,0,0,0,0,0,1,0,0],
    # 				 [0,0,0,0,0,0,0,0,0,1,0],
    # 				 [0,0,0,0,0,0,0,0,0,1,1],
    # 				 [0,0,0,0,0,0,0,0,0,0,0],
    # 				 [0,0,0,0,0,0,0,0,0,0,0],
    # 				 [0,0,0,0,0,0,0,0,0,0,0]],105)
    ##----- End Convert_adj_equ.py -----------------------------------------------##
    return locals()

@modulize('solve_linear')
def _solve_linear(__name__):
    ##----- Begin solve_linear.py ------------------------------------------------##
    import numpy as np
    import scipy.optimize
    
    # global N,f_VER, A_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, ar_max, ar_min
    
    def solve_linear(N,f_VER, A_VER, b_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, ar_max, ar_min):
    
    	# print(Aeq_VER)
    	value_opti_ver = scipy.optimize.linprog(f_VER,A_ub=A_VER,b_ub=b_VER,A_eq=Aeq_VER,b_eq=Beq_VER, bounds=(1,None), method='interior-point', callback=None, options=None, x0=None)
    
    
    	b_HOR=np.zeros([N-1,1],dtype=float)
    	X1=value_opti_ver['x']
    	X1 = np.array([X1])
    	X1 = np.transpose(X1)
    	# print(X1)
    	b_HOR = np.multiply(np.transpose(ar_min),np.dot(A_VER,X1))
    	print(b_HOR[:,0])
    	b_HOR = b_HOR[:,0]
    	value_opti_hor = scipy.optimize.linprog(f_HOR,A_ub=A_HOR,b_ub=b_HOR,A_eq=Aeq_HOR,b_eq=Beq_HOR, bounds=(1,None), method='interior-point', callback=None, options=None, x0=None)
    
    	X2=value_opti_hor['x']
    	X2 = np.array([X2])
    	X2 = np.transpose(X2)
    	W=np.dot(A_VER,X1)
    	H=np.dot(A_HOR,X2)
    	AR=H/W
    	AR = AR.round(2,None)
    	ar_max= ar_max.round(2,None)
    	flag = 0
    	# print(AR)
    	for i in range(N-1):
    		if AR[i] > ar_max[i]:
    				b_VER[i] = H[i] / ar_max[i]
    				flag = 1
    
    	if flag == 1:
    		[W, H] = solve_linear(N,f_VER, A_VER, b_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, ar_max, ar_min)
    
    	return [W, H]
    ##----- End solve_linear.py --------------------------------------------------##
    return locals()

@modulize('digraph_to_eq')
def _digraph_to_eq(__name__):
    ##----- Begin digraph_to_eq.py -----------------------------------------------##
    from Convert_adj_equ import Convert_adj_equ
    import numpy as np
    import scipy.optimize
    from solve_linear import solve_linear
    
    # global N,f_VER, A_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, ar_max, ar_min
    
    
    def digraph_to_eq(VER,HOR,inp_min,ar_pmt,inp_min_ar,inp_max_ar):
    		
    	# VER=[[0,1,1,0,0,0,1,0,0,1,1,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    
    	# HOR=[[0,1,0,1,0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
    
    	# digraph_to_eq(VER,HOR)
    	N=len(VER)
    
    	[f_VER,A_VER,Aeq_VER,Beq_VER]=Convert_adj_equ(VER,5)
    	[f_HOR,A_HOR,Aeq_HOR,Beq_HOR]=Convert_adj_equ(HOR,5)
    
    	# print(len(f_VER[0]))
    	# print(len(A_VER[0]))
    
    	# inp_min=[int(x) for x in input("Enter the minimum width of room: ").strip().split()]
    
    	#inp_max=[int(x) for x in input("Enter the maximum width of room: ").strip().split()]
    
    	# ar_pmt = int(input("Enter '0' to proceed with default AR_Range (0.5 to 2) or '1' to enter custom range: "))
    
    	if ar_pmt == 0:
    		ar_min = np.dot(np.ones([N],int),0.5)
    		ar_max = np.dot(np.ones([N],int),2)
    	else:
    		# ar_min=np.array([float(x) for x in input("Enter the minimum Aspect Ratio of room: ").strip().split()])
    		ar_min = np.array(inp_min_ar)
    		# ar_max=np.array([float(x) for x in input("Enter the maximum Aspect Ratio of room: ").strip().split()])
    		ar_max = np.array(inp_max_ar)	
    
    	print(ar_max)
    
    	size_inp_min=len(inp_min)
    
    	b_VER=np.dot(np.array(inp_min),-1)
    	b_VER = np.transpose(b_VER)
    	b_VER = b_VER.astype(float)
    
    	dimensions = solve_linear(N,f_VER, A_VER, b_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, ar_max, ar_min)
    
    	# print('Height = ',dimensions[1])
    	# print('\n Width = ',dimensions[0])
    
    	return [dimensions[0],dimensions[1]]
    ##----- End digraph_to_eq.py -------------------------------------------------##
    return locals()

@modulize('floorplan_to_st')
def _floorplan_to_st(__name__):
    ##----- Begin floorplan_to_st.py ---------------------------------------------##
    import numpy as np
    from digraph_to_eq import digraph_to_eq
    def floorplan_to_st(A,inp_min,inp_pmt,inp_min_ar,inp_max_ar):
    	# A=[[1,2,6,9,10],[3,4,7,11,12],[5,5,8,13,13]]
    	m=len(A)
    	n=len(A[0])
    	
    	for i in range(0,m):
    		for j in range(0,n):
    			A[i][j] +=1
    	
    	A = np.array(A)
    
    	len_dgph=np.amax(A)
    
    
    	ver_dgph=np.zeros((len_dgph,len_dgph),int)
    	north_adj=np.zeros((1,len_dgph),int)
    
    	for i in range(0,n):
    		north_adj[0][A[0][i]-1]=1
    	for i in range(0,n):
    		for j in range(0,m):
    			if j==0:
    				temp=A[j][i]
    			if A[j][i]!=temp:
    				ver_dgph[temp-1][A[j][i]-1]=1
    				temp=A[j][i]
    	VER=[]
    	for i in north_adj:
    		VER.append(i)
    
    	for i in ver_dgph:
    		VER = np.append(VER,[i],axis=0)
    
    	VER = np.insert(VER,0,[0],axis=1)
    
    	hor_dgph=np.zeros([len_dgph,len_dgph])
    	west_adj=np.zeros([1,len_dgph])
    
    	for i in range(0,m):
    		west_adj[0][A[i][0]-1]=1
    	for i in range(0,m):
    		for j in range(0,n):
    			if j==0:
    				temp=A[i][j]
    			if A[i][j]!=temp:
    				hor_dgph[temp-1][A[i][j]-1]=1
    				temp=A[i][j]
    	HOR=[]
    
    	for i in west_adj:
    		HOR.append(i)
    
    
    	for i in hor_dgph:
    		HOR = np.append(HOR,[i],axis=0)
    
    	HOR = np.insert(HOR,0,[0],axis=1)
    	# print(HOR)
    
    	[width,height] = digraph_to_eq(VER,HOR,inp_min,inp_pmt,inp_min_ar,inp_max_ar)
    
    	return [(-1)*width,(-1)*height,hor_dgph]
    ##----- End floorplan_to_st.py -----------------------------------------------##
    return locals()

@modulize('dimension_gui')
def _dimension_gui(__name__):
    ##----- Begin dimension_gui.py -----------------------------------------------##
    # import tkinter as tk
    
    
    # root_window = tk.Tk()
    # root_window.title('Rectangular Dual')
    # root_window.geometry(str(1000) + 'x' + str(200))
    # root_window.resizable(0, 0)
    # root_window.grid_columnconfigure(0, weight=1, uniform=1)
    # root_window.grid_rowconfigure(0, weight=1)
    
    # label_main = tk.Label(root_window, padx=5, textvariable="Enter minimum width and area for each room")
    # label_main.place(relx = 0.5,  
    #                    rely = 0.5, 
    #                    anchor = 'center') 
    # root_window.mainloop()
    import tkinter as tk 
    from tkinter import font
    from tkinter import messagebox
    def gui_fnc(nodes):
    	width = []
    	ar = []
    	ar1 =[]
    	root = tk.Tk() 
    
    	root.title('Dimensional Input')
    	root.geometry(str(1000) + 'x' + str(400))
    	Upper_right = tk.Label(root, text ="Enter minimum width and aspect ratio for each room",font = ("Times New Roman",12)) 
    	  
    	Upper_right.place(relx = 0.70,  
    					  rely = 0.1, 
    					  anchor ='ne')
    
    	text_head_width = []
    	text_head_area = []
    	text_head_area1 = []
    	text_room = []
    	value_width = []
    	value_area =[]
    	value_area1 =[]
    
    	for i in range(0,nodes):
    		i_value_x = int(i/10)
    		i_value_y = i%10
    		if(i_value_y == 0):
    			text_head_width.append("text_head_width_"+str(i_value_x+1))
    			text_head_width[i_value_x] = tk.Label(root,text = "Width")
    			text_head_width[i_value_x].place(relx = 0.30 + 0.20*i_value_x,  
    					  rely = 0.2, 
    					  anchor ='ne')
    			text_head_area.append("text_head_area_"+str(i_value_x+1))
    			text_head_area[i_value_x] = tk.Label(root,text = "Min AR")
    			text_head_area[i_value_x].place(relx = 0.35 + 0.20*i_value_x,  
    					  rely = 0.2, 
    					  anchor ='ne')
    			text_head_area1.append("text_head_area1_"+str(i_value_x+1))
    			text_head_area1[i_value_x] = tk.Label(root,text = "Max AR")
    			text_head_area1[i_value_x].place(relx = 0.40 + 0.20*i_value_x,  
    					  rely = 0.2, 
    					  anchor ='ne')
    		text_room.append("text_room_"+str(i))
    		text_room[i] = tk.Label(root, text ="Room"+str(i),font = ("Times New Roman",8)) 
    
    		text_room[i].place(relx = 0.25 + 0.20*i_value_x,  
    					  rely = 0.3 + (0.05 * i_value_y), 
    					  anchor ='ne')
    		value_width.append("value_width" + str(i))
    		value_width[i] = tk.Entry(root, width = 5)
    		value_width[i].place(relx = 0.30 +0.20*i_value_x,  
    					  rely = 0.3 +(0.05)*i_value_y, 
    					  anchor ='ne')
    		value_area.append("value_area"+str(i))
    		value_area[i] = tk.Entry(root, width = 5)
    		value_area[i].place(relx = 0.35+ 0.20*i_value_x,  
    					  rely = 0.3 + (0.05)*i_value_y, 
    					   anchor ='ne')
    		value_area1.append("value_area1"+str(i))
    		value_area1[i] = tk.Entry(root, width = 5)
    		value_area1[i].place(relx = 0.40+ 0.20*i_value_x,  
    					  rely = 0.3 + (0.05)*i_value_y, 
    					   anchor ='ne')
    	def button_clicked():
    		for i in range(0,nodes):
    			width.append(int(value_width[i].get()))
    			if(checkvar1.get() == 0):
    				ar.append(float(value_area[i].get()))
    				ar1.append(float(value_area[i].get()))
    		# if(len(width) != nodes or len(ar) != nodes or len(ar1)!= nodes):
    		# 	messagebox.showerror("Invalid DImensions", "Some entry is empty")
    		# print(width)
    		# print(area)
    		# else:
    		root.destroy()
    
    	def clicked():
    		if(checkvar1.get() == 0):
    			for i in range(0,nodes):
    				value_area[i].config(state="normal")
    				value_area1[i].config(state="normal")
    				ar = []
    				ar1 = []
    		else:
    			for i in range(0,nodes):
    				value_area[i].config(state="disabled")
    				value_area1[i].config(state="disabled")
    				ar = []
    				ar1 = []
    
    	button = tk.Button(root, text='Submit', padx=5, command=button_clicked)      
    	button.place(relx = 0.5,  
    					  rely = 0.9, 
    					  anchor ='ne')
    	checkvar1 = tk.IntVar()
    	c1 = tk.Checkbutton(root, text = "Default AR Range", variable = checkvar1,onvalue = 1, offvalue = 0,command=clicked)
    	c1.place(relx = 0.85, rely = 0.9, anchor = 'ne')
    
    	root.mainloop()
    	return width,ar,ar1,checkvar1.get()
    ##----- End dimension_gui.py -------------------------------------------------##
    return locals()

@modulize('K4')
def _K4(__name__):
    ##----- Begin K4.py ----------------------------------------------------------##
    import numpy as np
    import networkx as nx
    import itertools as itr
    import operations as opr
    import contraction
    
    class K4:
    
        def __init__(self):
            self.vertices = []
            self.sep_tri = []
            self.interior_vertex = 0
            self.edge_to_be_removed = []
            self.neighbour_vertex = 0
            self.case = 0
            self.identified = 0
            self.all_edges_to_be_removed = []
    
    def find_K4(graph):
        H = graph.directed
        all_cycles = list(nx.simple_cycles(H))
        all_quads = []
        k4 =[]
        for cycle in all_cycles:
            if(len(cycle) == 4):
                if((cycle[0],cycle[2]) in H.edges and (cycle[1],cycle[3])  in H.edges ):
                    if  not opr.list_comparer(cycle,all_quads,4):
                        all_quads.append(cycle)
                        temp = K4()
                        temp.vertices = cycle
                        values = find_sep_tri(cycle,graph)
                        temp.sep_tri = values[0]
                        temp.interior_vertex = values[1]
                        value = get_edge_to_be_removed(graph,temp.sep_tri)
                        temp.case = value[0]
                        temp.edge_to_be_removed = value[1]
                        if(temp.case != 2):
                            temp.all_edges_to_be_removed.append([temp.sep_tri[0],temp.sep_tri[1]])
                            temp.all_edges_to_be_removed.append([temp.sep_tri[1],temp.sep_tri[2]])
                            temp.all_edges_to_be_removed.append([temp.sep_tri[2],temp.sep_tri[0]])
                        graph.k4.append(temp)
    
                
    
    def find_sep_tri(cycle,graph):
        sep_tri =[]
        interior_vertex = 0
        for vertex in cycle:
            contraction.initialize_degrees(graph)
            if(graph.degrees[vertex] == 3):
                interior_vertex = vertex
                break
        for vertex in cycle:
            if(vertex!=interior_vertex):
                sep_tri.append(vertex)
        return sep_tri,interior_vertex
    
    def get_edge_to_be_removed(graph,sep_tri):
        all_triangles = graph.triangles
        ab = [[sep_tri[0],sep_tri[1]],[sep_tri[1],sep_tri[2]],[sep_tri[2],sep_tri[0]]]
        case = 0
        edge_to_be_removed = []
        for subset in ab:
            count = 0
            for triangle in all_triangles:
                if(subset[0] in triangle and subset[1] in triangle):
                    count +=1
            if(count == 8 and [subset[0],subset[1]] != edge_to_be_removed):
                case = 2
                edge_to_be_removed =[subset[0],subset[1]]
                break
            if(count == 6 and [subset[0],subset[1]] != edge_to_be_removed):
                case = 1
                edge_to_be_removed = [subset[0],subset[1]]
        if(case == 0):
            edge_to_be_removed = [sep_tri[0],sep_tri[1]]
        return case,edge_to_be_removed
    
        
    def get_neigbouring_vertices(graph,k4,edge_to_be_removed):
        if(k4.case == 1):
            H = nx.from_numpy_matrix(graph.matrix,create_using=nx.DiGraph)
            all_cycles = list(nx.simple_cycles(H))
            all_triangles = []
            for cycle in all_cycles:
                if len(cycle) == 3:
                    all_triangles.append(cycle)
            for triangle in all_triangles:
                if(edge_to_be_removed[0] in triangle and edge_to_be_removed[1] in triangle):
                    if(len([x for x in triangle if x not in k4.vertices])!=0 ):
                        k4.neighbour_vertex = [x for x in triangle if x not in k4.vertices][0]
        elif(k4.case == 2):
            for temp in graph.k4:
                if(temp.case == 2 and temp != k4):
                    if(edge_to_be_removed[0] in temp.vertices and edge_to_be_removed[1] in temp.vertices):
                        k4.neighbour_vertex = temp.interior_vertex
                        temp.identified = 1
    
        # for cycle in all_quads:
        #     if((cycle[0],cycle[2]) in H.edges and (cycle[1],cycle[3])  in H.edges ):
        #         if( not list_comparer(cycle,k4,4)):
        #             temp = PTPG.K4()
        #             temp.vertices = cycle
        #             temp.sep_tri = find_sep_tri(cycle)[0]
        #             temp.interior_vertex = find_sep_tri(cycle)[1]
        #             temp.case = get_edge_to_be_removed(temp.sep_tri,temp.vertices)[0]
        #             temp.edge_to_be_removed = get_edge_to_be_removed(temp.sep_tri,temp.vertices)[1]
        #             k4.append(cycle)
        #             graph.k4.append(temp)
    
        # k4 = []
    
        # for k4_cycle in graph.k4:
        #     if(k4_cycle.case == 1):
        #         all_triangles = get_all_triangles()
        #         for triangle in all_triangles:
        #             if(k4_cycle.edge_to_be_removed[0] in triangle and k4_cycle.edge_to_be_removed[1] in triangle):
        #                 if(len([x for x in triangle if x not in k4_cycle.vertices])!=0 ):
        #                     if([x for x in triangle if x not in k4_cycle.vertices][0] not in k4_cycle.neighbour_vertex):
        #                         k4_cycle.neighbour_vertex = [x for x in triangle if x not in k4_cycle.vertices]
        #         k4_cycle.identified = 1
        #         k4.append(k4_cycle)
        #     elif(k4_cycle.case == 2 and k4_cycle.identified == 0):
        #         for temp in graph.k4:
        #             if(temp.case == 2 and temp != k4_cycle):
        #                 if(k4_cycle.edge_to_be_removed[0] in temp.vertices and k4_cycle.edge_to_be_removed[1] in temp.vertices):
        #                     k4_cycle.neighbour_vertex = temp.interior_vertex
        #                     temp.identified = 1
        #         k4_cycle.identified = 1
        #         k4.append(k4_cycle)
    
        # graph.k4 = k4
    
    def resolve_K4(graph,k4,edge_to_be_removed,rdg_vertices,rdg_vertices2,to_be_merged_vertices):
        if(k4.case!= 0 and k4.identified!=1):
            get_neigbouring_vertices(graph,k4,edge_to_be_removed)
            print(k4.neighbour_vertex)
            k4.identified = 1
            rdg_vertices.append(edge_to_be_removed[0])
            rdg_vertices2.append(edge_to_be_removed[1])
            graph.node_count +=1		#extra vertex added
            new_adjacency_matrix = np.zeros([graph.node_count, graph.node_count], int)
            for i in range(len(graph.matrix)):
                for j in range(len(graph.matrix)):
                    new_adjacency_matrix[i][j] = graph.matrix[i][j]
            to_be_merged_vertices.append(graph.node_count-1)
            # print(k4_cycle.edge_to_be_removed)
            # print(graph.node_count-1)
            # print(k4_cycle.interior_vertex[0])
            # print(k4_cycle.neighbour_vertex[0])
            #extra edges being added and shortcut being deleted
            new_adjacency_matrix[edge_to_be_removed[0]][edge_to_be_removed[1]] = 0
            new_adjacency_matrix[edge_to_be_removed[1]][edge_to_be_removed[0]] = 0
            new_adjacency_matrix[graph.node_count-1][edge_to_be_removed[0]] = 1
            new_adjacency_matrix[graph.node_count-1][edge_to_be_removed[1]] = 1
            new_adjacency_matrix[graph.node_count-1][k4.interior_vertex] = 1
            new_adjacency_matrix[graph.node_count-1][k4.neighbour_vertex] = 1
            new_adjacency_matrix[edge_to_be_removed[0]][graph.node_count-1] = 1
            new_adjacency_matrix[edge_to_be_removed[1]][graph.node_count-1] = 1
            new_adjacency_matrix[k4.interior_vertex][graph.node_count-1] = 1
            new_adjacency_matrix[k4.neighbour_vertex][graph.node_count-1] = 1
            graph.edge_count += 3
            graph.matrix = new_adjacency_matrix
            graph.north +=1
            graph.east +=1
            graph.west +=1
            graph.south +=1     
        elif(k4.case == 0):
            rdg_vertices.append(edge_to_be_removed[0])
            rdg_vertices2.append(edge_to_be_removed[1])
            graph.node_count +=1        #extra vertex added
            new_adjacency_matrix = np.zeros([graph.node_count, graph.node_count], int)
            for i in range(len(graph.matrix)):
                for j in range(len(graph.matrix)):
                    new_adjacency_matrix[i][j] = graph.matrix[i][j]
            to_be_merged_vertices.append(graph.node_count-1)
            # print(k4_cycle.edge_to_be_removed)
            # print(graph.node_count-1)
            # print(k4_cycle.interior_vertex[0])
            # print(k4_cycle.neighbour_vertex[0])
            #extra edges being added and shortcut being deleted
            new_adjacency_matrix[edge_to_be_removed[0]][edge_to_be_removed[1]] = 0
            new_adjacency_matrix[edge_to_be_removed[1]][edge_to_be_removed[0]] = 0
            new_adjacency_matrix[graph.node_count-1][edge_to_be_removed[0]] = 1
            new_adjacency_matrix[graph.node_count-1][edge_to_be_removed[1]] = 1
            new_adjacency_matrix[graph.node_count-1][k4.interior_vertex] = 1
            new_adjacency_matrix[edge_to_be_removed[0]][graph.node_count-1] = 1
            new_adjacency_matrix[edge_to_be_removed[1]][graph.node_count-1] = 1
            new_adjacency_matrix[k4.interior_vertex][graph.node_count-1] = 1
            graph.edge_count += 2
            graph.matrix = new_adjacency_matrix
            graph.north +=1
            graph.east +=1
            graph.west +=1
            graph.south +=1
    ##----- End K4.py ------------------------------------------------------------##
    return locals()

@modulize('biconnectivity')
def _biconnectivity(__name__):
    ##----- Begin biconnectivity.py ----------------------------------------------##
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
    
    ##----- End biconnectivity.py ------------------------------------------------##
    return locals()

@modulize('ptpg')
def _ptpg(__name__):
    ##----- Begin ptpg.py --------------------------------------------------------##
    import numpy as np
    import networkx as nx
    import itertools as itr
    import operations as opr
    import warnings
    import shortcutresolver as sr
    import news
    import time
    from random import randint
    import drawing as draw 
    import tkinter as tk
    import turtle
    import ptpg
    import contraction as cntr
    import expansion as exp
    import flippable as flp
    import copy
    import gui
    from floorplan_to_st import floorplan_to_st
    import dimension_gui as dimgui
    import K4
    import biconnectivity as bcn
    
    def printe(string):
        box.insert('end',string)
    
    class PTPG:
    	
    	# Attribute Initiallization
    	def __init__(self,value):
    		# self.node_count = int(input("Enter the number of nodes in the graph: "))
    		# self.edge_count = int(input("Enter the number of edges in the graph: "))
    		# print(value)
    		self.node_count=value[0]
    		self.edge_count=value[1]
    		self.command = value[3]
    		self.dimensioned = value[4]
    		self.room_names = value[5]
    		self.node_color = value[6]
    		self.matrix = np.zeros((self.node_count, self.node_count), int)
    		for i in (value[2]):
    			self.matrix[i[0]][i[1]] = 1
    			self.matrix[i[1]][i[0]] = 1
    		
    		if(self.dimensioned == 1):
    			self.value1 = dimgui.gui_fnc(self.node_count)
    			self.inp_min = self.value1[0]
    			self.inp_area = self.value1[1]
    			self.ar_pmt = 0
    			self.ar_min = []
    			self.ar_max = []
    			if self.value1[3] == 0:
    				self.ar_pmt = 1
    				self.ar_min = self.value1[1]
    				self.ar_max = self.value1[2]
    		self.graph = nx.Graph()
    		self.graph.add_edges_from(value[2])
    		self.original_edge_count = 0
    		self.original_node_count = 0
    		self.north = self.node_count
    		self.east = self.node_count + 1
    		self.south = self.node_count + 2
    		self.west = self.node_count + 3
    		self.original_north = self.north
    		self.original_east = self.east
    		self.original_south = self.south
    		self.original_west = self.west
    		# self.matrix = np.zeros((self.node_count, self.node_count), int)
    		self.user_matrix = None
    		self.edge_matrix = None
    		
    		self.cip_list = []
    		self.civ = []
    		self.original_cip =[]
    		self.node_color_list =[]
    
    		self.node_position = None
    		self.degrees = None
    		self.good_vertices = None
    		self.contractions = []
    		self.rdg_vertices = []
    		self.to_be_merged_vertices = []
    		self.k4 = []
    
    		self.t1_matrix = None
    		self.t2_matrix = None
    		self.t1_longest_distance = [-1] * (self.node_count + 4)
    		self.t2_longest_distance = [-1] * (self.node_count + 4)
    		self.t1_longest_distance_value = -1
    		self.t2_longest_distance_value = -1
    		self.n_s_paths = []
    		self.w_e_paths = []
    
    		self.rel_matrix =[]
    		self.room_x = np.zeros(self.node_count)
    		self.room_x_list = []
    		self.room_y = np.zeros(self.node_count)
    		self.room_y_list = []
    		self.room_x_bottom_right = np.zeros(self.node_count)
    		self.room_x_bottom_right_list = []
    		self.room_x_bottom_left = np.zeros(self.node_count)
    		self.room_x_bottom_left_list =[]
    		self.room_x_top_right = np.zeros(self.node_count)
    		self.room_x_top_right_list =[]
    		self.room_x_top_left = np.zeros(self.node_count)
    		self.room_x_top_left_list = []
    		self.room_y_right_top = np.zeros(self.node_count)
    		self.room_y_right_top_list =[]
    		self.room_y_left_top = np.zeros(self.node_count)
    		self.room_y_left_top_list =[]
    		self.room_y_right_bottom = np.zeros(self.node_count)
    		self.room_y_right_bottom_list = []
    		self.room_y_left_bottom = np.zeros(self.node_count)
    		self.room_y_left_bottom_list = []
    		self.room_height = np.zeros(self.node_count)
    		self.room_height_list = []
    		self.room_width_list = []
    		self.room_width = np.zeros(self.node_count)
    		self.encoded_matrix = None
    		# print("Enter each edge in new line")
    		# for i in range(self.edge_count):
    		#     line = input()
    		#     node1 = int(line.split()[0])
    		#     node2 = int(line.split()[1])
    		#     self.matrix[node1][node2] = 1
    		#     self.matrix[node2][node1] = 1
    
    		self.directed = opr.get_directed(self)
    		self.triangles = opr.get_all_triangles(self)
    		self.outer_vertices = opr.get_outer_boundary_vertices(self)[0]
    		self.outer_boundary = opr.get_outer_boundary_vertices(self)[1]
    		self.shortcuts = sr.get_shortcut(self)
    		self.shortcut_list = []
    		self.origin = 50
    		self.boundaries = []
    		
    		self.area = []
    		self.Time = 0
    		self.articulation_points = [False] * (self.node_count)
    		self.no_of_articulation_points = 0
    		self.articulation_points_value = []
    		self.no_of_bcc = 0
    		self.bcc_sets = [set() for i in range(self.node_count)]
    		self.articulation_point_sets = [set() for i in range(self.node_count)]
    		self.added_edges = set()
    		self.removed_edges = set()
    		self.final_added_edges = set()
    		self.biconnected_vertices = []
    
    		self.ar_min = []
    		self.ar_max = []
    	"""
    	Adding the NESW vertices to the original graph 
    	"""
    
    	def isBiconnected(self):
    		h = nx.from_numpy_matrix(self.matrix)
    		return nx.is_biconnected(h)
    	def isBCUtil(self, u, visited, parent, low, disc):
    
    		children = 0
    
    		visited[u] = True
    
    		disc[u] = self.Time
    		low[u] = self.Time
    		self.Time += 1
    		for v in self.find_neighbors(u):
    			if self.matrix[u][v] == 1:
    				# If v is not visited yet, then make it a child of u
    				# in DFS tree and recur for it
    				if visited[v] == False:
    					parent[v] = u
    					children += 1
    					if self.isBCUtil(v, visited, parent, low, disc):
    						return True
    
    					# Check if the subtree rooted with v has a connection to
    					# one of the ancestors of u
    					low[u] = min(low[u], low[v])
    
    					# u is an articulation point in following cases
    					# (1) u is root of DFS tree and has two or more children.
    					if parent[u] == -1 and children > 1:
    						# self.articulation_points[u] = True
    						return True
    
    					# (2) If u is not root and low value of one of its child is more
    					# than discovery value of u.
    					if parent[u] != -1 and low[v] >= disc[u]:
    						# self.articulation_points[u] = True
    						return True
    
    				elif v != parent[u]:  # Update low value of u for parent function calls.
    					low[u] = min(low[u], disc[v])
    			else:
    				continue
    
    		return False
    
    	def isBC(self):
    
    		visited = [False] * (self.node_count)
    		disc = [float("Inf")] * (self.node_count)
    		low = [float("Inf")] * (self.node_count)
    		parent = [-1] * (self.node_count)
    		if self.isBCUtil(0, visited, parent, low, disc):
    			return False
    
    		if any(i == False for i in visited):
    			return False
    		"""
    		for i in visited:
    			if visited[i] is False:
    				return False
    			else:
    				continue
    		"""
    		return True
    
    	def BCCUtil(self, u, parent, low, disc, st):
    
    		# Count of children in current node
    		children = 0
    		#visited[u] = True
    		# Initialize discovery time and low value
    		disc[u] = self.Time
    		low[u] = self.Time
    		self.Time += 1
    
    		# Recur for all the vertices adjacent to this vertex
    		for v in range(self.node_count):
    			if self.matrix[u][v] == 1:
    				# If v is not visited yet, then make it a child of u
    				# in DFS tree and recur for it
    				if disc[v] == -1:
    					parent[v] = u
    					children += 1
    					st.append((u, v))  # store the edge in stack
    					self.BCCUtil(v, parent, low, disc, st)
    
    					# Check if the subtree rooted with v has a connection to
    					# one of the ancestors of u
    					# Case 1 -- per Strongly Connected Components Article
    					low[u] = min(low[u], low[v])
    
    					# If u is an articulation point, pop
    					# all edges from stack till (u, v)
    					if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
    						self.no_of_bcc += 1  # increment count
    						self.articulation_points[u] = True
    						w = -1
    						while w != (u, v):
    							w = st.pop()
    							# print("In bccutil no of bcc = " , self.no_of_bcc)
    							self.bcc_sets[0].add(w[0])
    							# self.bcc_sets[(self.no_of_bcc) - 1].add(w[1])
    							# print("Printing from bccutil")
    							# print(w[0])
    							print(w, "    ")
    						#print("")
    
    				elif v != parent[u] and low[u] > disc[v]:
    					'''Update low value of 'u' only of 'v' is still in stack 
    					(i.e. it's a back edge, not cross edge). 
    					Case 2 
    					-- per Strongly Connected Components Article'''
    
    					low[u] = min(low[u], disc[v])
    
    					st.append((u, v))
    
    	def print_biconnected_components(self):
    		visited = [False] * (self.node_count)
    		disc = [-1] * (self.node_count)
    		low = [-1] * (self.node_count)
    		parent = [-1] * (self.node_count)
    		st = []
    		# print("no of bcc = ", self.no_of_bcc)
    		# print(self.articulation_points)
    		for i in range(self.node_count):
    			if disc[i] == -1:
    				self.BCCUtil(i, parent, low, disc, st)
    
    			if st:
    				self.no_of_bcc = self.no_of_bcc + 1
    
    				while st:
    					w = st.pop()
    					# print("printing from print_biconnected_components")
    					# print(w[0])
    					# print("printing from print_biconnected_components, no of bcc = ", self.no_of_bcc)
    					# self.bcc_sets[(self.no_of_bcc) - 1].add(w[0])
    					# self.bcc_sets[(self.no_of_bcc) - 1].add(w[1])
    					print(w, "    ")
    				#print("")
    
    		# print(self.bcc_sets)
    
    	def utility_function_for_initialize_bcc_sets(self, u, bcc_sets, parent, low, disc, st):
    		children = 0
    		# visited[u] = True
    		# Initialize discovery time and low value
    		disc[u] = self.Time
    		low[u] = self.Time
    		self.Time += 1
    
    		# Recur for all the vertices adjacent to this vertex
    		for v in range(self.node_count):
    			if self.matrix[u][v] == 1:
    				# If v is not visited yet, then make it a child of u
    				# in DFS tree and recur for it
    				if disc[v] == -1:
    					parent[v] = u
    					children += 1
    					st.append((u, v))  # store the edge in stack
    					self.utility_function_for_initialize_bcc_sets(v, bcc_sets, parent, low, disc, st)
    
    					# Check if the subtree rooted with v has a connection to
    					# one of the ancestors of u
    					# Case 1 -- per Strongly Connected Components Article
    					low[u] = min(low[u], low[v])
    
    					# If u is an articulation point, pop
    					# all edges from stack till (u, v)
    					if parent[u] == -1 and children > 1 or parent[u] != -1 and low[v] >= disc[u]:
    						self.no_of_bcc += 1  # increment count
    						self.articulation_points[u] = True
    						w = -1
    						while w != (u, v):
    							w = st.pop()
    							# print("In utility_function_for_initialize_bcc_sets no of bcc = ", self.no_of_bcc)
    							bcc_sets[(self.no_of_bcc) - 1].add(w[0])
    							bcc_sets[(self.no_of_bcc) - 1].add(w[1])
    							# print("Printing from bccutil")
    							# print(w[0])
    							# print(w)
    						# print("")
    
    				elif v != parent[u] and low[u] > disc[v]:
    					'''Update low value of 'u' only of 'v' is still in stack 
    					(i.e. it's a back edge, not cross edge). 
    					Case 2 
    					-- per Strongly Connected Components Article'''
    
    					low[u] = min(low[u], disc[v])
    
    					st.append((u, v))
    
    	def initialize_bcc_sets(self):
    		disc = [-1] * (self.node_count)
    		low = [-1] * (self.node_count)
    		parent = [-1] * (self.node_count)
    		st = []
    		# self.bcc_sets = [set() for i in range(self.no_of_bcc)]
    		self.no_of_bcc = 0
    		# print("no of bcc = ", self.no_of_bcc)
    		# print(self.articulation_points)
    		for i in range(self.node_count):
    			if disc[i] == -1:
    				self.utility_function_for_initialize_bcc_sets(i, self.bcc_sets, parent, low, disc, st)
    
    			if st:
    				self.no_of_bcc = self.no_of_bcc + 1
    
    				while st:
    					w = st.pop()
    					# print("printing from print_biconnected_components")
    					# print(w[0])
    					# print("printing from initialize_bcc_sets, no of bcc = ", self.no_of_bcc)
    					self.bcc_sets[(self.no_of_bcc) - 1].add(w[0])
    					self.bcc_sets[(self.no_of_bcc) - 1].add(w[1])
    					# print(w)
    				# print("")
    		self.bcc_sets = [x for x in self.bcc_sets if x]
    		# print(len(self.bcc_sets))
    		# print(self.bcc_sets)
    		# self.find_articulation_points()
    		# self.remove_articulation_points_from_bcc_sets()
    		# print(self.bcc_sets)
    
    	def find_articulation_points(self):
    		# self.no_of_articulation_points = 0
    		for i in range(self.node_count):
    			if self.articulation_points[i]:
    				self.no_of_articulation_points += 1
    				self.articulation_points_value.append(i)
    				self.articulation_point_sets[i].add(i)
    		self.articulation_point_sets = [x for x in self.articulation_point_sets if x]
    
    	def find_neighbors(self, v):
    		h = nx.from_numpy_matrix(self.matrix)
    		nl = []
    		for n in h.neighbors(v):
    			nl.append(n)
    		return nl
    
    	def make_biconnected(self):
    		for i in range(len(self.articulation_points_value)):
    			nl = self.find_neighbors(self.articulation_points_value[i])
    			for j in range(0, (len(nl) - 1)):
    				if not self.belong_in_same_block(nl[j], nl[j+1]):
    
    					self.matrix[nl[j]][nl[j+1]] = 1
    					self.matrix[nl[j+1]][nl[j]] = 1
    					self.added_edges.add((nl[j], nl[j+1]))
    					if (self.articulation_points_value[i], nl[j]) in self.added_edges or\
    							(nl[j], self.articulation_points_value[i]) in self.added_edges:
    						self.matrix[self.articulation_points_value[i]][nl[j]] = 0
    						self.matrix[nl[j]][self.articulation_points_value[i]] = 0
    						self.removed_edges.add((self.articulation_points_value[i], nl[j]))
    						self.removed_edges.add((nl[j], self.articulation_points_value[i]))
    					if (self.articulation_points_value[i], nl[j+1]) in self.added_edges or\
    							(nl[j+1], self.articulation_points_value[i]) in self.added_edges:
    						self.matrix[self.articulation_points_value[i]][nl[j+1]] = 0
    						self.matrix[nl[j+1]][self.articulation_points_value[i]] = 0
    						self.removed_edges.add((self.articulation_points_value[i], nl[j+1]))
    						self.removed_edges.add((nl[j+1], self.articulation_points_value[i]))
    		self.final_added_edges = self.added_edges - self.removed_edges
    
    	def remove_articulation_points_from_bcc_sets(self):
    		for i in self.articulation_points_value:
    			for j in range(self.no_of_articulation_points + 1):
    				if i in self.bcc_sets[j]:
    					self.bcc_sets[j].remove(i)
    
    	def belong_in_same_block(self, a, b):
    		for i in range(len(self.bcc_sets)):
    			if (a in self.bcc_sets[i]) and (b in self.bcc_sets[i]):
    				return True
    		return False
    
    	def create_single_dual(self,mode,pen,textbox):
    		global box
    		box = textbox
    		# if (self.isBiconnected()):
    			# 	print("The given graph is biconnected")
    			# 	#print("Below are the biconnected components")
    			# 	#G.print_biconnected_components()
    			# else:
    			# 	print("The given graph is not biconnected")
    			# 	print("Making the graph biconnected")
    			# 	#G.print_biconnected_components()
    			# 	#print(G.no_of_articulation_points)
    			# 	#print(G.articulation_points_value)
    			# 	#print(G.bcc_sets)
    			# 	#print(G.articulation_point_sets)
    			# 	#print(G.articulation_points_value)
    			# 	#print(G.belong_in_same_block(3, 1))
    			# 	#print("Below are the biconnected components")
    			# 	#G.print_biconnected_components()
    			# 	self.initialize_bcc_sets()
    			# 	self.find_articulation_points()
    			# 	self.make_biconnected()
    			# 	print("Is the graph now Biconnected : ", self.isBiconnected(), "; As shown in figure 2")
    			# 	print("The added edges are : ", self.final_added_edges)
    			# 	self.triangles = opr.get_all_triangles(self)
    			# 	for i in self.final_added_edges:
    			# 		print(i[0],i[1])
    			# 		bcn.biconnectivity_transformation(self,i,self.biconnected_vertices)
    		self.triangles = opr.get_all_triangles(self)
    		K4.find_K4(self)
    		for i in self.k4:
    			# print(i.vertices)
    			# print(i.sep_tri)
    			# print(i.interior_vertex)
    			# print(i.edge_to_be_removed)
    			# print(i.case)
    			K4.resolve_K4(self,i,i.edge_to_be_removed,self.rdg_vertices,self.rdg_vertices2,self.to_be_merged_vertices)
    		# print(self.matrix)
    		self.directed = opr.get_directed(self)
    		self.triangles = opr.get_all_triangles(self)
    		self.outer_vertices = opr.get_outer_boundary_vertices(self)[0]
    		self.outer_boundary = opr.get_outer_boundary_vertices(self)[1]
    		self.shortcuts = sr.get_shortcut(self)
    		self.civ = news.find_cip_single(self)
    		news.add_news_vertices(self)
    		print("North Boundary: ", self.civ[0])
    		print("East Boundary: ", self.civ[1])
    		print("South Boundary: ", self.civ[2])
    		print("West Boundary: ",self.civ[3])
    		for i in range(0,len(self.to_be_merged_vertices)):
    			self.node_color.append(self.node_color[self.rdg_vertices[i]])
    		self.node_position = nx.planar_layout(nx.from_numpy_matrix(self.matrix))
    		cntr.initialize_degrees(self)
    		cntr.initialize_good_vertices(self)
    		v, u = cntr.contract(self)
    		while v != -1:
    			v, u = cntr.contract(self)
    			# draw.draw_undirected_graph(self,pen)
    			# input()
    		# print(self.contractions)
    		exp.get_trivial_rel(self)
    		while len(self.contractions) != 0:
    			exp.expand(self)
    		draw.construct_rdg(self,self.to_be_merged_vertices,self.rdg_vertices)
    		# for i  in range(0,len(self.to_be_merged_vertices)):
    		#   print(self.room_x[self.to_be_merged_vertices[i]],self.room_y[self.to_be_merged_vertices[i]],self.room_width[self.to_be_merged_vertices[i]],self.room_height[self.to_be_merged_vertices[i]],self.room_x_top_left[self.to_be_merged_vertices[i]],self.room_x_top_right[self.to_be_merged_vertices[i]],self.room_y_left_top[self.to_be_merged_vertices[i]],self.room_y_left_bottom[self.to_be_merged_vertices[i]],self.room_x_bottom_left[self.to_be_merged_vertices[i]],self.room_x_bottom_right[self.to_be_merged_vertices[i]],self.room_y_right_top[self.to_be_merged_vertices[i]],self.room_y_right_bottom[self.to_be_merged_vertices[i]])
    		#   print(self.room_x[self.rdg_vertices[i]],self.room_y[self.rdg_vertices[i]],self.room_width[self.rdg_vertices[i]],self.room_height[self.rdg_vertices[i]],self.room_x_top_left[self.rdg_vertices[i]],self.room_x_top_right[self.rdg_vertices[i]],self.room_y_left_top[self.rdg_vertices[i]],self.room_y_left_bottom[self.rdg_vertices[i]],self.room_x_bottom_left[self.rdg_vertices[i]],self.room_x_bottom_right[self.rdg_vertices[i]],self.room_y_right_top[self.rdg_vertices[i]],self.room_y_right_bottom[self.rdg_vertices[i]]) 
    		# print(self.room_x,self.room_y,self.room_width,self.room_height,self.room_x_top_left,self.room_x_top_right,self.room_y_left_top,self.room_y_left_bottom,self.room_x_bottom_left,self.room_x_bottom_right,self.room_y_right_top,self.room_y_right_bottom)
    		if(mode == 1):
    			draw.draw_rdg(self,1,pen,self.to_be_merged_vertices)
    
    	def create_single_floorplan(self,pen,textbox):
    		global box
    		box = textbox
    		self.create_single_dual(0,pen,textbox)
    		self.encoded_matrix = opr.get_encoded_matrix(self)
    		B = copy.deepcopy(self.encoded_matrix)
    		A = copy.deepcopy(self.encoded_matrix)
    		minimum_width = min(self.inp_min)
    		for i in range(0,len(self.biconnected_vertices)):
    			self.inp_min.append(0)
    			self.ar_min.append(0.5)
    			self.ar_max.append(2)
    		for i in range(0,len(self.to_be_merged_vertices)):
    			self.inp_min.append(minimum_width)
    			self.ar_min.append(0.5)
    			self.ar_max.append(2)
    
    		[width,height,hor_dgph] = floorplan_to_st(A,self.inp_min,self.ar_pmt,self.ar_min,self.ar_max)
    		A=B
    		# print(A)
    		width = np.transpose(width)
    		height = np.transpose(height)
    		self.room_width = width.flatten()
    		self.room_height = height.flatten()
    		draw.construct_rfp(self,hor_dgph,self.to_be_merged_vertices,self.rdg_vertices)
    		for i in range(0,len(self.room_x)):
    			self.room_x[i]=round(self.room_x[i],3)
    			# print(self.room_x[i])
    		for i in range(0,len(self.room_y)):
    			self.room_y[i]=round(self.room_y[i],3)
    			# print(self.room_x[i],self.room_y[i],self.room_width[i],self.room_height[i],self.room_x_top_left[i],self.room_x_top_right[i],self.room_y_left_top[i],self.room_y_left_bottom[i],self.room_x_bottom_left[i],self.room_x_bottom_right[i],self.room_y_right_top[i],self.room_y_right_bottom[i])
    		# print(self.room_x,self.room_y,self.room_width,self.room_height,self.room_x_top_left,self.room_x_top_right,self.room_y_left_top,self.room_y_left_bottom,self.room_x_bottom_left,self.room_x_bottom_right,self.room_y_right_top,self.room_y_right_bottom)
    			# print(self.room_y[i])
    		opr.calculate_area(self,self.to_be_merged_vertices,self.rdg_vertices)
    		draw.draw_rdg(self,1,pen,self.to_be_merged_vertices)
    
    
    	def create_multiple_dual(self,mode,pen,textbox):
    		global box
    		box = textbox
    		# if (self.isBiconnected()):
    		# 	print("The given graph is biconnected")
    		# 	#print("Below are the biconnected components")
    		# 	#G.print_biconnected_components()
    		# else:
    		# 	print("The given graph is not biconnected")
    		# 	print("Making the graph biconnected")
    		# 	#G.print_biconnected_components()
    		# 	#print(G.no_of_articulation_points)
    		# 	#print(G.articulation_points_value)
    		# 	#print(G.bcc_sets)
    		# 	#print(G.articulation_point_sets)
    		# 	#print(G.articulation_points_value)
    		# 	#print(G.belong_in_same_block(3, 1))
    		# 	#print("Below are the biconnected components")
    		# 	#G.print_biconnected_components()
    		# 	self.initialize_bcc_sets()
    		# 	self.find_articulation_points()
    		# 	self.make_biconnected()
    		# 	print("Is the graph now Biconnected : ", self.isBiconnected(), "; As shown in figure 2")
    		# 	print("The added edges are : ", self.final_added_edges)
    		# 	self.triangles = opr.get_all_triangles(self)
    		# 	for i in self.final_added_edges:
    		# 		print(i[0],i[1])
    		# 		bcn.biconnectivity_transformation(self,i,self.biconnected_vertices)
    		self.triangles = opr.get_all_triangles(self)
    		K4.find_K4(self)
    		if(len(self.k4) == 0):
    			
    			self.directed = opr.get_directed(self)
    			self.triangles = opr.get_all_triangles(self)
    			self.outer_vertices = opr.get_outer_boundary_vertices(self)[0]
    			self.outer_boundary = opr.get_outer_boundary_vertices(self)[1]
    			self.shortcuts = sr.get_shortcut(self)
    		# self.civ = news.find_cip_single(self)
    			start = time.time()
    			# print(self.node_count)
    			# print(self.edge_count)
    			# print(self.matrix)
    			self.boundaries = opr.find_possible_boundary(opr.ordered_outer_boundary(self))
    			self.cip_list = news.populate_cip_list(self)
    			# self.civ = self.cip_list[0]
    			# self.original_cip = self.civ.copy()
    			self.edge_matrix = self.matrix.copy()
    			self.original_edge_count = self.edge_count
    			self.original_node_count = self.node_count
    			if(len(self.cip_list) == 0):
    				size = len(self.shortcuts)-4
    				self.shortcut_list = list(itr.combinations(self.shortcuts,len(self.shortcuts)-4))
    		#print(self.cip_list)
    		#input()
    
    			no_of_boundaries = 0
    		# self.cip_list = [[[0, 6], [6, 1, 7], [7, 2, 4, 8], [8, 9, 0]], [[6], [6, 1, 7], [7, 2, 4, 8], [8, 9, 0, 6]], [[6, 1], [1, 7], [7, 2, 4, 8], [8, 9, 0, 6]], [[6, 1, 7], [7], [7, 2, 4, 8], [8, 9, 0, 6]], [[6, 1, 7], [7, 2], [2, 4, 8], [8, 9, 0, 6]], [[6, 1, 7], [7, 2, 4], [4, 8], [8, 9, 0, 6]], [[6, 1, 7], [7, 2, 4, 8], [8], [8, 9, 0, 6]], [[6, 1, 7], [7, 2, 4, 8], [8, 9], [9, 0, 6]], [[6, 1, 7], [7, 2, 4, 8], [8, 9, 0], [0, 6]], [[1, 7], [7, 2, 4, 8], [8, 9, 0, 6], [6, 1]], [[7], [7, 2, 4, 8], [8, 9, 0, 6], [6, 1, 7]], [[7, 2], [2, 4, 8], [8, 9, 0, 6], [6, 1, 7]], [[7, 2, 4], [4, 8], [8, 9, 0, 6], [6, 1, 7]], [[7, 2, 4, 8], [8], [8, 9, 0, 6], [6, 1, 7]], [[7, 2, 4, 8], [8, 9], [9, 0, 6], [6, 1, 7]], [[7, 2, 4, 8], [8, 9, 0], [0, 6], [6, 1, 7]], [[7, 2, 4, 8], [8, 9, 0, 6], [6], [6, 1, 7]], [[7, 2, 4, 8], [8, 9, 0, 6], [6, 1], [1, 7]], [[2, 4, 8], [8, 9, 0, 6], [6, 1, 7], [7, 2]], [[4, 8], [8, 9, 0, 6], [6, 1, 7], [7, 2, 4]], [[8], [8, 9, 0, 6], [6, 1, 7], [7, 2, 4, 8]], [[8, 9], [9, 0, 6], [6, 1, 7], [7, 2, 4, 8]], [[8, 9, 0], [0, 6], [6, 1, 7], [7, 2, 4, 8]], [[8, 9, 0, 6], [6], [6, 1, 7], [7, 2, 4, 8]], [[8, 9, 0, 6], [6, 1], [1, 7], [7, 2, 4, 8]], [[8, 9, 0, 6], [6, 1, 7], [7], [7, 2, 4, 8]], [[8, 9, 0, 6], [6, 1, 7], [7, 2], [2, 4, 8]], [[8, 9, 0, 6], [6, 1, 7], [7, 2, 4], [4, 8]], [[9, 0, 6], [6, 1, 7], [7, 2, 4, 8], [8, 9]]]
    		# # print(self.shortcut_list) 
    		# print(self.edge_matrix)
    		# print(self.civ)
    		# input()
    		# news.add_news_vertices(self)
    		# # print(self.matrix)
    		# self.node_position = nx.planar_layout(nx.from_numpy_matrix(self.matrix))
    		# # draw.draw_undirected_graph(self,pen)
    		# # input()
    		# cntr.initialize_degrees(self)
    		# cntr.initialize_good_vertices(self)
    		# # comb = itr.permutations(self.good_vertices)
    		# # for x in comb:
    		# # self.good_vertices = list(x)
    		# v, u = cntr.contract(self)
    		# while v != -1:
    		#     v, u = cntr.contract(self)
    		#     # draw.draw_undirected_graph(self,pen)
    		#     # input()
    		# # print(self.contractions)
    		# exp.get_trivial_rel(self)
    		# while len(self.contractions) != 0:
    		#     exp.expand(self)
    		# print(self.matrix)
    		# draw.draw_directed_graph(self,pen)
    		# input()
    		# print(self.degrees)
    			count = 0
    			if(len(self.cip_list)== 0):
    				for resolver in self.shortcut_list:
    					node_color1 = self.node_color
    					node_color2 =self.node_color
    					rdg_vertices = []
    					rdg_vertices2 = []
    					to_be_merged_vertices = []
    					for i in range(0,size):
    						sr.remove_shortcut(resolver[i],self,rdg_vertices,rdg_vertices2,to_be_merged_vertices)
    
    					self.civ = news.find_cip_single(self)
    					for i in range(0,len(to_be_merged_vertices)):
    						node_color1.append(self.node_color[rdg_vertices[i]])
    					for i in range(0,len(to_be_merged_vertices)):
    						node_color2.append(self.node_color[rdg_vertices2[i]])						
    					print("North Boundary: ", self.civ[0])
    					print("East Boundary: ", self.civ[1])
    					print("South Boundary: ", self.civ[2])
    					print("West Boundary: ",self.civ[3])
    					news.add_news_vertices(self)
    					# print(self.matrix)
    					
    					self.node_position = nx.planar_layout(nx.from_numpy_matrix(self.matrix))
    					# draw.draw_undirected_graph(self,pen)
    					# input()
    					cntr.initialize_degrees(self)
    					cntr.initialize_good_vertices(self)
    					# comb = itr.permutations(self.good_vertices)
    					# for x in comb:
    					# self.good_vertices = list(x)
    					v, u = cntr.contract(self)
    					while v != -1:
    						v, u = cntr.contract(self)
    						# draw.draw_undirected_graph(self,pen)
    						# input()
    					# print(self.contractions)
    					exp.get_trivial_rel(self)
    					while len(self.contractions) != 0:
    						exp.expand(self)
    					rel_matrix =[]
    					rel_matrix.append(self.matrix)
    					self.rdg_vertices.append(rdg_vertices)
    					self.rdg_vertices2.append(rdg_vertices2)
    					self.to_be_merged_vertices.append(to_be_merged_vertices)
    					self.node_color_list.append(node_color1)
    					self.node_color_list.append(node_color2)
    					# print(self.rel_matrix)
    					for i in rel_matrix:
    						self.matrix = i
    						# print(self.user_matrix)
    						flippable_edges = flp.get_flippable_edges(self,i)
    						flippable_vertices = flp.get_flippable_vertices(self,i)[0]
    						flippable_vertices_neighbours = flp.get_flippable_vertices(self,i)[1]
    						# print(flippable_edges)
    						# print(flippable_vertices)
    						for j in range(0,len(flippable_edges)):
    							new_rel = flp.resolve_flippable_edge(flippable_edges[j],self,i)
    							if(not any(np.array_equal(new_rel, i) for i in rel_matrix)):
    								# print("Entered")
    								rel_matrix.append(new_rel)
    								self.rdg_vertices.append(rdg_vertices)
    								self.rdg_vertices2.append(rdg_vertices2)
    								self.to_be_merged_vertices.append(to_be_merged_vertices)
    								self.node_color_list.append(node_color1)
    								self.node_color_list.append(node_color2)
    						for j in range(0,len(flippable_vertices)):
    							# print("Entered")
    							new_rel = flp.resolve_flippable_vertex(flippable_vertices[j],flippable_vertices_neighbours[j],self,i)
    							if(not any(np.array_equal(new_rel, i) for i in rel_matrix)):
    								rel_matrix.append(new_rel)
    								self.rdg_vertices.append(rdg_vertices)
    								self.rdg_vertices2.append(rdg_vertices2)
    								self.to_be_merged_vertices.append(to_be_merged_vertices)
    								self.node_color_list.append(node_color1)
    								self.node_color_list.append(node_color2)
    					count +=1
    					if(count != len(self.shortcut_list)):
    						self.node_count = self.original_node_count
    						self.edge_count = self.original_edge_count
    						self.matrix = self.edge_matrix.copy()
    						self.north = self.original_north
    						self.west = self.original_west
    						self.east = self.original_east
    						self.south = self.original_south
    					for i in rel_matrix:
    						self.rel_matrix.append(i)
    					print("Number of different floor plans: ",len(rel_matrix))
    					print("\n")
    					# for i in self.rel_matrix:
    					#     input()
    					#     self.matrix = i
    					#     draw.construct_rdg(self)
    					# # encoded_matrix = opr.get_encoded_matrix(self)
    					# # if(not any(np.array_equal(encoded_matrix, i) for i in self.encoded_matrix)):
    					#     # self.encoded_matrix.append(encoded_matrix)
    					#     draw.draw_rdg(self,pen)
    					self.civ = self.original_cip.copy()
    					
    			else:
    				self.civ = self.cip_list[0]
    				self.original_cip = self.civ.copy()
    				for k in self.cip_list:
    					node_color = self.node_color
    					self.civ = k
    					news.add_news_vertices(self)
    					# print(self.civ)
    					#input()
    					# print("Checking...")
    					# print(self.matrix)
    					if(opr.is_complex_triangle(self) == True):
    						count +=1
    						if(count != len(self.cip_list)):
    							self.node_count = self.original_node_count
    							self.edge_count = self.original_edge_count
    							self.matrix = self.edge_matrix.copy()
    						continue
    					# for i in range(0,len(to_be_merged_vertices)):
    					# 	node_color.append(self.node_color[self.rdg_vertices[i]])
    					print("North Boundary: ", self.civ[0])
    					print("East Boundary: ", self.civ[1])
    					print("South Boundary: ", self.civ[2])
    					print("West Boundary: ",self.civ[3])
    					no_of_boundaries += 1
    					# print("Boundary count: ",no_of_boundaries)
    					self.node_position = nx.planar_layout(nx.from_numpy_matrix(self.matrix))
    					# draw.draw_undirected_graph(self,pen)
    					# input()
    					cntr.initialize_degrees(self)
    					cntr.initialize_good_vertices(self)
    					# print(self.good_vertices)
    					# comb = itr.permutations(self.good_vertices)
    					# for x in comb:
    					# self.good_vertices = list(x)
    					v, u = cntr.contract(self)
    					while v != -1:
    						v, u = cntr.contract(self)
    						# draw.draw_undirected_graph(self,pen)
    						# input()
    					# print(self.contractions)
    					exp.get_trivial_rel(self)
    					while len(self.contractions) != 0:
    						exp.expand(self)
    					rel_matrix =[]
    					rel_matrix.append(self.matrix)
    					self.node_color_list.append(node_color)
    					# draw.draw_directed_graph(self,pen)
    					# print(self.rel_matrix)
    					for i in rel_matrix:
    						self.matrix = i
    						# print(self.user_matrix)
    						flippable_edges = flp.get_flippable_edges(self,i)
    						flippable_vertices = flp.get_flippable_vertices(self,i)[0]
    						flippable_vertices_neighbours = flp.get_flippable_vertices(self,i)[1]
    						# print(flippable_edges)
    						# print(flippable_vertices)
    						for j in range(0,len(flippable_edges)):
    							new_rel = flp.resolve_flippable_edge(flippable_edges[j],self,i)
    							if(not any(np.array_equal(new_rel, i) for i in rel_matrix)):
    								# print("Entered")
    								# self.rdg_vertices.append(rdg_vertices)
    								# self.rdg_vertices2.append(rdg_vertices2)
    								rel_matrix.append(new_rel)
    								self.node_color_list.append(node_color)
    						for j in range(0,len(flippable_vertices)):
    							# print("Entered")
    							new_rel = flp.resolve_flippable_vertex(flippable_vertices[j],flippable_vertices_neighbours[j],self,i)
    							if(not any(np.array_equal(new_rel, i) for i in rel_matrix)):
    								rel_matrix.append(new_rel)
    								self.node_color_list.append(node_color)
    								# self.rdg_vertices.append(rdg_vertices)
    								# self.rdg_vertices2.append(rdg_vertices2)
    
    					count +=1
    					if(count != len(self.cip_list)):
    						self.node_count = self.original_node_count
    						self.edge_count = self.original_edge_count
    						self.matrix = self.edge_matrix.copy()
    					for i in rel_matrix:
    						self.rel_matrix.append(i)
    					print("Number of different floor plans: ",len(rel_matrix))
    					print("\n")
    		
    
    			
    
    		# print(len(self.to_be_merged_vertices))
    		# print(len(self.rdg_vertices))
    			print("Total number of different floor plans: ",len(self.rel_matrix))
    			print("Total boundaries used:", no_of_boundaries)
    			# for i in self.rel_matrix:
    			#     self.matrix = i
    			#     draw.draw_directed_graph(self,pen)
    			#     input()
    			end = time.time()
    			print(f"Runtime of the program is {end - start}")
    
    		else:
    			self.directed = opr.get_directed(self)
    			self.triangles = opr.get_all_triangles(self)
    			self.outer_vertices = opr.get_outer_boundary_vertices(self)[0]
    			self.outer_boundary = opr.get_outer_boundary_vertices(self)[1]
    			self.shortcuts = sr.get_shortcut(self)
    		# self.civ = news.find_cip_single(self)
    			start = time.time()
    			# print(self.node_count)
    			# print(self.edge_count)
    			# print(self.matrix)
    			# self.boundaries = opr.find_possible_boundary(opr.ordered_outer_boundary(self))
    			# self.cip_list = news.populate_cip_list(self)
    			# self.civ = self.cip_list[0]
    			# self.original_cip = self.civ.copy()
    			self.edge_matrix1 = self.matrix.copy()
    			self.original_edge_count1 = self.edge_count
    			self.original_node_count1 = self.node_count
    			# if(len(self.cip_list) == 0):
    			# 	size = len(self.shortcuts)-4
    			# 	self.shortcut_list = list(itr.combinations(self.shortcuts,len(self.shortcuts)-4))
    		#print(self.cip_list)
    		#input()
    			print(self.matrix)
    			no_of_boundaries = 0
    			count = 0
    			# self.edge_matrix1 = self.matrix.copy()
    			# self.original_edge_count1 = self.edge_count
    			# self.original_node_count1 = self.node_count
    			check = 1
    			for j in self.k4:
    				if(j.case !=2 ):
    					check = 0
    					break
    			for number in range(0,3):
    				to_be_merged_vertices = []
    				rdg_vertices = []
    				rdg_vertices2 =[]
    				
    				for j in self.k4:
    					print(j.vertices)
    					print(j.sep_tri)
    					print(j.interior_vertex)
    					print(j.edge_to_be_removed)
    					if(j.case!=2):
    						print(j.all_edges_to_be_removed[number])
    					print(j.case)   
    					if(j.case  == 2):
    						K4.resolve_K4(self,j,j.edge_to_be_removed,rdg_vertices,rdg_vertices2,to_be_merged_vertices)
    					else:
    						K4.resolve_K4(self,j,j.all_edges_to_be_removed[number],rdg_vertices,rdg_vertices2,to_be_merged_vertices)
    				print(to_be_merged_vertices)
    				print(self.matrix)
    				self.directed = opr.get_directed(self)
    				self.triangles = opr.get_all_triangles(self)
    				self.outer_vertices = opr.get_outer_boundary_vertices(self)[0]
    				self.outer_boundary = opr.get_outer_boundary_vertices(self)[1]
    				self.shortcuts = sr.get_shortcut(self)
    			# self.civ = news.find_cip_single(self)
    				start = time.time()
    				self.boundaries = opr.find_possible_boundary(opr.ordered_outer_boundary(self))
    				self.cip_list = news.populate_cip_list(self)
    				self.edge_matrix = self.matrix.copy()
    				self.original_edge_count = self.edge_count
    				self.original_node_count = self.node_count
    				if(len(self.cip_list) == 0):
    					size = len(self.shortcuts)-4
    					self.shortcut_list = list(itr.combinations(self.shortcuts,len(self.shortcuts)-4))
    				no_of_boundaries = 0
    
    				self.civ = self.cip_list[0]
    				self.original_cip = self.civ.copy()
    				for k in self.cip_list:
    					node_color1 = self.node_color.copy()
    					node_color2 = self.node_color.copy()
    					self.civ = k
    					news.add_news_vertices(self)
    					# print(self.civ)
    					#input()
    					# print("Checking...")
    					# print(self.matrix)
    					if(opr.is_complex_triangle(self) == True):
    						count +=1
    						if(count != len(self.cip_list)):
    							self.node_count = self.original_node_count
    							self.edge_count = self.original_edge_count
    							self.matrix = self.edge_matrix.copy()
    						continue
    					for i in range(0,len(to_be_merged_vertices)):
    						print(rdg_vertices[i])
    						# print(self.node_color)
    						node_color1.append(self.node_color[rdg_vertices[i]])
    						print(node_color1)
    					for i in range(0,len(to_be_merged_vertices)):
    						print(rdg_vertices2[i])
    						node_color2.append(self.node_color[rdg_vertices2[i]])
    						print(node_color2)	
    					print("North Boundary: ", self.civ[0])
    					print("East Boundary: ", self.civ[1])
    					print("South Boundary: ", self.civ[2])
    					print("West Boundary: ",self.civ[3])
    					no_of_boundaries += 1
    					# print("Boundary count: ",no_of_boundaries)
    					self.node_position = nx.planar_layout(nx.from_numpy_matrix(self.matrix))
    					# draw.draw_undirected_graph(self,pen)
    					# input()
    					cntr.initialize_degrees(self)
    					cntr.initialize_good_vertices(self)
    					# print(self.good_vertices)
    					# comb = itr.permutations(self.good_vertices)
    					# for x in comb:
    					# self.good_vertices = list(x)
    					v, u = cntr.contract(self)
    					while v != -1:
    						v, u = cntr.contract(self)
    						# draw.draw_undirected_graph(self,pen)
    						# input()
    					# print(self.contractions)
    					exp.get_trivial_rel(self)
    					while len(self.contractions) != 0:
    						exp.expand(self)
    					rel_matrix =[]
    					rel_matrix.append(self.matrix)
    					self.rdg_vertices.append(rdg_vertices)
    					self.rdg_vertices2.append(rdg_vertices2)
    					self.to_be_merged_vertices.append(to_be_merged_vertices)
    					self.node_color_list.append(node_color1)
    					self.node_color_list.append(node_color2)
    					# draw.draw_directed_graph(self,pen)
    					# print(self.rel_matrix)
    					for i in rel_matrix:
    						self.matrix = i
    						# print(self.user_matrix)
    						flippable_edges = flp.get_flippable_edges(self,i)
    						flippable_vertices = flp.get_flippable_vertices(self,i)[0]
    						flippable_vertices_neighbours = flp.get_flippable_vertices(self,i)[1]
    						# print(flippable_edges)
    						# print(flippable_vertices)
    						for j in range(0,len(flippable_edges)):
    							new_rel = flp.resolve_flippable_edge(flippable_edges[j],self,i)
    							if(not any(np.array_equal(new_rel, i) for i in rel_matrix)):
    								# print("Entered")
    								self.rdg_vertices.append(rdg_vertices)
    								self.rdg_vertices2.append(rdg_vertices2)
    								self.to_be_merged_vertices.append(to_be_merged_vertices)
    								self.node_color_list.append(node_color1)
    								self.node_color_list.append(node_color2)
    								rel_matrix.append(new_rel)
    						for j in range(0,len(flippable_vertices)):
    							# print("Entered")
    							new_rel = flp.resolve_flippable_vertex(flippable_vertices[j],flippable_vertices_neighbours[j],self,i)
    							if(not any(np.array_equal(new_rel, i) for i in rel_matrix)):
    								rel_matrix.append(new_rel)
    								self.rdg_vertices.append(rdg_vertices)
    								self.rdg_vertices2.append(rdg_vertices2)
    								self.to_be_merged_vertices.append(to_be_merged_vertices)
    								self.node_color_list.append(node_color1)
    								self.node_color_list.append(node_color2)
    
    					count +=1
    					if(count != len(self.cip_list)):
    						self.node_count = self.original_node_count
    						self.edge_count = self.original_edge_count
    						self.matrix = self.edge_matrix.copy()
    					for i in rel_matrix:
    						self.rel_matrix.append(i)
    					print("Number of different floor plans: ",len(rel_matrix))
    					print("\n")
    				if(number!=2 and check == 0):
    					self.node_count = self.original_node_count1
    					self.edge_count = self.original_edge_count1
    					self.matrix = self.edge_matrix1.copy()
    					self.north = self.original_north
    					self.west = self.original_west
    					self.east = self.original_east
    					self.south = self.original_south
    					for j in self.k4:
    						j.identified = 0
    				elif(check == 1):
    					break
    
    				
    
    			# print(len(self.to_be_merged_vertices))
    			# print(len(self.rdg_vertices))
    			print("Total number of different floor plans: ",len(self.rel_matrix))
    			print("Total boundaries used:", no_of_boundaries)
    			# for i in self.rel_matrix:
    			#     self.matrix = i
    			#     draw.draw_directed_graph(self,pen)
    			#     input()
    			end = time.time()
    			print(f"Runtime of the program is {end - start}")
    
    
    	
    		# print(self.inp_min)
    		# print(self.inp_area)
    
    
    		# inp_min= []
    		# count = 0
    		if(mode == 1):
    			count = 0
    			origin_count = 1
    			# inp_min = [int(x) for x in input("Enter the minimum width of room: ").strip().split()]
    			# inp_area=[int(x) for x in input("Enter the minimum area of each room: ").strip().split()]
    			for i in self.rel_matrix:
    				# print("Press enter to get a new floorplan")
    				# input()
    				self.matrix = i
    				# print(self.matrix)
    				if(len(self.to_be_merged_vertices)!= 0):
    					# print(self.node_color_list[count])
    					self.node_color = self.node_color_list[count]
    					draw.construct_rdg(self,self.to_be_merged_vertices[count],self.rdg_vertices[count])
    					# if(origin_count != 1):
    					#   self.origin += int((self.room_x[np.where(self.room_x == np.max(self.room_x))] + self.room_width[np.where(self.room_x == np.max(self.room_x))] + 500)[0])
    					# self.room_width_list.append(self.room_width)
    					# self.room_height_list.append(self.room_height)
    					# self.room_x_list.append(self.room_x)
    					# self.room_y_list.append(self.room_y)
    					# self.room_x_bottom_left_list.append(self.room_x_bottom_left)
    					# self.room_x_bottom_right_list.append(self.room_x_bottom_right)
    					# self.room_x_top_left_list.append(self.room_x_top_left)
    					# self.room_x_top_right_list.append(self.room_x_top_right)
    					# self.room_y_left_bottom_list.append(self.room_y_left_bottom)
    					# self.room_y_left_top_list.append(self.room_y_left_top)
    					# self.room_y_right_bottom_list.append(self.room_y_right_bottom)
    					# self.room_y_right_top_list.append(self.room_y_right_top)
    					# print(self.room_x_list,self.room_y_list,self.room_width_list,self.room_height_list,self.room_x_top_left_list,self.room_x_top_right_list,self.room_y_left_top_list,self.room_y_left_bottom_list,self.room_x_bottom_left_list,self.room_x_bottom_right_list,self.room_y_right_top_list,self.room_y_right_bottom_list)
    					# count+=1
    					# origin_count +=1
    					if(origin_count != 1):
    						self.origin += 1000
    					draw.draw_rdg(self,origin_count,pen,self.to_be_merged_vertices[count])
    					origin_count +=1
    					count +=1
    					
    				else:
    					self.node_color = self.node_color_list[origin_count-1]
    					draw.construct_rdg(self,self.to_be_merged_vertices,self.rdg_vertices)
    					if(origin_count != 1):
    						self.origin += 1000
    					draw.draw_rdg(self,origin_count,pen,self.to_be_merged_vertices)
    					origin_count +=1
    					# self.encoded_matrix = opr.get_encoded_matrix(self)
    					# B = copy.deepcopy(self.encoded_matrix)
    					# A = copy.deepcopy(self.encoded_matrix)
    					# [width,height,hor_dgph] = floorplan_to_st(A,self.inp_min,self.inp_area)
    					# A=B
    					# # print(A)
    					# width = np.transpose(width)
    					# height = np.transpose(height)
    					# self.room_width = width.flatten()
    					# self.room_height = height.flatten()
    					# draw.construct_rfp(self,hor_dgph)
    					# for i in range(0,len(self.room_x)):
    					#   self.room_x[i]=round(self.room_x[i],3)
    					#   # print(self.room_x[i])
    					# for i in range(0,len(self.room_y)):
    					#   self.room_y[i]=round(self.room_y[i],3)
    					#   # print(self.room_y[i])
    					# if(origin_count != 1):
    					#   self.origin += 1000
    					# self.room_width_list.append(self.room_width)
    					# self.room_height_list.append(self.room_height)
    					# self.room_x_list.append(self.room_x)
    					# self.room_y_list.append(self.room_y)
    					# self.room_x_bottom_left_list.append(self.room_x_bottom_left)
    					# self.room_x_bottom_right_list.append(self.room_x_bottom_right)
    					# self.room_x_top_left_list.append(self.room_x_top_left)
    					# self.room_x_top_right_list.append(self.room_x_top_right)
    					# self.room_y_left_bottom_list.append(self.room_y_left_bottom)
    					# self.room_y_left_top_list.append(self.room_y_left_top)
    					# self.room_y_right_bottom_list.append(self.room_y_right_bottom)
    					# self.room_y_right_top_list.append(self.room_y_right_top)
    					# print(self.room_x_list,self.room_y_list,self.room_width_list,self.room_height_list,self.room_x_top_left_list,self.room_x_top_right_list,self.room_y_left_top_list,self.room_y_left_bottom_list,self.room_x_bottom_left_list,self.room_x_bottom_right_list,self.room_y_right_top_list,self.room_y_right_bottom_list)
    
    					# origin_count +=1
    					# count+=1
    			# encoded_matrix = opr.get_encoded_matrix(self)
    			# if(not any(np.array_equal(encoded_matrix, i) for i in self.encoded_matrix)):
    				# self.encoded_matrix.append(encoded_matrix)
    				
    			# print(self.matrix)
    			# print(self.user_matrix)
    		
    	def create_multiple_floorplan(self,pen,textbox):
    		global box
    		box = textbox
    		self.create_multiple_dual(0,pen)
    		count = 0
    		origin_count = 1
    		minimum_width = min(self.inp_min)
    		# print(self.to_be_merged_vertices)
    		for i in range(0,len(self.biconnected_vertices)):
    			self.inp_min.append(0)
    			self.ar_min.append(0.5)
    			self.ar_max.append(2)
    		if(len(self.to_be_merged_vertices)!=0):
    			for i in range(0,len(self.to_be_merged_vertices[0])):
    				self.inp_min.append(minimum_width)
    				self.ar_min.append(0.5)
    				self.ar_max.append(2)
    		for i in range(0,len(self.rel_matrix)):
    		# print("Press enter to get a new floorplan")
    		# input()
    			self.matrix = self.rel_matrix[i]
    			# self.room_width = self.room_width_list[i]
    			# self.room_height = self.room_height_list[i]
    			# self.room_x = self.room_x_list[i]
    			# self.room_y = self.room_y_list[i]
    			# self.room_x_bottom_left = self.room_x_bottom_left_list[i]
    			# self.room_x_bottom_right = self.room_x_bottom_right_list[i]
    			# self.room_x_top_left = self.room_x_top_left_list[i]
    			# self.room_x_top_right = self.room_x_top_right_list[i]
    			# self.room_y_left_bottom = self.room_y_left_bottom_list[i]
    			# self.room_y_left_top = self.room_y_left_top_list[i]
    			# self.room_y_right_bottom = self.room_y_right_bottom_list[i]
    			# self.room_y_right_top = self.room_y_right_top_list[i]
    			# print(self.matrix)
    			if(len(self.to_be_merged_vertices)!= 0):
    				self.node_color = self.node_color_list[count]
    				draw.construct_rdg(self,self.to_be_merged_vertices[count],self.rdg_vertices[count])
    				
    				# if(origin_count != 1):
    				#   self.origin += int((self.room_x[np.where(self.room_x == np.max(self.room_x))] + self.room_width[np.where(self.room_x == np.max(self.room_x))] + 500)[0])
    				# draw.draw_rdg(self,origin_count,pen,self.to_be_merged_vertices)
    				# self.room_width_list[i] = self.room_width
    				# self.room_height_list[i] = self.room_height
    				# self.room_x_list[i] = self.room_x
    				# self.room_y_list[i] = self.room_y
    				# self.room_x_bottom_left_list[i] = self.room_x_bottom_left
    				# self.room_x_bottom_right_list[i] = self.room_x_bottom_right
    				# self.room_x_top_left_list[i] = self.room_x_top_left
    				# self.room_x_top_right_list[i] = self.room_x_top_right
    				# self.room_y_left_bottom_list[i] = self.room_y_left_bottom
    				# self.room_y_left_top_list[i] = self.room_y_left_top
    				# self.room_y_right_bottom_list[i] = self.room_y_right_bottom
    				# self.room_y_right_top_list[i] = self.room_y_right_top
    				# print(self.room_x_list,self.room_y_list,self.room_width_list,self.room_height_list,self.room_x_top_left_list,self.room_x_top_right_list,self.room_y_left_top_list,self.room_y_left_bottom_list,self.room_x_bottom_left_list,self.room_x_bottom_right_list,self.room_y_right_top_list,self.room_y_right_bottom_list)
    				self.encoded_matrix = opr.get_encoded_matrix(self)
    				B = copy.deepcopy(self.encoded_matrix)
    				A = copy.deepcopy(self.encoded_matrix)
    				[width,height,hor_dgph] = floorplan_to_st(A,self.inp_min,self.ar_pmt,self.ar_min,self.ar_max)
    				A=B
    				# print(A)
    				width = np.transpose(width)
    				height = np.transpose(height)
    				self.room_width = width.flatten()
    				self.room_height = height.flatten()
    				draw.construct_rfp(self,hor_dgph,self.to_be_merged_vertices[count],self.rdg_vertices[count])
    				for i in range(0,len(self.room_x)):
    					self.room_x[i]=round(self.room_x[i],3)
    					# print(self.room_x[i])
    				for i in range(0,len(self.room_y)):
    					self.room_y[i]=round(self.room_y[i],3)
    					print(self.room_x[i],self.room_y[i],self.room_width[i],self.room_height[i],self.room_x_top_left[i],self.room_x_top_right[i],self.room_y_left_top[i],self.room_y_left_bottom[i],self.room_x_bottom_left[i],self.room_x_bottom_right[i],self.room_y_right_top[i],self.room_y_right_bottom[i])
    				# print(self.room_x,self.room_y,self.room_width,self.room_height,self.room_x_top_left,self.room_x_top_right,self.room_y_left_top,self.room_y_left_bottom,self.room_x_bottom_left,self.room_x_bottom_right,self.room_y_right_top,self.room_y_right_bottom)
    					# print(self.room_y[i])
    				opr.calculate_area(self,self.to_be_merged_vertices[count],self.rdg_vertices[count])
    				draw.draw_rdg(self,(count+1),pen,self.to_be_merged_vertices[count])
    				self.area =[]
    				count+=1
    				origin_count +=1
    				if(origin_count != 1):
    					self.origin += 1000
    				self.node_color = self.node_color_list[count]
    				draw.construct_rfp(self,hor_dgph,self.to_be_merged_vertices[count],self.rdg_vertices2[count])
    				for i in range(0,len(self.room_x)):
    					self.room_x[i]=round(self.room_x[i],3)
    					# print(self.room_x[i])
    				for i in range(0,len(self.room_y)):
    					self.room_y[i]=round(self.room_y[i],3)
    					print(self.room_x[i],self.room_y[i],self.room_width[i],self.room_height[i],self.room_x_top_left[i],self.room_x_top_right[i],self.room_y_left_top[i],self.room_y_left_bottom[i],self.room_x_bottom_left[i],self.room_x_bottom_right[i],self.room_y_right_top[i],self.room_y_right_bottom[i])
    				# print(self.room_x,self.room_y,self.room_width,self.room_height,self.room_x_top_left,self.room_x_top_right,self.room_y_left_top,self.room_y_left_bottom,self.room_x_bottom_left,self.room_x_bottom_right,self.room_y_right_top,self.room_y_right_bottom)
    					# print(self.room_y[i])
    				opr.calculate_area(self,self.to_be_merged_vertices[count],self.rdg_vertices2[count])
    				draw.draw_rdg(self,(count+1),pen,self.to_be_merged_vertices[count])
    				self.area =[]
    				origin_count+=1
    				if(origin_count != 1):
    					self.origin += 500
    				
    				count+=1
    			else:
    				self.node_color = self.node_color_list[count]
    				draw.construct_rdg(self,self.to_be_merged_vertices,self.rdg_vertices)
    				self.encoded_matrix = opr.get_encoded_matrix(self)
    				B = copy.deepcopy(self.encoded_matrix)
    				A = copy.deepcopy(self.encoded_matrix)
    				[width,height,hor_dgph] = floorplan_to_st(A,self.inp_min,self.ar_pmt,self.ar_min,self.ar_max)
    				A=B
    				# print(A)
    				width = np.transpose(width)
    				height = np.transpose(height)
    				self.room_width = width.flatten()
    				self.room_height = height.flatten()
    				draw.construct_rfp(self,hor_dgph,self.to_be_merged_vertices,self.rdg_vertices)
    				for i in range(0,len(self.room_x)):
    					self.room_x[i]=round(self.room_x[i],3)
    					# print(self.room_x[i])
    				for i in range(0,len(self.room_y)):
    					self.room_y[i]=round(self.room_y[i],3)
    					# print(self.room_y[i])
    				if(origin_count != 1):
    					self.origin += 500
    				opr.calculate_area(self,self.to_be_merged_vertices,self.rdg_vertices)
    				draw.draw_rdg(self,(count+1),pen,self.to_be_merged_vertices)
    				self.area =[]
    				origin_count +=1
    				count+=1
    				# self.room_width_list[i] = self.room_width
    				# self.room_height_list[i] = self.room_height
    				# self.room_x_list[i] = self.room_x
    				# self.room_y_list[i] = self.room_y
    				# self.room_x_bottom_left_list[i] = self.room_x_bottom_left
    				# self.room_x_bottom_right_list[i] = self.room_x_bottom_right
    				# self.room_x_top_left_list[i] = self.room_x_top_left
    				# self.room_x_top_right_list[i] = self.room_x_top_right
    				# self.room_y_left_bottom_list[i] = self.room_y_left_bottom
    				# self.room_y_left_top_list[i] = self.room_y_left_top
    				# self.room_y_right_bottom_list[i] = self.room_y_right_bottom
    				# self.room_y_right_top_list[i] = self.room_y_right_top
    				# print(self.room_x_list,self.room_y_list,self.room_width_list,self.room_height_list,self.room_x_top_left_list,self.room_x_top_right_list,self.room_y_left_top_list,self.room_y_left_bottom_list,self.room_x_bottom_left_list,self.room_x_bottom_right_list,self.room_y_right_top_list,self.room_y_right_bottom_list)
    
    
    
    	  
    		
    
    
    
    ##----- End ptpg.py ----------------------------------------------------------##
    return locals()

@modulize('plot_graphs')
def _plot_graphs(__name__):
    ##----- Begin plot_graphs.py -------------------------------------------------##
    import os
    import matplotlib.pyplot as plt
    import networkx as nx
    import warnings
    warnings.simplefilter("ignore")
    # from generate import generation_next
    
    
    def plot_graphs(init_len, graph_list):
        folder_path = f"RFP_Graph_Plots/Len {init_len}"
        os.makedirs(folder_path, exist_ok=True)
        graph_no = 1
        # _fig, _ax = plt.subplots()
        for graph in graph_list:
            plt.figure(graph_no)
            try:
                nx.draw_planar(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
            except:
                nx.draw(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
            # Inner Graph IG_<Initial Vertices>_<Graph_Size>_<Graph No.>.png
            plt.savefig(f'{folder_path}/graph_{init_len}_{graph.size()}_{graph_no}.png')
            plt.show()
            
            graph_no += 1
            plt.clf()
    
    
    # g=nx.path_graph(5)
    # plot_graphs(len(g.nodes),[g])
    
    ##----- End plot_graphs.py ---------------------------------------------------##
    return locals()

@modulize('NESW')
def _NESW(__name__):
    ##----- Begin NESW.py --------------------------------------------------------##
    import networkx as nx # Networkx Library
    import numpy as np
    import matplotlib.pyplot as plt # Matplot lib
    
    
    #  NESW Function
    def num_cips(G):
        H = G.to_directed()
        # H = G.copy()
        # Get all triangles
        all_cycles = list(nx.simple_cycles(H))
        all_triangles = []
        for cycle in all_cycles:
            if len(cycle) == 3:
                all_triangles.append(cycle)
    
        # Get edges on outer boundary
        outer_boundary = []
        for edge in H.edges:
            count = 0
            for triangle in all_triangles:
                if edge[0] in triangle and edge[1] in triangle:
                    count += 1
            if count == 2:
                outer_boundary.append(edge)
    
        # Get Vertex-Set of outerboundary
        outer_vertices = []
        for edge in outer_boundary:
            if edge[0] not in outer_vertices:
                outer_vertices.append(edge[0])
            if edge[1] not in outer_vertices:
                outer_vertices.append(edge[1])
    
        # Get top,left,right and bottom boundaries of graph
        civ = []
        # Finds all corner implying paths in the graph
        while len(outer_vertices) > 1:
            cip_store = [outer_vertices[0]]	#stores the corner implying paths
            outer_vertices.pop(0)
            for vertices in cip_store:
                for vertex in outer_vertices:
                    cip_store_copy = cip_store.copy()
                    cip_store_copy.pop(len(cip_store) - 1)
                    if (cip_store[len(cip_store) - 1], vertex) in outer_boundary:
                        cip_store.append(vertex)
                        outer_vertices.remove(vertex)
                        if cip_store_copy is not None:	#checks for existence of shortcut
                            for vertex1 in cip_store_copy:
                                if (vertex1, vertex) in H.edges:
                                    cip_store.remove(vertex)
                                    outer_vertices.append(vertex)
                                    break
            civ.append(cip_store)		#adds the corner implying path to civ
            outer_vertices.insert(0, cip_store[len(cip_store) - 1])	#handles the last vertex of the corner implying path added
            if len(outer_vertices) == 1:		#works for the last vertex left in the boundary
                 last_cip=0
                 first_cip=0
                 merge_possible =0
                 for test in civ[len(civ)-1]:			#checks last corner implying path
                     if((test,civ[0][0]) in H.edges and (test,civ[0][0]) not in outer_boundary ):
                         last_cip = 1
                         first_cip = 0
                         break
                 for test in civ[0]:				#checks first corner implying path
                     if((test,outer_vertices[0]) in H.edges and (test,outer_vertices[0]) not in outer_boundary):
                         last_cip = 1
                         first_cip = 1
                         break
                 if last_cip == 0 and len(civ)!=2:		#if merge is possible as well as both civs are available for last vertex
                     for test in civ[len(civ)-1]:
                         for test1 in civ[0]:
                             if ((test,test1) in H.edges and (test,test1) not in H.edges):
                                 merge_possible = 1
                     if(merge_possible == 1):                  #adding last vertex to last civ
                         civ[len(civ)-1].append(civ[0][0])
                     else:                                     #merging first and last civ
                         civ[0] = civ[len(civ)-1] + list(set(civ[0]) - set(civ[len(civ)-1]))
                         civ.pop()
                 elif(last_cip == 0 and len(civ)==2):	      #if there are only 2 civs
                     civ[len(civ)-1].append(civ[0][0])
                 elif (last_cip ==1 and first_cip == 0):      #adding last vertex to first civ
                     civ[0].insert(0,outer_vertices[0])
                 elif (last_cip ==0 and first_cip == 1):      #adding last vertex to last civ
                     civ[len(civ)-1].append(civ[0][0])
                 elif (last_cip == 1 and first_cip == 1):     #making a new corner implying path
                     civ.append([outer_vertices[0],civ[0][0]])
        #print("Number of corner implying paths: ", len(civ))
        #print("Corner implying paths: ", civ)
        return len(civ)
    
    
    
    
    
    ##----- End NESW.py ----------------------------------------------------------##
    return locals()

@modulize('tests')
def _tests(__name__):
    ##----- Begin tests.py -------------------------------------------------------##
    import networkx as nx
    from networkx.algorithms.isomorphism import GraphMatcher
    import matplotlib.pyplot as plt
    from plot_graphs import plot_graphs
    from NESW import num_cips
    def print(string):
        box.insert('end',string)
        box.insert('end',"\n")
    def tester(graph,textbox):
        global box
        box = textbox
        check= True
    
        if not planarity_check(graph,textbox):
            check = False
        
        elif not complex_triangle_check(graph,textbox):
            check = False
    
        else:
            if not cip_rule_check(graph,textbox):
                check = False
            
            # if  not civ_rule_check(graph,textbox):
            #     check=False
            #     textbox.insert('end',"\n")
                # print("=> civ rule failed\n")
    
        if check:
            print("=> RFP exists\n")
            # plot_graphs(len(graph),[graph])
            # plot_graph([graph])
        else:
            print("=> RFP doesn't exist\n")
            # plot_graphs(len(graph),[graph])
    
    def planarity_check(given_graph,textbox):
        global box
        box = textbox
        # Planarity Check
        if not nx.check_planarity(given_graph)[0]:
            print("=> The graph is non-planar\n")
            return 0
        return 1
    
    def cip_rule_check(graph,textbox):
        global box
        box = textbox
        """
        Given A Graph
        Returns true if civ rule is satisfied
    
        CIP Rule = 2 CIP on outer biconnected components and no CIP on
        inner biconnected components
        """
        cip_check = True
        outer_comps, inner_comps, single_component = component_break(graph,textbox)
        if not single_component:
            # CIP Rule for Outer Components
            if len(outer_comps)>2:
                print("BNG is not a path graph\n")
                return False
            for comp in outer_comps:
                print(f"Checking biconnected component {list(comp)}")
                if num_cips(comp) > 2 :
                    cip_check = False
                    print(f"    Num civs ={num_cips(comp)}\n")
                    print(f"    Maximum possible civ =2\n")
                    print('Invalid')
                else:
                    print('Valid')
            # CIP Rule for Inner Components
            for comp in inner_comps:
                print(f"Checking biconnected component {list(comp)}")
                if num_cips(comp) > 0 :
                    cip_check = False
                    print(f"    Num civs ={num_cips(comp)}\n")
                    print(f"    Maximum possible civ =0\n")
                    print('Invalid')
                else:
                    print('Valid')
        else:
            # CIP Rule for single_component Components
            print(f"Checking biconnected component {list(single_component)}")
            if num_cips(single_component) > 4:
                cip_check = False
                print(f"    Num civs ={num_cips(single_component)}")
                print(f"    Maximum possible civ =4\n")
                print('Invalid')
            else:
                print('Valid')
        if not cip_check:
            print("=> civ rule failed\n")
        return cip_check
    
    def component_break(given_graph,textbox):
        global box
        box = textbox
        """
        Given a graph,
        returns [list of the 2 outer components(1 articulation point) with 2cip],
        [list of other inner components(2 articulation points) with 0 civ]
        """
        test_graph = given_graph.copy()
        cutvertices = list(nx.articulation_points(test_graph))
        inner_components = []
        outer_components = []
        if len(cutvertices) == 0:
            single_component = test_graph
            return 0, 0, single_component
        for peice_edges in nx.biconnected_component_edges(test_graph):
            peice = nx.Graph()
            peice.add_edges_from(list(peice_edges))
            num_cutverts = 0
            for cutvert in cutvertices:
                if cutvert in peice.nodes():
                    num_cutverts += 1
    
            if num_cutverts == 2:
                inner_components.append(peice)
            elif num_cutverts == 1:
                outer_components.append(peice)
            else:
                print("Not a PTPG\n")
                print(test_graph.edges())
    
            if len(outer_components) > 2:
                # print("BNG is not a path\n")
                print(test_graph)
        return outer_components, inner_components, 0
    
    def complex_triangle_check(graph,textbox):
        global box
        box = textbox
        for compedges in nx.biconnected_component_edges(graph):
            comp=nx.Graph()
            comp.add_edges_from(compedges)
            H = comp.to_directed()
            all_cycles = list(nx.simple_cycles(H))
            all_triangles = []
            for cycle in all_cycles:
                if len(cycle) == 3:
                    all_triangles.append(cycle)
    
            if (comp.size() - len(comp) +1) > (len(all_triangles)/2):
                print("=> Not triangled\n")
                return False
            elif (comp.size() - len(comp) +1) < (len(all_triangles)/2):
                print("=> complex triangle exists\n")
                return False
        return True
    
    def civfinder(g,textbox):
        global box
        box = textbox
        
        # nx.draw_planar(g,labels=None)
        # print(g.edges)
        corner_set = []
        for ver in g.nodes:
            if nx.degree(g,ver)==2:
                corner_set.append(ver)
            else:
                if nx.degree(g,ver)==1:
                    corner_set.append(ver)
                    corner_set.append(ver)
        # print(f"    corner set = {corner_set}\n")
        return corner_set
    def civ_rule_check(graph,textbox):
        global box
        box = textbox
        civ_check = True
        outer_comps, inner_comps, single_component = component_break(graph,textbox)
        # list, list, element
        # print(outer_comps, inner_comps, single_component)
    
        if not single_component:
            if len(outer_comps) >2:
                print("BNG is not a path graph\n")
                return False
            cut= nx.articulation_points(graph)
            cut= set(cut)
            # CIP Rule for Outer Components
            for comp in outer_comps:
                cut1= []
                for i in cut:
                    if i in comp:
                        cut1.append(i)
                civ_check= eachcompciv(comp,cut1,2,textbox)
            # CIP Rule for Inner Components
            for comp in inner_comps:
                cut1= []
                for i in cut:
                    if i in comp:
                        cut1.append(i)
                civ_check= eachcompciv(comp,cut1,0,textbox)
        else:
            # CIP Rule for single_component Components
            cut = []
            civ_check= eachcompciv(single_component,cut,4,textbox)
        return civ_check
    
    def eachcompciv(graph,cut,maxciv,textbox):
        global box
        box = textbox
        # print(f"checking component {list(graph)}\n")
        set1= set(cut)
        # print(f"    cut set = {list(set1)}\n")
        y= [ i for i in list(civfinder(graph,textbox)) if i not in set1]
        x=len(y)
            # print(f"    num civs ={x}\n")
            # print(f"    maximum possible civ ={maxciv}\n")
        if x>maxciv :
            # print("    component failed\n")
            return False
            # print("    true\n")
        return True
    
    # g=nx.Graph()
    # g.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5)]) #wrapped vertex attached
    # g.add_edges_from([(0, 1), (1, 2),  (2, 3), (0,3),(1,3),(0,2)]) #complex triangle
    # g.add_edges_from([(0, 1), (1, 2),  (2, 3), (3,4),(1,3)]) #civ rule
    # g.add_edges_from([(0, 1), (1, 2),  (2, 3), (3,0),(2,5),(2,4),(3,5),(3,4),(4,5)]) #1 non-triangle with 1 ST
    
    # g.add_edges_from([(0, 1), (1, 2), (0,2),(3,4),(4,5),(5,3),(0,3),(0,4),(0,5),(2,4),(1,4),(2,5)]) #k4 subgraph
    # g.add_edges_from([(0, 1), (0,3),(1,2),(2,3),(1,3),(0,4),(4,1),(1,5),(2,5),(2,6),(3,6),(3,7),(0,7),(3,8),(6,8),(2,9),(6,9)]) #5 civs
    # g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4),
    #  (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)])#complex without K4 .. 2 lines
    # g.add_edges_from([(0, 1), (1, 2),(1,3),(1,4)])
    # RFPchecker(g)
    
    
    
    
    
    ##----- End tests.py ---------------------------------------------------------##
    return locals()

@modulize('check')
def _check(__name__):
    ##----- Begin check.py -------------------------------------------------------##
    import networkx as nx
    import tests
    def print(string):
        box.insert('end',string)
        box.insert('end',"\n")
    def checker(value,textbox):
        global box
        box = textbox
        print("Edge set")
        # textbox.insert('end',"Enter the edge set")
        edgeset = []
        edgeset=value[2]
        # for i in range(int(input("No. of Edges: "))):
            # edgeset.append(list(map(int, input().split())))
        g=nx.Graph()
        g.add_edges_from(edgeset)
        print(g.edges)
        # textbox.insert('end',g.edges)
        # textbox.insert('end',"\n")
        tests.tester(g,textbox)
    
    
    
    
    
    ##----- End check.py ---------------------------------------------------------##
    return locals()

@modulize('plotter')
def _plotter(__name__):
    ##----- Begin plotter.py -----------------------------------------------------##
    import networkx as nx
    import matplotlib.pyplot as plt
    import ptpg
    import tkinter as tk
    import turtle
    def plot(cir,m):
        pos=nx.spring_layout(cir) # positions for all nodes
        nx.draw_networkx(cir,pos, labels=None,node_size=400 ,node_color='#4b8bc8',font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1, bbox=None, ax=None)
        nx.draw_networkx_edges(cir,pos)
        nx.draw_networkx_nodes(cir,pos,
                            nodelist=list(range(m,len(cir))),
                            node_color='r',
                            node_size=500,
                        alpha=1)
        plt.show()
    
    def RFP_plot(spanned):
        
        G= ptpg.PTPG(spanned)
    
        root_window = tk.Tk()
        root_window.title('Rectangular Dual')
        root_window.geometry(str(1366) + 'x' + str(700))
        root_window.resizable(0, 0)
        root_window.grid_columnconfigure(0, weight=1, uniform=1)
        root_window.grid_rowconfigure(0, weight=1)
    
        border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}
    
        canvas = tk.Canvas(root_window, **border_details,width = 100000000000, height =2000)
        canvas.pack_propagate(0)
        canvas.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
    
        scroll_x = tk.Scrollbar(root_window, orient="horizontal", command=canvas.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")
    
        scroll_y = tk.Scrollbar(root_window, orient="vertical", command=canvas.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
    
        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        canvas.configure(scrollregion=canvas.bbox("all"))
    
        pen = turtle.RawTurtle(canvas)
        pen.speed(100)
    
        G.create_single_dual(1,pen)
    
        # screenshot = pyautogui.screenshot()
        # screenshot.save("screen.png")
    
        root_window.mainloop()
    
    
    ##----- End plotter.py -------------------------------------------------------##
    return locals()

@modulize('circulation')
def _circulation(__name__):
    ##----- Begin circulation.py -------------------------------------------------##
    import networkx as nx
    import matplotlib.pyplot as plt
    from plotter import plot
    def span(graph):
        n=len(graph)
        m = n
    
        # nx.draw_spring(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
        # plt.show()
        print("choose a door")
        k ,j = map(int, input().split())
        n=adder(k,j,graph,n,-1,m)
        return graph
    
    def adder(i,j,graph,n,br,m):
        
        for ne in nx.common_neighbors(graph,i,j):
            if ne < m :
                graph.add_edges_from([(n,i),(n,j)])
                graph.remove_edge(i,j)
                graph.add_edge(n,ne)
                if br>0:
                    graph.add_edge(n,br)
                # plot(graph,m)                  #can remove this line if not required to plot
                n+=1
                prev=n-1
                n=adder( ne,i,graph,n,n-1,m)
                n=adder( ne,j,graph,n,prev,m)
    
        return n
                
    
    def BFS(graph):
        # nx.draw_spring(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
        # plt.show()
    
        n = len(graph)
        m = n
        s = (0 ,1 , -1)
    
        # print("choose a door")
        # i ,j = map(int, input().split())
        # s[0] = i
        # s[1] = j
        queue = []
        queue.append(s)
    
        while ( queue ):
            s = queue.pop(0)
            for ne in nx.common_neighbors(graph,s[0],s[1]):
                if ne < m :
                    graph.add_edge(s[0],n)
                    graph.add_edge(s[1],n)
                    graph.remove_edge(s[0],s[1])
                    if s[2]>0:
                        graph.add_edge(n,s[2])
                    graph.add_edge(n,ne)
                    # plot(graph,m)  
                    n+=1
                    queue.append((ne,s[0],n-1))
                    queue.append((ne,s[1],n-1))
        return graph
        
    
    
    ##----- End circulation.py ---------------------------------------------------##
    return locals()

@modulize('screen')
def _screen(__name__):
    ##----- Begin screen.py ------------------------------------------------------##
    import tkinter as tk
    import turtle
    
    def create_pen():
        # root = tk.Tk()
        root_window = tk.Tk()
        root_window.state('zoomed')
        # root_window = tk.Frame(root,width=1000,height=700)
        l1 = tk.Label(root_window, text = "Rectangular dual")
    
        l1.grid(row=1,column=0)
        
        # root_window.geometry(str(1366) + 'x' + str(700))
        # root_window.resizable(0, 0)
        root_window.grid_columnconfigure(0, weight=1, uniform=1)
        root_window.grid_rowconfigure(0, weight=1)
        border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}
    
        canvas = tk.Canvas(root_window, **border_details,width = 100, height =200)
        canvas.pack_propagate(0)
        canvas.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)
    
        scroll_x = tk.Scrollbar(root_window, orient="horizontal", command=canvas.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")
    
        scroll_y = tk.Scrollbar(root_window, orient="vertical", command=canvas.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
    
        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        canvas.configure(scrollregion=canvas.bbox("all"))
    
        pen = turtle.RawTurtle(canvas)
        pen.speed(100)
        # root_window.mainloop()
        return pen, root_window
    
    
    if __name__ == '__main__':
        pen , root = create_pen()
        root.mainloop()
    ##----- End screen.py --------------------------------------------------------##
    return locals()

@modulize('main')
def _main(__name__):
    ##----- Begin main.py --------------------------------------------------------##
    import networkx as nx
    import warnings
    import numpy as np
    import ptpg
    import shortcutresolver as sr
    import news
    import time
    from random import randint
    import drawing as draw 
    import tkinter as tk
    import turtle
    import ptpg
    import contraction as cntr
    import expansion as exp
    import operations as opr
    import flippable
    import networkx as nx
    import gui
    import warnings
    import check
    import circulation
    import plotter
    import screen
    def printe(string):
        gclass.textbox.insert('end',string)
        gclass.textbox.insert('end',"\n")
    
    warnings.filterwarnings("ignore") 
    gclass = gui.gui_class()
    
    while (gclass.command!="end"):
    	G = ptpg.PTPG(gclass.value)
    	# print(gclass.command)
    	# print(G.graph.edges())
    	if ( gclass.command =="checker"):
    		check.checker(gclass.value,gclass.textbox)
    	else:
    		printe("\nEdge Set")
    		printe(G.graph.edges())
    		frame = tk.Frame(gclass.root)
    		frame.grid(row=2,column=1)
    		if (gclass.command == "circulation"):
    			m =len(G.graph)
    			spanned = circulation.BFS(G.graph)
    			# plotter.plot(spanned,m)
    			colors= gclass.value[6].copy()
    			for i in range(0,100):
    				colors.append('#FF4C4C')
    			# print(colors)
    			rnames = G.room_names
    			rnames.append(tk.StringVar(None,value="Corridor"))
    			for i in range(0,100):
    				rnames.append(tk.StringVar(None,value=""))
    			# print(rnames)
    			
    			parameters= [len(spanned), spanned.size() , spanned.edges() , 0,0 ,rnames,colors]
    			C = ptpg.PTPG(parameters)
    			C.create_single_dual(1,gclass.pen,gclass.textbox)
    
    		elif(gclass.command == "single"):
    			# root_window.state('zoomed')
    			if(G.dimensioned == 0):
    				G.create_single_dual(1,gclass.pen,gclass.textbox)
    			else:
    				G.create_single_floorplan(gclass.pen,gclass.textbox)
    		elif(gclass.command == "multiple"):
    			if(G.dimensioned == 0):
    				G.create_multiple_dual(1,gclass.pen,gclass.textbox)
    			else:
    				G.create_multiple_floorplan(gclass.pen,gclass.textbox)
    
    	gclass.root.wait_variable(gclass.end)
    	gclass.graph_ret()
    	gclass.ocan.tscreen.resetscreen()
    	gclass.pen.speed(100000)
    ##----- End main.py ----------------------------------------------------------##
    return locals()

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
    cip = []
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
        cip.append(cip_store)       #adds the corner implying path to cip
        outer_vertices.insert(0, cip_store[len(cip_store) - 1]) #handles the last vertex of the corner implying path added
        if len(outer_vertices) == 1:        #works for the last vertex left in the boundary
            last_cip=0
            first_cip=0
            merge_possible =0
            for test in cip[len(cip)-1]:            #checks last corner implying path
                if((test,cip[0][0]) in H.edges and (test,cip[0][0]) not in outer_boundary ):
                    last_cip = 1
                    first_cip = 0
                    break
            for test in cip[0]:             #checks first corner implying path
                if((test,outer_vertices[0]) in H.edges and (test,outer_vertices[0]) not in outer_boundary):
                    last_cip = 1
                    first_cip = 1
                    break
            if last_cip == 0 and len(cip)!=2:       #if merge is possible as well as both cips are available for last vertex
                for test in cip[len(cip)-1]:
                    for test1 in cip[0]:
                        if ((test,test1) in H.edges and (test,test1) not in H.edges):
                            merge_possible = 1
                if(merge_possible == 1):                  #adding last vertex to last cip
                    cip[len(cip)-1].append(cip[0][0])
                else:                                     #merging first and last cip
                    cip[0] = cip[len(cip)-1] + list(set(cip[0]) - set(cip[len(cip)-1]))
                    cip.pop()
            elif(last_cip == 0 and len(cip)==2):          #if there are only 2 cips
                cip[len(cip)-1].append(cip[0][0])
            elif (last_cip ==1 and first_cip == 0):      #adding last vertex to first cip
                cip[0].insert(0,outer_vertices[0])
            elif (last_cip ==0 and first_cip == 1):      #adding last vertex to last cip
                cip[len(cip)-1].append(cip[0][0])
            elif (last_cip == 1 and first_cip == 1):     #making a new corner implying path
                cip.append([outer_vertices[0],cip[0][0]])

    print(cip)

    if(len(sr.get_shortcut(graph))==0):
        cip.append(cip[0]+cip[1])
        cip[len(cip)-1].pop(len(cip[0]))
        cip.pop(0)
        cip.pop(0)
    if(len(cip)<4):
        for i in range(4-len(cip)):
            index = cip.index(max(cip,key =len))
            # print(index)
            create_cip(cip,index)
    return cip

def find_cip_single(graph):
    boundary_vertices = opr.ordered_outer_boundary(graph)
    boundary_vertices.append(boundary_vertices[0])
    outer_boundary = opr.get_outer_boundary_vertices(graph)[1]
    cip = []
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
            cip.append(temp)
            temp = []
            temp.append(value)
            temp.append(boundary_vertices[i])
        else:
            temp.append(boundary_vertices[i])
    cip.append(temp)
    merge = 0
    if(len(cip)>1):
        for i in cip[len(cip)-1]:
            for j in cip[0]:
                if (i,j) not in outer_boundary and graph.matrix[i][j] == 1:
                    merge = 1
                    break
            if merge == 1:
                break
        if merge == 0:
            for i in range(1,len(cip[0])):
                cip[len(cip)-1].append(cip[0][i])
            cip.pop(0)
    print(cip)
    if(len(cip)<4):
        for i in range(4-len(cip)):
            index = cip.index(max(cip,key =len))
            # print(index)
            create_cip(cip,index)
    return cip

# get four corner implying paths in case there are less than 4 cips
def create_cip(cip,index):
    cip.insert(index + 1, cip[index])
    length = int(len(cip[index])/2)
    cip[index] = cip[index][0:2]
    del cip[index + 1][0:1]

# connect cips to north, east, west and south vertices
def news_edges(graph,matrix,cip, source_vertex):
    for vertex in cip:
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
    cip = graph.cip
    # print(cip)
    if(len(cip)>4):
        shortcut = sr.get_shortcut(graph)
        print('Shortcut:')
        print(shortcut)
        while(len(shortcut)>4):
            index = randint(0,len(shortcut)-1)
            sr.remove_shortcut(shortcut[index],graph,graph.rdg_vertices,graph.rdg_vertices2,graph.to_be_merged_vertices)
            shortcut.pop(index)
        print(shortcut)
        cip = find_cip_single(graph)
    # if(len(cip)<4):
    #     for i in range(4-len(cip)):
    #         index = cip.index(max(cip,key =len))
    #         # print(index)
    #         create_cip(cip,index)
            # print(cip)
    # print(cip)
    # cip = [[6,0,4],[4,1,5],[5,7],[7,2,6]]
    graph.node_count += 4
    new_adjacency_matrix = np.zeros([graph.node_count, graph.node_count], int)
    new_adjacency_matrix[0:graph.matrix.shape[0],0:graph.matrix.shape[1]] = graph.matrix
    news_edges(graph,new_adjacency_matrix,cip[0], graph.north)
    news_edges(graph,new_adjacency_matrix,cip[1], graph.east)
    news_edges(graph,new_adjacency_matrix,cip[2], graph.south)
    news_edges(graph,new_adjacency_matrix,cip[3], graph.west)
    graph.edge_count += 4
    connect_news(new_adjacency_matrix,graph)
    graph.matrix = new_adjacency_matrix.copy()
    graph.user_matrix = new_adjacency_matrix



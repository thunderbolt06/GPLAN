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

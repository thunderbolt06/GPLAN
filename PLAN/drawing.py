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
def draw_rdg(graph,count,pen,to_be_merged_vertices,mode):
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
            # if(mode == 2):
            #     pen.color('red')
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
    
    if mode == 2:
        for i in to_be_merged_vertices:
            pen.color('red')
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])

            pen.pendown()
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y[i]) * scale + origin['y'])
            pen.setposition((graph.room_x[i] + graph.room_width[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y[i]+ graph.room_height[i]) * scale + origin['y'])
            pen.setposition((graph.room_x[i]) * scale + origin['x'],
                                (graph.room_y_left_top[i]) * scale + origin['y'])
            pen.setposition(graph.room_x[i] * scale + origin['x'], graph.room_y[i] * scale + origin['y'])
            pen.penup()

    for i in range(graph.room_x.shape[0]):
        pen.color('black')
        if(i not in to_be_merged_vertices):
            pen.setposition(((2 * graph.room_x[i]) * scale / 2) + origin['x']+5,
                            ((2 * graph.room_y[i] + graph.room_height[i]) * scale / 2) + origin['y'])
            pen.write(graph.room_names[i])
            pen.penup()
        if(i in to_be_merged_vertices and mode == 2):
            pen.setposition(((2 * graph.room_x[i] ) * scale / 2) + origin['x'] + 5,
                            ((2 * graph.room_y[i] + graph.room_height[i]) * scale / 2) + origin['y'])
            # pen.write(i)
            pen.penup()

    pen.color('black')
    value = 1
    if(len(graph.area) != 0):
        pen.setposition(dim[0]* scale + origin['x']+50, dim[1]* scale + origin['y']-30)
        pen.write('         Area   Height   Width' ,font=("Arial", 20, "normal"))
        for i in range(0,len(graph.area)):
            
            pen.setposition(dim[0]* scale + origin['x']+50, dim[1]* scale + origin['y']-30-value*30)
            pen.write(graph.room_names[i] + ': \t'+ str(graph.area[i]) + "\t" + str(round(graph.room_height[i],2))+ "\t"+ str(round(graph.room_width[i],2)),font=("Arial", 15, "normal"))
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
            pen.write(graph.room_names[i])
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
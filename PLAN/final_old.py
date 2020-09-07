import math
import sys
import time
import tkinter as tk
# from tkinter import Tkconstants
import tkinter.messagebox
import turtle
import warnings
from functools import partial
from pprint import pprint
# from tkinter import *
import main
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# sys.path.insert(0, 'Min_Area')
import scipy.optimize
from networkx.algorithms import bipartite

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

# -------------------- input.py -------------------- #

def moveType(a, b, c, d):
    if a == c and d > b:
        return "RIGHT"
    elif a == c and d <= b:
        return "LEFT"
    elif a <= c and d == b: 
        return "DOWN"
    elif a > c and d == b:
        return "UP"

def preProcessVertices(vertices):
    new_vertices = []
    for i in range(len(vertices)-1):
        a, b = vertices[i]
        c, d = vertices[i+1]
        if moveType(a, b, c, d) == "RIGHT":
            for i in range(b, d):
                new_vertices.append([a, i])
        elif moveType(a, b, c, d) == "LEFT":
            for i in range(b, d, -1):
                new_vertices.append([a, i])
        elif moveType(a, b, c, d) == "UP":
            for i in range(a, c, -1):
                new_vertices.append([i, d])
        elif moveType(a, b, c, d) == "DOWN":
            for i in range(a, c):
                new_vertices.append([i, d])
    new_vertices.append(vertices[0])
    return new_vertices

def getVerticesType(vertices):
    concave = []
    collinear = []
    movement = []
    for i in range(len(vertices)-1):
        a, b = vertices[i]
        c, d = vertices[i+1]
        movement.append(moveType(a, b, c, d))
        
    for i in range(len(movement)-1):
        if (movement[i], movement[i+1]) == ("DOWN", "RIGHT") \
        or (movement[i], movement[i+1]) == ("LEFT", "DOWN") \
        or (movement[i], movement[i+1]) == ("UP", "LEFT") \
        or (movement[i], movement[i+1]) == ("RIGHT", "UP"):
            concave.append(i+1)
        if (movement[i], movement[i+1]) == ("DOWN", "DOWN") \
        or (movement[i], movement[i+1]) == ("LEFT", "LEFT") \
        or (movement[i], movement[i+1]) == ("UP", "UP") \
        or (movement[i], movement[i+1]) == ("RIGHT", "RIGHT"):
            collinear.append(i+1)
    return collinear, concave
        
def compute_partition(p, viz=False):
    # x and y contain list of x and y coordinates respectively
    p = preProcessVertices(p)
    x = []
    y = []
    for i,j in p:
        x.append(i)
        y.append(j)

    # separating concave and collinear vertices
    collinear_vertices, concave_vertices = getVerticesType(p)
    
    # finding the chords inside the polygon
    horizontal_chords = []
    vertical_chords = []

    # middles is used because, there are cases when there is a chord between vertices
    # and they intersect with external chords, hence if there is any vertex in between 
    # two vertices then skip that chord. 
    for i in range(len(concave_vertices)):
        for j in range(i+1,len(concave_vertices)):
            if concave_vertices[j] != concave_vertices[i] + 1:
                middles = []
                if y[concave_vertices[i]] == y[concave_vertices[j]]:
                    for k in range(len(x)):
                        if y[concave_vertices[i]] == y[k] and (x[concave_vertices[i]] < x[k] and x[concave_vertices[j]] > x[k] \
                                                              or x[concave_vertices[i]] > x[k] and x[concave_vertices[j]] < x[k]):
                            middles.append(k)
                    if len(middles) == 0:
                        horizontal_chords.append((concave_vertices[i],concave_vertices[j]))
                middles = []
                if x[concave_vertices[i]] == x[concave_vertices[j]]:
                    for k in range(len(x)):
                        if x[concave_vertices[i]] == x[k] and (y[concave_vertices[i]] < y[k] and y[concave_vertices[j]] > y[k] \
                                                              or y[concave_vertices[i]] > y[k] and y[concave_vertices[j]] < y[k]):
                            middles.append(k)
                    if len(middles) == 0:
                        vertical_chords.append((concave_vertices[i],concave_vertices[j]))
            
    temp_hori = horizontal_chords[:]
    temp_verti = vertical_chords[:]

    for i in range(len(collinear_vertices)):
        for j in range(len(concave_vertices)):
            middles = []
            if y[collinear_vertices[i]] == y[concave_vertices[j]]:
                if collinear_vertices[i] < concave_vertices[j]:
                    for k in range(len(x)):
                        if y[k] == y[collinear_vertices[i]] and (x[k] < x[concave_vertices[j]] \
                            and x[k] > x[collinear_vertices[i]] or x[k] > x[concave_vertices[j]] \
                            and x[k] < x[collinear_vertices[i]]):
                            middles.append(k)
                    if collinear_vertices[i]+1 == concave_vertices[j]:
                        middles.append(0)
                else:
                    for k in range(len(x)):
                        if y[k] == y[collinear_vertices[i]] and (x[k] > x[concave_vertices[j]] \
                            and x[k] < x[collinear_vertices[i]] or x[k] < x[concave_vertices[j]] \
                            and x[k] > x[collinear_vertices[i]]):
                            middles.append(k)
                    if collinear_vertices[i] == concave_vertices[j]+1:
                        middles.append(0)
                if len(middles) == 0:
                    horizontal_chords.append((collinear_vertices[i],concave_vertices[j]))
            middles = []
            if x[collinear_vertices[i]] == x[concave_vertices[j]]:
                if collinear_vertices[i] < concave_vertices[j]:
                    for k in range(len(x)):
                        if x[k] == x[collinear_vertices[i]] and (y[k] < y[concave_vertices[j]] \
                            and y[k] > y[collinear_vertices[i]] or y[k] > y[concave_vertices[j]] \
                            and y[k] < y[collinear_vertices[i]]):
                            middles.append(k)
                    if collinear_vertices[i]+1 == concave_vertices[j]:
                        middles.append(0)
                else:
                    for k in range(len(x)):
                        if x[k] == x[collinear_vertices[i]] and (y[k] > y[concave_vertices[j]] \
                            and y[k] < y[collinear_vertices[i]] or y[k] < y[concave_vertices[j]] \
                            and y[k] > y[collinear_vertices[i]]):
                            middles.append(k)
                    if collinear_vertices[i] == concave_vertices[j]+1:
                        middles.append(0)
                if len(middles) == 0:
                    vertical_chords.append((collinear_vertices[i],concave_vertices[j]))    
    
    # displaying all attributes and important parameters involved
    # plotting the initial input given
    if viz:
        print ("Initial input rectillinear graph")
        fig, ax = plt.subplots()
        plt.gca().invert_yaxis()
        ax.plot(y, x, color='black')
        ax.scatter(y, x, color='black')
        for i in range(len(x)):
            ax.annotate(i, (y[i],x[i]))
        plt.show()
        plt.clf()
    
        print("collinear_vertices = ", collinear_vertices)
        print("concave_vertices =", concave_vertices)
        print("horizontal_chords = " ,horizontal_chords)
        print("vertical_chords = ",vertical_chords)
        
        # drawing the maximum partitioned polygon 
        print("The maximum partitioned rectillinear polygon")
        fig, ax = plt.subplots()
        ax.plot(y, x, color='black')
        ax.scatter(y, x, color='black')
        plt.gca().invert_yaxis()
        for i in range(len(x)):
            ax.annotate(i, (y[i],x[i]))
        for i,j in horizontal_chords:
            ax.plot([y[i],y[j]],[x[i],x[j]],color='black')
        for i,j in vertical_chords:
            ax.plot([y[i],y[j]],[x[i],x[j]],color='black')
        plt.show()
        plt.clf()
    # MAXIMUM PARTITION CODE ENDS ---------------------------------

    # MINIMUM PARTITION CODE STARTS -------------------------------
   
    horizontal_chords = temp_hori[:]
    vertical_chords = temp_verti[:]

    # Creating a bipartite graph from the set of chords
    G = nx.Graph()
    for i,h in enumerate(horizontal_chords):
        y1 = y[h[0]]
        x1 = min(x[h[0]] ,x[h[1]] )
        x2 = max(x[h[0]] ,x[h[1]])
        G.add_node(i, bipartite=1)
        for j,v in enumerate(vertical_chords):
            x3 = x[v[0]]
            y3 = min(y[v[0]],y[v[1]])
            y4 = max(y[v[0]],y[v[1]])
            G.add_node(j + len(horizontal_chords),bipartite=0)
            if x1 <= x3 and x3 <=x2 and y3 <= y1 and y1 <= y4:    
                G.add_edge(i, j + len(horizontal_chords))
    
    if len(horizontal_chords) == 0:
        for j,v in enumerate(vertical_chords):
            x3 = x[v[0]]
            y3 = min(y[v[0]],y[v[1]])
            y4 = max(y[v[0]],y[v[1]])
            G.add_node(j,bipartite=0)
    
    # finding the maximum matching of the bipartite graph, G.
    top_nodes = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0]
    maximum_matching = nx.bipartite.maximum_matching(G, top_nodes = top_nodes)
    
    maximum_matching_list = []
    for i,j in maximum_matching.items():
        maximum_matching_list += [(i,j)]

    
    M = nx.Graph()
    M.add_edges_from(maximum_matching_list)
    maximum_matching = M.edges()
    
    # breaking up into two sets
    V = {n for n, d in G.nodes(data=True) if d['bipartite']==0}
    H = set(G) - V

    free_vertices = []
    for u in H:
        temp = []
        for v in V:
            if (u,v) in maximum_matching or (v,u) in maximum_matching:
                temp += [v]
        if len(temp) == 0:
            free_vertices += [u]
    for u in V:
        temp = []
        for v in H:
            if (u,v) in maximum_matching or (v,u) in maximum_matching:
                temp += [v]
        if len(temp) == 0:
            free_vertices += [u]
            
    # finding the maximum independent set
    max_independent = []
    while len(free_vertices) != 0 or len(maximum_matching) != 0:
        if len(free_vertices) != 0 :
            u = free_vertices.pop()
            max_independent += [u]
        else:
            u, v = list(maximum_matching).pop()
            M.remove_edge(u,v)
            G.remove_edge(u,v)
            max_independent += [u]

        for v in list(G.neighbors(u)):
            G.remove_edge(u, v)
            for h in G.nodes():
                if (v,h) in maximum_matching:
                    M.remove_edge(v,h)
                    free_vertices += [h]
                if (h,v) in maximum_matching:
                    M.remove_edge(h,v)
                    free_vertices += [h]

    
    # drawing the partitioned polygon 
    independent_chords = []
    for i in max_independent:
        if (i >= len(horizontal_chords)):
            independent_chords += [vertical_chords[i-len(horizontal_chords)]]
        else:
            independent_chords += [horizontal_chords[i]]
    unmatched_concave_vertices = [i for i in concave_vertices]
    for i,j in independent_chords:
        if i in unmatched_concave_vertices:
            unmatched_concave_vertices.remove(i)
        if j in unmatched_concave_vertices:
            unmatched_concave_vertices.remove(j)
    
    nearest_chord = []
    for i in unmatched_concave_vertices:
        dist = 0
        nearest_distance = math.inf
        for j in max_independent:
            if j < len(horizontal_chords):
                temp1, temp2 = horizontal_chords[j]
                if abs(y[i] - y[temp1]) < nearest_distance and \
                (x[i] <= x[temp1] and x[i] >= x[temp2] or x[i] >= x[temp1] and x[i] <= x[temp2]) \
                and abs(temp1 - i) != 1 and abs(temp2 - i) != 1:
                    middles = []
                    for u in range(len(x)):
                        if x[i] == x[u] and (y[i] < y[u] and y[u] < y[temp1] or y[temp1] < y[u] and y[u] < y[i]):
                            middles.append(u)
                    if len(middles) == 0:
                        nearest_distance = abs(y[i] - y[temp1])
                        dist = y[temp1] - y[i]

        if nearest_distance != math.inf:
            nearest_chord.append((i,dist)) 
        else:
            for k in collinear_vertices:
                if x[k] == x[i] and abs(y[k] - y[i]) < nearest_distance and abs(k-i) != 1:
                    middles = []
                    for u in range(len(x)):
                        if x[i] == x[u] and (y[i] < y[u] and y[u] < y[k] or y[k] < y[u] and y[u] < y[i]):
                            middles.append(u)
                    if len(middles) == 0:
                        nearest_distance = abs(y[i] - y[k])
                        dist = y[k] - y[i]
            nearest_chord.append((i,dist)) 
     
    if viz:
        print("The minimum partitioned rectillinear polygon")
        fig, ax = plt.subplots()
        ax.plot(y, x, color='black')
        ax.scatter(y, x, color='black')
        for i in range(len(x)):
            ax.annotate(i, (y[i],x[i]))
        plt.gca().invert_yaxis()
        

    for k,(i,j) in enumerate(independent_chords):
        if viz:
            ax.plot([y[i],y[j]],[x[i],x[j]], color='black')
        independent_chords[k] = [[x[i],y[i]], [x[j],y[j]]]
    for k, (i,dist) in enumerate(nearest_chord):
        if viz:
            ax.plot([y[i], y[i]+dist], [x[i],x[i]], color='black')
        nearest_chord[k] = [[x[i],y[i]], [x[i],y[i]+dist]]
    
    if viz:
        plt.show()

    # MAXIMUM PARTITION CODE ENDS
    
    lines = independent_chords + nearest_chord
    return lines
    
class Input:
    def __init__(self):
        self.prevX = None
        self.prevY = None
        self.cell_size = None
        self.room_list = []
        self.current_room = []
        self.current_points = []
        self.current_lines = []
        self.current_orthogonal_partitioned_room = []
        self.matrix = None
        self.label_cnt = None
        self.orthogonal_rooms = {}

    def post_processing(self):
        m, n = self.matrix.shape
        
        def get_indexes(matrix):
            m, n = matrix.shape
            for i in range(m):
                if matrix[i, :].any() != 0:
                    break

            for j in range(n):
                if matrix[:, j].any() != 0:
                    break
            return i, j 
        
        c, d = get_indexes(self.matrix)
        updated = self.matrix[c:m, d:n]
        self.matrix = updated
    
    def get_matrix(self):
        return self.matrix.astype(int).tolist()
    
    def get_orthogonal_rooms(self):
        return list(self.orthogonal_rooms.values())

    def create_grid(self, event = None):
        w = self.c.winfo_width() # Get current width of canvas
        h = self.c.winfo_height() # Get current height of canvas
        self.c.delete('grid_line') # Will only remove the grid_line

        # Creates all vertical lines at intevals of 100
        for i in range(int(self.cell_size/2), w, self.cell_size):
            self.c.create_line([(i, 0), (i, h)], tag='grid_line', fill="grey")

        # Creates all horizontal lines at intevals of 100
        for i in range(int(self.cell_size/2), h, self.cell_size):
            self.c.create_line([(0, i), (w, i)], tag='grid_line', fill="grey")

    def get_drawable_vertex(self, x):
        x_ = x * self.cell_size
        x_ += (self.cell_size/2)
        return x_

    def label_callback(self, event):
        x = int(event.x / self.cell_size)
        y = int(event.y / self.cell_size)

        if self.prevX is not None:
            if self.prevX != x and self.prevY != y:
                return

            line = self.c.create_line(self.get_drawable_vertex(self.prevX),
                                  self.get_drawable_vertex(self.prevY), 
                                  self.get_drawable_vertex(x),
                                  self.get_drawable_vertex(y),
                                  width=2, 
                                  fill = "gray")
            self.current_lines.append(line)

        point = self.c.create_oval(self.get_drawable_vertex(x) - 3,
                                   self.get_drawable_vertex(y) - 3,
                                   self.get_drawable_vertex(x) + 3, 
                                   self.get_drawable_vertex(y) + 3,
                                   fill="gray")

        self.current_points.append(point)

        self.prevX = x
        self.prevY = y

        if([y,x] in self.current_room):

            self.current_room.append([y,x])
            self.room_list.append(self.current_room)

            for point in self.current_points:
                self.c.itemconfig(point, fill="black")
            for line in self.current_lines:
                self.c.itemconfig(line, fill="black")

            self.current_lines = []
            self.current_room = []
            self.current_points = []
            
            self.prevX = None
            self.prevy = None
            tk.messagebox.showinfo("Message", "Room No. " + str(len(self.room_list)) + " added!")    
        else:
            self.current_room.append([y,x])

    def quit(self):
        self.post_processing()
        self.root.quit()
    
    def button_click2(self):
        self.c.unbind('<Button-1>')
#         self.control.destroy()
        self.fill_rectangular_rooms(self.room_list)
        polygons_gui, added_lines_gui = self.split_orthogonal_rooms(self.room_list)

        print('Partitioning of Orthogonal Rooms (if any) Done...')

        if(len(polygons_gui) != 0):
            self.title_var.set("Label Partitoned Rooms")
            self.c.bind('<Button-1>', self.label_partitions_callback)
        self.control.configure(text = "Finish", command=self.quit)
    

    def get_shape(self, inputlist):
            c = max([max([i for i, j in current_room]) for current_room in inputlist])
            d = max([max([j for i, j in current_room]) for current_room in inputlist])
            return c, d
        
    def fill_rectangular_rooms(self, room_list):
        shape = self.get_shape(room_list)
        self.matrix = np.zeros(shape)
        self.label_cnt = 1
        for i, room in enumerate(room_list):
            if len(room) == 5:
                x0, y0 = room[0]
                x1, y1 = room[2]
                for i in range(x0, x1):
                    for j in range(y0, y1):
                        self.matrix[i, j] = self.label_cnt
                # label in gui
                x0, y0 = self.get_label_position(room[0:-1])
    #             print(x0, y0)
                self.c.create_text(self.get_drawable_vertex(x0),
                              self.get_drawable_vertex(y0), 
                              text = str(self.label_cnt),
                              font = "16")
                self.label_cnt += 1

    def get_index(self, points):
        # pprint(points)
        pointA = points[0]
        pointB = points[1]

        point = Point((pointA[0] + pointB[0])/2, (pointA[1] + pointB[1])/2)
        # print(point)
        for i, room in enumerate(self.room_list):
            if len(room) > 5:
                polygon_room = Polygon(room)
                # pprint(processed_room)
                if polygon_room.contains(point):
                    return i
        return None

    def label_partitions_callback(self, event):
        x = int(event.x / self.cell_size)
        y = int(event.y / self.cell_size)

        self.current_orthogonal_partitioned_room.append([y, x])

        if len(self.current_orthogonal_partitioned_room) == 2:

            # fill matrix
            self.fill_partitioned_rooms(self.current_orthogonal_partitioned_room)

            # label in gui
            x0, y0 = self.get_label_position(self.current_orthogonal_partitioned_room)
    #         print(x0, y0)
            self.c.create_text(self.get_drawable_vertex(x0),
                          self.get_drawable_vertex(y0), 
                          text = str(self.label_cnt),
                         font = "16")

            # update self.orthogonal rooms
            index = self.get_index(self.current_orthogonal_partitioned_room)
            if index is not None and index in self.orthogonal_rooms:
                self.orthogonal_rooms[index].append(self.label_cnt)
            elif index is not None:
                self.orthogonal_rooms[index] = [self.label_cnt]
            else:
                print("Error! Couldn't find points in any orthogonal room")

            self.label_cnt += 1
            self.current_orthogonal_partitioned_room = []            

    def get_label_position(self, room):
        x0 = 0
        y0 = 0
        for point in room:
    #         print(point[1], point[0])
            x0 += point[1]
            y0 += point[0]
        x0 /= len(room)
        y0 /= len(room)
        return x0, y0

    def fill_partitioned_rooms(self, room_points):
        x0, y0 = room_points[0]
        x1, y1 = room_points[1]
        for i in range(x0, x1):
            for j in range(y0, y1):
                self.matrix[i, j] = self.label_cnt

    def highlight_polygon(self, room):
        lines = []
        for i in range(len(room)-1):
            y0, x0 = room[i]
            y1, x1 = room[i+1]
            line = self.c.create_line(self.get_drawable_vertex(x0),
                                  self.get_drawable_vertex(y0), 
                                  self.get_drawable_vertex(x1),
                                  self.get_drawable_vertex(y1),
                                  width=5, 
                                  fill = "black")
            lines.append(line)
        return lines

    def split_orthogonal_rooms(self, room_list):
        orthogonal = []
        for i, room in enumerate(room_list):
            if len(room) > 5:
                orthogonal.append(i)

        currLabel = len(room_list)+1 
        polygons_gui = []
        added_lines_gui = []
        
        for i in orthogonal:
            added_lines = compute_partition(room_list[i])
            polygon = self.highlight_polygon(room_list[i])
            polygons_gui.append(polygon)
            
            for line in added_lines:
                y0, x0 = line[0][0], line[0][1]
                y1, x1 = line[1][0], line[1][1]
                line = self.c.create_line(self.get_drawable_vertex(x0),
                                  self.get_drawable_vertex(y0), 
                                  self.get_drawable_vertex(x1),
                                  self.get_drawable_vertex(y1),
                                  width=2, 
                                  fill = "red")
                added_lines_gui.append(line)
        return polygons_gui, added_lines_gui
    
    def switch(self):
        self.root.destroy()
        main.run()

    def exit(self):
        self.root.destroy()

    def draw(self, cell_size = 50):
        self.cell_size = cell_size  
        self.root = tk.Tk()
        frame = tk.Frame(self.root, height=1000, width=500)
        frame.pack(fill="both", expand=True)

        self.c = tk.Canvas(frame, height=500, width=500, bg='white')

        self.c.pack(side = tk.BOTTOM, padx=(5,5), pady=(5,5), fill="both", expand=True)
        self.c.bind('<Configure>', self.create_grid)    
        
        self.c.bind('<Button-1>', self.label_callback)


        self.control = tk.Button(self.root, text="OK", command = self.button_click2)
        self.control.pack(side = tk.TOP, padx = (0,5), pady = (5,5))
    
        self.gplan = tk.Button(self.root, text="Switch to GPLAN", command = self.switch)
        self.gplan.pack(side = tk.RIGHT, padx = (0,5), pady = (5,5))

        self.title_var = tk.StringVar(frame,value="Draw Floor Plan")
        self.label = tk.Label(frame, textvariable=self.title_var, relief=tk.RAISED)
        self.label.pack(padx = (5,5), pady = (5,0))
        self.root.protocol("WM_DELETE_WINDOW", self.exit)
        self.root.mainloop()

# -------------------- input.py -------------------- #


# -------------------- preprocess_irregular.py -------------------- #

def preprocess_irregular(A, viz=False):
	# A = [[0, 1, 0, 0], 
	# 	 [0, 1, 0, 0], 
	# 	 [0, 2, 2, 2], 
	# 	 [0, 2, 2, 2]]

	rows = len(A)
	columns = len(A[0])

	A = np.array(A)
	user_rooms = np.amax(A)

	# For labelling virtual rectangular rooms to make it RFP
	count = user_rooms+1

	row=0
	while(row<rows):
		column = 0
		while(column<columns):
			if(A[row][column]==0):
				while(column<columns and A[row][column]==0):
					A[row][column] = count
					column += 1
				column -= 1
				count += 1
			column += 1
		row += 1

	total_rooms = np.amax(A)

	if viz:
		print(A)
		print('User rooms = {}'.format(user_rooms))
		print('Total rooms after virtual addition to make RFP = {}'.format(total_rooms))

	return(A, user_rooms, total_rooms)

# -------------------- preprocess_irregular.py -------------------- #


# -------------------- floor_plan.py -------------------- #

class FloorPlan:
    def __init__(self, encoded_matrix, user_rooms, total_rooms):
        self.room_width = None
        self.room_height = None
        self.encoded_matrix = encoded_matrix
        self.room_x = None
        self.room_y = None
        self.hor_dgph = None

        self.user_rooms = user_rooms
        self.total_rooms = total_rooms
    
    def compute_dimensions(self):
        [width, height, self.hor_dgph] = floorplan_to_st(self.encoded_matrix, self.user_rooms, self.total_rooms)
        width = np.transpose(width)
        height = np.transpose(height)
        self.room_width = width.flatten()
        self.room_height = height.flatten()
        self.room_x = np.zeros(len(self.room_width),float)
        self.room_y = np.zeros(len(self.room_width),float)

    def compute_coordinates(self):
    
        if(self.hor_dgph is None):
            print("Error: Run compute_dimensions() first!")
            return 

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

        self.hor_dgph = np.array(self.hor_dgph)
        self.hor_dgph = self.hor_dgph.transpose()
        xmin = float(0)
        ymin = float(0)
        
        B = self.encoded_matrix.copy()
        
        for i in range(0,len(B)):
            for j in range(0,len(B[0])):
                B[i][j] -= 1
        
        m = len(B[0])
        n = len(B)
        N = np.amax(B)+1
        rect_drawn = []
        
        j = 0
        C = [[-1 for i in range(0,len(B[0]))] for i in range(0,len(B))]
    
        while j < len(B[0]):
            rows = []
            for i in range(0,len(B)):
                if B[i][j] not in rows:
                    rows.append(B[i][j])
            k = 0
            for k in range(0,len(rows)):
                C[k][j] = rows[k]
            j += 1
        
        xR = np.zeros((N),float)
        for i in range(0,m):
            xmax = np.zeros((N),float)
            ymin = 0
            for j in range(0,n):
                if C[j][i] == -1:
                    break
                else:
                    if any(ismember(rect_drawn, C[j][i])):
                        ymin = ymin + self.room_height[C[j][i]]
                        xmax=np.zeros((N),float)
                        xmax[0] = xR[C[j][i]]
                        continue
                    else:
                        if not any(find_sp(self.hor_dgph[C[j][i]])):
                            ymin = ymin
                        else:
                            l = find(self.hor_dgph[C[j][i]])
                            xmin = xR[l]
                    self.room_x[C[j][i]], self.room_y[C[j][i]] = xmin,ymin 
                    rect_drawn.append(C[j][i])
                    xmax[C[j][i]] = xmin + self.room_width[C[j][i]]
                    xR[C[j][i]] = xmax[C[j][i]]
                    ymin = ymin + self.room_height[C[j][i]]
                    
            xmax = xmax[xmax!=0]
            xmin = min(xmax)

    def collinear(self, l1, l2):
        if(l1[0][0] == l1[1][0] and l2[0][0] == l2[1][0] and l1[0][0] == l2[0][0]):
            return True
        if(l1[0][1] == l1[1][1] and l2[0][1] == l2[1][1] and l1[0][1] == l2[0][1]):
            return True
        return False     

    def intersect(self, lines, line):
        removed_lines = []
        p = line[0]
        q = line[1]

        for [a, b] in lines:
            if(self.collinear([a, b], line)):
                if a[0] == b[0]: #vertical
                    m = [a[1], b[1]]
                    n = [p[1], q[1]]
                    m.sort()
                    n.sort()
                    if m[1] > n[0]: # if intersect
                        removed_lines.append([[a[0], max(n[0], m[0])], [a[0], min(m[1], n[1])]])
                elif a[1] == b[1]: #horizontal
                    m = [a[0], b[0]]
                    n = [p[0], q[0]]
                    m.sort()
                    n.sort()
                    if m[1] > n[0]: # if intersect
                        removed_lines.append([[max(n[0], m[0]), a[1]], [min(m[1], n[1]), a[1]]])
        return removed_lines

    def get_removed_lines(self, orthogonal_rooms = None):
        removed_lines = []
        for rooms in orthogonal_rooms:
            lines = []
            for i in rooms:
                i -= 1
                a = [round(self.room_x[i]), round(self.room_y[i])]
                b = [round(self.room_x[i] + self.room_width[i]), round(self.room_y[i])]
                c = [round(self.room_x[i] + self.room_width[i]), 
                    round(self.room_y[i] + self.room_height[i])]
                d = [round(self.room_x[i]), round(self.room_y[i] + self.room_height[i])]
                
                room_walls = [[a, b],
                                [b, c],
                                [c, d],
                                [d, a]]

                # pprint(room_walls)
                # print(lines)
                # print(room_walls)


                for wall in room_walls:
                    # print(lines)
                    # print(wall)
                    removed_lines += self.intersect(lines, wall)
                    # for removed_line in removed_lines:
                    #     print(removed_line)
                    #     p = removed_line[0]
                    #     q = removed_line[1]
                        # plt.plot([p[0], q[0]], [p[1], q[1]], color='red')   

                lines.append([a, b])
                lines.append([b, c])
                lines.append([c, d])
                lines.append([d, a])

        return removed_lines

    def draw_rfp(self, orthogonal_rooms, ax, draw_partitions=False):
        
        if(self.room_x is None):
            print("Error: Run compute_coordinates() first!")
            return

        ax.invert_yaxis()
        if draw_partitions:
        	ax.title.set_text('Dimensioned RFP with Labeled Partitions')
        else:
        	ax.title.set_text('Dimensioned RFP')

        removed_lines = self.get_removed_lines(orthogonal_rooms)        
        
        for i in range(self.user_rooms):
            if self.room_width[i] == 0:
                continue

            a = [round(self.room_x[i]), round(self.room_y[i])]
            b = [round(self.room_x[i] + self.room_width[i]), round(self.room_y[i])]
            c = [round(self.room_x[i] + self.room_width[i]), 
                round(self.room_y[i] + self.room_height[i])]
            d = [round(self.room_x[i]), round(self.room_y[i] + self.room_height[i])]
            
            room_walls = [[a, b],
                            [b, c],
                            [c, d],
                            [d, a]]

            for wall in room_walls:
                intersection = self.intersect(removed_lines, wall)
                if(len(intersection) == 0):
                    m = wall[0]
                    n = wall[1]
                    ax.plot([m[0], n[0]], [m[1], n[1]], color='black',linewidth=4)
                else:
                    line = intersection[0]
                    line.sort()
                    wall.sort()
                    x1 = wall[0][0]
                    y1 = wall[0][1]

                    x2 = line[0][0]
                    y2 = line[0][1]
                    
                    x3 = line[1][0]
                    y3 = line[1][1]

                    x4 = wall[1][0]
                    y4 = wall[1][1]
                    
                    ax.plot([x1, x2], [y1, y2], color='black',linewidth=4)
                    ax.plot([x3, x4], [y3, y4], color='black',linewidth=4)
                    
                    if draw_partitions: 
                        ax.plot([x2, x3], [y2, y3], color='red')

            x = (a[0] + c[0]) / 2 - 1
            y = (a[1] + c[1]) / 2
            if draw_partitions:
            	ax.text(x, y, str(i+1), fontweight='bold', fontsize=16.5)
            	ax.text(x-4, y+3, " ( " + str(round(self.room_height[i],2)) + ' x ' + str(round(self.room_width[i],2)) + ' )',fontsize=10)

# -------------------- floor_plan.py -------------------- #


# -------------------- floorplan_to_st.py -------------------- #

def floorplan_to_st(A, user_rooms, total_rooms):
	# A=[
	# [0,0,0,0,6,6,0,0],
	# [0,2,4,4,6,6,0,0],
	# [1,2,3,5,6,6,7,11],
	# [0,2,3,8,8,8,9,11],
	# [0,0,3,0,10,0,9,0],
	# [0,0,0,0,10,0,0,0],
	# ]

	m=len(A)
	n=len(A[0])
	
	A = np.array(A)

	len_dgph=np.amax(A)
	
	ver_dgph=np.zeros((len_dgph,len_dgph),int)
	north_adj=np.zeros((1,len_dgph),int)
	south_adj=np.zeros((1,len_dgph+1),int)

	for i in range(0,n):
		for j in range(0,m):
			if((j == 0 and A[j][i] != 0) or (j > 0 and A[j-1][i] == 0 and A[j][i] != 0)):
				north_adj[0][A[j][i]-1] = 1

	for i in range(n-1,0,-1):
		for j in range(m-1,0,-1):
			if((j == m-1 and A[j][i] != 0) or (j < m-1 and A[j+1][i] == 0 and A[j][i] != 0)):
				south_adj[0][A[j][i]] = 1

	for i in range(0,n):
		temp = 0
		for j in range(0,m):
			if temp == 0:
				if(A[j][i] != 0):
					temp = A[j][i]
			elif A[j][i] != temp and A[j][i] != 0:
				ver_dgph[temp-1][A[j][i]-1] = 1
				temp = A[j][i]
			elif A[j][i] == 0:
				temp = 0
	
	VER=[]
	for i in north_adj:
		VER.append(i)

	for i in ver_dgph:
		VER = np.append(VER,[i],axis=0)

	VER = np.insert(VER, len_dgph, south_adj[0], axis=1)
	VER = np.insert(VER,0,[0],axis=1)
	VER = np.insert(VER,len_dgph+1,[0],axis=0)
	
	hor_dgph=np.zeros([len_dgph,len_dgph])
	west_adj=np.zeros([1,len_dgph])
	east_adj=np.zeros([1,len_dgph+1])

	for i in range(0, m):
		for j in range(0, n):
			if((j == 0 and A[i][j] != 0) or (j > 0 and A[i][j-1] == 0 and A[i][j] != 0)):
				west_adj[0][A[i][j]-1] = 1

	for i in range(m-1,0,-1):
		for j in range(n-1,0,-1):
			if((j == n-1 and A[i][j] != 0) or (j < n-1 and A[i][j+1] == 0 and A[i][j] != 0)):
				east_adj[0][A[i][j]] = 1

	for i in range(0,m):
		temp = 0
		for j in range(0,n):
			if temp == 0:
				if A[i][j] != 0:
					temp = A[i][j]
			elif A[i][j] != temp and A[i][j] != 0:
				hor_dgph[temp-1][A[i][j]-1]=1
				temp = A[i][j]
			elif A[i][j] == 0:
				temp = 0
	HOR=[]

	for i in west_adj:
		HOR.append(i)

	for i in hor_dgph:
		HOR = np.append(HOR,[i],axis=0)

	HOR = np.insert(HOR, len_dgph, east_adj[0], axis=1)
	HOR = np.insert(HOR,0,[0],axis=1)
	HOR = np.insert(HOR,len_dgph+1,[0],axis=0)

	# print("NORTH")
	# print(north_adj)
	# print(ver_dgph)
	# print(south_adj)
	# print(VER)

	# print("WEST")
	# print(west_adj)
	# print(hor_dgph)
	# print(HOR)

	[width,height] = digraph_to_eq(VER,HOR, user_rooms, total_rooms)

	# print ((-1)*width, (-1)*height)
	
	return [(-1)*width,(-1)*height,hor_dgph]

# -------------------- floorplan_to_st.py -------------------- #

def gui_fnc(nodes):
	width = []
	ar = []
	root = tk.Toplevel() 

	root.title('Dimensional Input')
	root.geometry(str(1000) + 'x' + str(400))
	Upper_right = tk.Label(root, text ="Enter minimum width and aspect ratio for each room",font = ("Times New Roman",12)) 
	  
	Upper_right.place(relx = 0.70,  
					  rely = 0.1, 
					  anchor ='ne')

	text_head_width = []
	text_head_area = []
	text_room = []
	value_width = []
	value_area =[]
	w = []
	minA = []
	
	
	for i in range(0,nodes):
		i_value_x = int(i/10)
		i_value_y = i%10
		w.append(tk.IntVar(None)) 
		minA.append(tk.IntVar(None)) 
		if(i_value_y == 0):
			text_head_width.append("text_head_width_"+str(i_value_x+1))
			text_head_width[i_value_x] = tk.Label(root,text = "Width")
			text_head_width[i_value_x].place(relx = 0.30 + 0.20*i_value_x,  
					  rely = 0.2, 
					  anchor ='ne')
			text_head_area.append("text_head_area_"+str(i_value_x+1))
			text_head_area[i_value_x] = tk.Label(root,text = "Area")
			text_head_area[i_value_x].place(relx = 0.35 + 0.20*i_value_x,  
					  rely = 0.2, 
					  anchor ='ne')
		text_room.append("text_room_"+str(i))
		text_room[i] = tk.Label(root, text ="Room"+str(i),font = ("Times New Roman",8)) 

		text_room[i].place(relx = 0.25 + 0.20*i_value_x,  
					  rely = 0.3 + (0.05 * i_value_y), 
					  anchor ='ne')
		value_width.append("value_width" + str(i))
		value_width[i] = tk.Entry(root, width = 5,textvariable=w[i])
		value_width[i].place(relx = 0.30 +0.20*i_value_x,  
					  rely = 0.3 +(0.05)*i_value_y, 
					  anchor ='ne')
		value_area.append("value_area"+str(i))
		value_area[i] = tk.Entry(root, width = 5,textvariable=minA[i])
		value_area[i].place(relx = 0.35+ 0.20*i_value_x,  
					  rely = 0.3 + (0.05)*i_value_y, 
					   anchor ='ne')
	def button_clicked():
		for i in range(0,nodes):
			width.append(int(value_width[i].get()))
			if(checkvar1.get() == 0):
				ar.append(float(value_area[i].get()))
		# if(len(width) != nodes or len(ar) != nodes or len()!= nodes):
		# 	messagebox.showerror("Invalid DImensions", "Some entry is empty")
		# print(width)
		# print(area)
		# else:
		root.destroy()

	def clicked():
		if(checkvar1.get() == 0):
			for i in range(0,nodes):
				value_area[i].config(state="normal")
				ar = []
		else:
			for i in range(0,nodes):
				value_area[i].config(state="disabled")
				ar = []

	button = tk.Button(root, text='Submit', padx=5, command=button_clicked)      
	button.place(relx = 0.5,  
					  rely = 0.9, 
					  anchor ='ne')
	checkvar1 = tk.IntVar()
	c1 = tk.Checkbutton(root, text = "Default AR Range", variable = checkvar1,onvalue = 1, offvalue = 0,command=clicked)
	c1.place(relx = 0.85, rely = 0.9, anchor = 'ne')

	root.wait_window(root)
	print(width,ar,checkvar1.get())
	return [width,ar]
	
# -------------------- digraph_to_eq.py -------------------- #
# from layout_dim import  gui_fnc
def digraph_to_eq(VER, HOR, user_rooms,total_rooms):
	N=len(VER)
	[f_VER,A_VER,Aeq_VER,Beq_VER]=Convert_adj_equ(VER,5)
	[f_HOR,A_HOR,Aeq_HOR,Beq_HOR]=Convert_adj_equ(HOR,5)

	[inp_min,inp_area] = gui_fnc(user_rooms)

	# inp_area=[int(x) for x in input("Enter the minimum area of each room: ").strip().split()]

    # v = gui_fnc(user_rooms)
    
	for i in range(total_rooms-user_rooms):
		inp_min.append(0)
		inp_area.append(0)
	

	b_VER = np.dot(np.array(inp_min),-1)
	b_VER = np.transpose(b_VER)
	b_VER = b_VER.astype(float)
	dimensions = solve_linear(N,f_VER, A_VER, b_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, inp_area)

	return [dimensions[0],dimensions[1]]

# -------------------- digraph_to_eq.py -------------------- #


# -------------------- Convert_adj_equ.py -------------------- #

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
	for i in range(0,N-1):
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

# -------------------- Convert_adj_equ.py -------------------- #


# -------------------- solve_linear.py -------------------- #

def solve_linear(N,f_VER, A_VER, b_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, inp_area):

	# print(Aeq_VER)
	# print("A_ub", A_VER)
	# print("b_ub", b_VER)
	value_opti_ver = scipy.optimize.linprog(f_VER,A_ub=A_VER,b_ub=b_VER,A_eq=Aeq_VER,b_eq=Beq_VER, bounds=(1,None), method='interior-point', callback=None, options=None, x0=None)


	X1=value_opti_ver['x']
	X1 = np.array([X1])
	X1 = np.transpose(X1)
	# print(X1)
	
	W=np.dot(A_VER,X1)
	# print(W)

	inp_height = np.array(inp_area) / np.transpose(np.dot(W,-1))
	b_HOR=np.zeros([N-1,1],dtype=float)
	b_HOR = np.dot(np.array(inp_height),-1)
	
	# print(inp_height)
	# print(b_HOR)
	
	value_opti_hor = scipy.optimize.linprog(f_HOR,A_ub=A_HOR,b_ub=b_HOR,A_eq=Aeq_HOR,b_eq=Beq_HOR, bounds=(1,None), method='interior-point', callback=None, options=None, x0=None)

	X2=value_opti_hor['x']
	X2 = np.array([X2])
	X2 = np.transpose(X2)
	H=np.dot(A_HOR,X2)
	
	return [W, H]

# -------------------- solve_linear.py -------------------- #
def run():
	i = Input()

	print('----------User Instructions----------')
	print('1. All rooms are to be drawn in a clockwise manner.')
	print('2. Kindly ensure that the start point and end point are same for each room while drawing the rooms.')
	print('3. After partitioning, only the partitions of orthogonal rooms need to be labelled as per user\'s convenience and rectangular rooms are implicitly labelled.')
	print('4. Minimum width and area for rooms are to be entered in the order of room labels.\n')

	i.draw(cell_size = 30)

	encoded_matrix, user_rooms, total_rooms = preprocess_irregular(i.get_matrix())

	print('User rooms = {}'.format(user_rooms))

	G = FloorPlan(encoded_matrix, user_rooms, total_rooms)

	G.compute_dimensions()
	print('Computing Dimensions...')

	G.compute_coordinates()
	print('Getting plot coordinates...')
	
	# plt.figure(figsize=(50 , 50))

	f, ax = plt.subplots(figsize=(50, 50))

	print('Plotting Dimensioned RFPs...')
	ax.set_aspect('equal', 'box')
	G.draw_rfp(i.get_orthogonal_rooms(), ax)

	f, ax = plt.subplots(figsize=(50, 50))
	ax.set_aspect('equal', 'box')
	G.draw_rfp(i.get_orthogonal_rooms(), ax,  draw_partitions = True)

	plt.show()
	# plt.clf(

if __name__=='__main__':
	run()


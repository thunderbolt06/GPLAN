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
import checker

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
		self.inp_min = []
		self.inp_area = []
		self.ar_pmt = 0
		self.ar_min = []
		self.ar_max = []
		if(self.dimensioned == 1):
			self.rfp_test = checker.rfp_checker(self.matrix)
			print(self.rfp_test)
			if(self.rfp_test):
				self.value1 = dimgui.gui_fnc(self.node_count)
				# print("tag1",self.value1)
				self.inp_min = self.value1[0]
				self.inp_height = self.value1[2]
				# if self.value1[3] == 0:
				# 	self.ar_pmt = 1
				# 	self.ar_min = self.value1[1]
				# 	self.ar_max = self.value1[2]

		self.graph = nx.Graph()
		self.graph.add_edges_from(value[2])

		self.original_edge_count = 0
		self.original_node_count = 0
		self.original_edge_count1 = 0
		self.original_node_count1 = 0
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
		self.edge_matrix1 = None
		
		self.cip_list = []
		self.cip = []
		self.original_cip =[]
		self.node_color_list =[]

		self.node_position = None
		self.degrees = None
		self.good_vertices = None
		self.contractions = []
		self.rdg_vertices = []
		self.to_be_merged_vertices = []
		self.k4 = []
		self.rdg_vertices2 =[]

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
		self.area = []
		# print("Enter each edge in new line")
		# for i in range(self.edge_count):
		#     line = input()
		#     node1 = int(line.split()[0])
		#     node2 = int(line.split()[1])
		#     self.matrix[node1][node2] = 1
		#     self.matrix[node2][node1] = 1
		# self.inp_min=[int(x) for x in input("Enter the minimum width of room: ").strip().split()]
		# self.multiple_rfp = int(input("Multiple RFP?"))
		# self.dimensioned = int(input("Dimensioned?"))
		self.directed = opr.get_directed(self)
		self.triangles = opr.get_all_triangles(self)
		self.outer_vertices = opr.get_outer_boundary_vertices(self)[0]
		self.outer_boundary = opr.get_outer_boundary_vertices(self)[1]
		self.shortcuts = None
		self.shortcut_list = []
		self.origin = 50
		self.boundaries = []
		
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
		self.original_edge_count = self.edge_count
		self.original_node_count = self.node_count
		self.triangles = opr.get_all_triangles(self)
		K4.find_K4(self)
		for i in self.k4:
			K4.resolve_K4(self,i,i.edge_to_be_removed,self.rdg_vertices,self.rdg_vertices2,self.to_be_merged_vertices)
		self.directed = opr.get_directed(self)
		self.triangles = opr.get_all_triangles(self)
		self.outer_vertices = opr.get_outer_boundary_vertices(self)[0]
		self.outer_boundary = opr.get_outer_boundary_vertices(self)[1]
		self.shortcuts = sr.get_shortcut(self)
		self.cip = news.find_cip_single(self)
		# self.cip =[ [0,1,2,3,4,5,6],[6,7,8,9,10,11,12],[12,13,14,15],[15,16,17,18,19,0]]
		news.add_news_vertices(self)
		print("North Boundary: ", self.cip[0])
		print("East Boundary: ", self.cip[1])
		print("South Boundary: ", self.cip[2])
		print("West Boundary: ",self.cip[3])
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
			draw.draw_rdg(self,1,pen,self.to_be_merged_vertices,mode)
		if(mode == 2):
			draw.draw_rdg(self,1,pen,self.to_be_merged_vertices,mode)

	def create_single_floorplan(self,pen,textbox,mode):
		global box
		box = textbox
		if(mode == 0):
			self.create_single_dual(0,pen,textbox)
		self.encoded_matrix = opr.get_encoded_matrix(self)
		B = copy.deepcopy(self.encoded_matrix)
		A = copy.deepcopy(self.encoded_matrix)
		# minimum_width = min(self.inp_min)
		# for i in range(0,len(self.biconnected_vertices)):
		# 	self.inp_min.append(0)
		# 	self.ar_min.append(0.5)
		# 	self.ar_max.append(2)
		# for i in range(0,len(self.to_be_merged_vertices)):
		# 	self.inp_min.append(minimum_width)
		# 	self.ar_min.append(0.5)
		# 	self.ar_max.append(2)
		print(self.to_be_merged_vertices,self.biconnected_vertices)
		print(A,self.inp_min,self.ar_pmt,self.ar_min,self.ar_max)
		[width,height,hor_dgph] = floorplan_to_st(A,self.inp_min,self.inp_height)
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
		draw.draw_rdg(self,1,pen,self.to_be_merged_vertices,mode)


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
		# self.cip = news.find_cip_single(self)
			start = time.time()
			# print(self.node_count)
			# print(self.edge_count)
			# print(self.matrix)
			self.boundaries = opr.find_possible_boundary(opr.ordered_outer_boundary(self))
			self.cip_list = news.populate_cip_list(self)
			# self.cip = self.cip_list[0]
			# self.original_cip = self.cip.copy()
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
		# print(self.cip)
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

					self.cip = news.find_cip_single(self)
					for i in range(0,len(to_be_merged_vertices)):
						node_color1.append(self.node_color[rdg_vertices[i]])
					for i in range(0,len(to_be_merged_vertices)):
						node_color2.append(self.node_color[rdg_vertices2[i]])						
					print("North Boundary: ", self.cip[0])
					print("East Boundary: ", self.cip[1])
					print("South Boundary: ", self.cip[2])
					print("West Boundary: ",self.cip[3])
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
					
					print("\n")
					# for i in self.rel_matrix:
					#     input()
					#     self.matrix = i
					#     draw.construct_rdg(self)
					# # encoded_matrix = opr.get_encoded_matrix(self)
					# # if(not any(np.array_equal(encoded_matrix, i) for i in self.encoded_matrix)):
					#     # self.encoded_matrix.append(encoded_matrix)
					#     draw.draw_rdg(self,pen)
					self.cip = self.original_cip.copy()
					
			else:
				self.cip = self.cip_list[0]
				self.original_cip = self.cip.copy()
				for k in self.cip_list:
					node_color = self.node_color
					self.cip = k
					news.add_news_vertices(self)
					# print(self.cip)
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
					print("North Boundary: ", self.cip[0])
					print("East Boundary: ", self.cip[1])
					print("South Boundary: ", self.cip[2])
					print("West Boundary: ",self.cip[3])
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
			printe("Number of different floor plans: ")
			printe(len(self.rel_matrix))
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
		# self.cip = news.find_cip_single(self)
			start = time.time()
			# print(self.node_count)
			# print(self.edge_count)
			# print(self.matrix)
			# self.boundaries = opr.find_possible_boundary(opr.ordered_outer_boundary(self))
			# self.cip_list = news.populate_cip_list(self)
			# self.cip = self.cip_list[0]
			# self.original_cip = self.cip.copy()
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
			# self.cip = news.find_cip_single(self)
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

				self.cip = self.cip_list[0]
				self.original_cip = self.cip.copy()
				for k in self.cip_list:
					node_color1 = self.node_color.copy()
					node_color2 = self.node_color.copy()
					self.cip = k
					news.add_news_vertices(self)
					# print(self.cip)
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
					print("North Boundary: ", self.cip[0])
					print("East Boundary: ", self.cip[1])
					print("South Boundary: ", self.cip[2])
					print("West Boundary: ",self.cip[3])
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
					r=0
					for i in rel_matrix:
						self.matrix = i
						if (r==10):
							break
						r+=1
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
			printe("Number of different floor plans: ")
			printe(len(rel_matrix))
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
			r=0
			for i in self.rel_matrix:
				# print("Press enter to get a new floorplan")
				# input()
				if(r==10):
					break
				r+=1
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
					draw.draw_rdg(self,origin_count,pen,self.to_be_merged_vertices[count],mode)
					origin_count +=1
					count +=1
					self.node_color = self.node_color_list[count]
					draw.construct_rdg(self,self.to_be_merged_vertices[count],self.rdg_vertices2[count])
					if(origin_count != 1):
						self.origin += 500
					draw.draw_rdg(self,origin_count,pen,self.to_be_merged_vertices[count],mode)
					origin_count +=1
					count +=1
					
				else:
					self.node_color = self.node_color_list[origin_count-1]
					draw.construct_rdg(self,self.to_be_merged_vertices,self.rdg_vertices)
					if(origin_count != 1):
						self.origin += 500
					draw.draw_rdg(self,origin_count,pen,self.to_be_merged_vertices,mode)
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
		
	def create_multiple_floorplan(self,pen,textbox,mode):
		global box
		box = textbox
		self.create_multiple_dual(0,pen,textbox)
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
		r=0
		for i in range(0,len(self.rel_matrix)):
		# print("Press enter to get a new floorplan")
		# input()
			if(r==10):
				break
			r+=1
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
				[width,height,hor_dgph] = floorplan_to_st(A,self.inp_min,self.inp_height)
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
				draw.draw_rdg(self,(count+1),pen,self.to_be_merged_vertices[count],mode)
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
				draw.draw_rdg(self,(count+1),pen,self.to_be_merged_vertices[count],mode)
				self.area =[]
				origin_count+=1
				if(origin_count != 1):
					self.origin += 1000
				
				count+=1
			else:
				self.node_color = self.node_color_list[count]
				draw.construct_rdg(self,self.to_be_merged_vertices,self.rdg_vertices)
				self.encoded_matrix = opr.get_encoded_matrix(self)
				B = copy.deepcopy(self.encoded_matrix)
				A = copy.deepcopy(self.encoded_matrix)
				[width,height,hor_dgph] = floorplan_to_st(A,self.inp_min,self.inp_height)
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
					self.origin += 1000
				opr.calculate_area(self,self.to_be_merged_vertices,self.rdg_vertices)
				draw.draw_rdg(self,(count+1),pen,self.to_be_merged_vertices,mode)
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



	  
		



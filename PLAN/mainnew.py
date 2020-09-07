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
import matplotlib.pyplot as plt
import checker
from tkinter import messagebox
import dimension_gui as dimgui
from Dissected_RFP import dissected
def run():
	def printe(string):
		gclass.textbox.insert('end',string)
		gclass.textbox.insert('end',"\n")

	warnings.filterwarnings("ignore") 
	gclass = gui.gui_class()

	while (gclass.command!="end"):
		
		# print(G.graph.edges())
		if(gclass.command=="dissection"):
			matrix, dimensioned_matrix = dissected().submit()
			
			cir =nx.Graph()
			cir = nx.from_numpy_matrix(matrix)
			print("hi")
			nx.draw(cir)
			plt.show()
		elif( gclass.command =="checker"):
			check.checker(gclass.value,gclass.textbox)
		else:
			G = ptpg.PTPG(gclass.value)
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
				rnames.append("Corridor")
				for i in range(0,100):
					rnames.append("")
				# print(rnames)
				
				parameters= [len(spanned), spanned.size() , spanned.edges() , 0,0 ,rnames,colors]
				C = ptpg.PTPG(parameters)
				C.create_single_dual(1,gclass.pen,gclass.textbox)

			elif(gclass.command == "single"):
				test_result = checker.gui_checker(G.matrix)
				if(not test_result[0]):
					messagebox.showerror("Invalid Graph", "Graph is not planar")
				elif(not test_result[1]):
					messagebox.showerror("Invalid Graph", "Graph is not triangular")
				elif(not test_result[2]):
					messagebox.showerror("Invalid Graph", "Graph is not biconnected")
				else:
					if(G.dimensioned == 0):
						G.create_single_dual(1,gclass.pen,gclass.textbox)
					else:
						if(G.rfp_test == False):
							G.create_single_dual(2,gclass.pen,gclass.textbox)
							messagebox.showinfo("Orthogonal Floor Plan","The input graph has an orthogonal floorplan.Rooms with red boundary are the additional rooms which will be added but later merged.Please provide dimensions for the extra rooms as well.")
							print(G.original_node_count+len(G.to_be_merged_vertices))
							value1 = dimgui.gui_fnc(G.original_node_count+len(G.to_be_merged_vertices))
							G.inp_min = value1[0]
							G.inp_height = value1[2]
							# if value1[3] == 0:
							# 	G.ar_pmt = 1
							# 	G.ar_min = self.value1[1]
							# 	G.ar_max = self.value1[2]
							gclass.pen.clear()
							G.create_single_floorplan(gclass.pen,gclass.textbox,1)

						else:
							G.create_single_floorplan(gclass.pen,gclass.textbox,0)
			elif(gclass.command == "multiple"):
				test_result = checker.gui_checker(G.matrix)
				if(not test_result[0]):
					messagebox.showerror("Invalid Graph", "Graph is not planar")
				elif(not test_result[1]):
					messagebox.showerror("Invalid Graph", "Graph is not triangular")
				elif(not test_result[2]):
					messagebox.showerror("Invalid Graph", "Graph is not biconnected")
				else:
					if(G.dimensioned == 0):
						G.create_multiple_dual(1,gclass.pen,gclass.textbox)
						
					else:
						G.create_multiple_floorplan(gclass.pen,gclass.textbox,0)

		gclass.root.wait_variable(gclass.end)
		gclass.tbox.clear()
		gclass.graph_ret()
		gclass.ocan.add_tab()

		# gclass.ocan.tscreen.resetscreen()
		gclass.pen = gclass.ocan.getpen()
		gclass.pen.speed(100000)

if __name__ == "__main__":
	run()
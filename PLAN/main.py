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
import matplotlib.pyplot as plt
import warnings
import check
import circulation
import plotter
import screen
import checker
from tkinter import messagebox
import dimension_gui as dimgui
# from Dissected_RFP import dissected
def run():
	def printe(string):
		gclass.textbox.insert('end',string)
		gclass.textbox.insert('end',"\n")

	warnings.filterwarnings("ignore") 
	gclass = gui.gui_class()

	while (gclass.command!="end"):
		# print(G.graph.edges())
		if(gclass.command=="dissection"):
			# gclass.frame2.destroy
			# dclass = dissected(gclass.frame2)
			# gclass
			# dclass.end()
			# matrix = dissected().submit()
			print(gclass.dclass.mat)

			dis =nx.Graph()
			dis = nx.from_numpy_matrix(gclass.dclass.mat)
			m = len(dis)
			
			spanned = circulation.BFS(dis,gclass.e1.get(),gclass.e2.get())
			
			colors = ['#4BC0D9','#76E5FC','#6457A6','#5C2751','#7D8491','#BBBE64','#64F58D','#9DFFF9','#AB4E68','#C4A287','#6F9283','#696D7D','#1B1F3B','#454ADE','#FB6376','#6C969D','#519872','#3B5249','#A4B494','#CCFF66','#FFC800','#FF8427','#0F7173','#EF8354','#795663','#AF5B5B','#667761','#CF5C36','#F0BCD4','#ADB2D3','#FF1B1C','#6A994E','#386641','#8B2635','#2E3532','#124E78']*10

			rnames = []
			for i in range(1,m+1):
				rnames.append('Room' + str(i))
			rnames.append("Corridor")
			for i in range(1,10):
				colors[m+i-1] = '#FF4C4C'
				rnames.append("")
				
			parameters= [len(spanned), spanned.size() , spanned.edges() , 0,0 ,rnames,colors]
			C = ptpg.PTPG(parameters)
			C.create_single_dual(1,gclass.pen,gclass.textbox)
			
            # self.master.app = self.master.PlotApp(self.master.frame2,self.master)
			# plotter.plot(spanned,m)
			# nx.draw_spring(dis, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
			# plt.show()
			# nx.draw_spring(dis_cir, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
			# plt.show()
		if ( gclass.command =="checker"):
			check.checker(gclass.value,gclass.textbox)
		else:
			G = ptpg.PTPG(gclass.value)

			printe("\nEdge Set")
			printe(G.graph.edges())
			frame = tk.Frame(gclass.root)
			frame.grid(row=2,column=1)
			if (gclass.command == "circulation"):
				m =len(G.graph)
				spanned = circulation.BFS(G.graph,1,2)
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
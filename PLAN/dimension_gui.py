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
	min_width = []
	max_width = []
	min_height =[]
	max_height = []
	root = tk.Toplevel() 

	root.title('Dimensional Input')
	root.geometry(str(1000) + 'x' + str(400))
	Upper_right = tk.Label(root, text ="Enter dimensional constraints for each room",font = ("Times New Roman",12)) 
	  
	Upper_right.place(relx = 0.70,  
					  rely = 0.1, 
					  anchor ='ne')

	text_head_width = []
	# text_head_width1 = []
	text_head_area = []
	# text_head_area1 = []
	text_room = []
	value_width= []
	# value_width1 = []
	value_area =[]
	# value_area1 =[]
	w = []
	# w1=[]
	minA = []
	# maxA = []
	
	
	for i in range(0,nodes):
		i_value_x = int((i/10))
		i_value_y = i%10
		w.append(tk.IntVar(None))
		# w1.append(tk.IntVar(None)) 
		minA.append(tk.IntVar(None)) 
		# maxA.append(tk.IntVar(None)) 
		if(i_value_y == 0):
			text_head_width.append("text_head_width_"+str(i_value_x+1))
			text_head_width[i_value_x] = tk.Label(root,text = "Min Width")
			text_head_width[i_value_x].place(relx = 0.30 + 0.30*i_value_x,  
					  rely = 0.2, 
					  anchor ='ne')
			# text_head_width1.append("text_head_width1_"+str(i_value_x+1))
			# text_head_width1[i_value_x] = tk.Label(root,text = "Max Width")
			# text_head_width1[i_value_x].place(relx = 0.40 + 0.20*i_value_x,  
			# 		  rely = 0.2, 
			# 		  anchor ='ne')
			text_head_area.append("text_head_area_"+str(i_value_x+1))
			text_head_area[i_value_x] = tk.Label(root,text = "Min Height")
			text_head_area[i_value_x].place(relx = 0.40 + 0.30*i_value_x,  
					  rely = 0.2, 
					  anchor ='ne')
			# text_head_area1.append("text_head_area1_"+str(i_value_x+1))
			# text_head_area1[i_value_x] = tk.Label(root,text = "Max Height")
			# text_head_area1[i_value_x].place(relx = 0.60 + 0.20*i_value_x,  
			# 		  rely = 0.2, 
			# 		  anchor ='ne')
		text_room.append("text_room_"+str(i))
		text_room[i] = tk.Label(root, text ="Room"+str(i),font = ("Times New Roman",8)) 

		text_room[i].place(relx = 0.20 + 0.30*i_value_x,  
					  rely = 0.3 + (0.05 * i_value_y), 
					  anchor ='ne')
		value_width.append("value_width" + str(i))
		value_width[i] = tk.Entry(root, width = 5,textvariable=w[i])
		value_width[i].place(relx = 0.30 +0.30*i_value_x,  
					  rely = 0.3 +(0.05)*i_value_y, 
					  anchor ='ne')
		# value_width1.append("value_width1" + str(i))
		# value_width1[i] = tk.Entry(root, width = 5,textvariable=w1[i])
		# value_width1[i].place(relx = 0.40 +0.20*i_value_x,  
		# 			  rely = 0.3 +(0.05)*i_value_y, 
		# 			  anchor ='ne')
		value_area.append("value_area"+str(i))
		value_area[i] = tk.Entry(root, width = 5,textvariable=minA[i])
		value_area[i].place(relx = 0.40+ 0.30*i_value_x,  
					  rely = 0.3 + (0.05)*i_value_y, 
					   anchor ='ne')
		# value_area1.append("value_area1"+str(i))
		# value_area1[i] = tk.Entry(root, width = 5,textvariable=maxA[i])
		# value_area1[i].place(relx = 0.60+ 0.20*i_value_x,  
		# 			  rely = 0.3 + (0.05)*i_value_y, 
		# 			   anchor ='ne')
	def button_clicked():
		for i in range(0,nodes):
			min_width.append(int(value_width[i].get()))
			# max_width.append(int(value_width1[i].get()))
			min_height.append(int(value_area[i].get()))
			# max_height.append(int(value_area1[i].get()))
		# if(len(width) != nodes or len(ar) != nodes or len(ar1)!= nodes):
		# 	messagebox.showerror("Invalid DImensions", "Some entry is empty")
		# print(width)
		# print(area)
		# else:
		root.destroy()

	# def clicked():
	# 	if(checkvar1.get() == 0):
	# 		for i in range(0,nodes):
	# 			value_area[i].config(state="normal")
	# 			value_area1[i].config(state="normal")
	# 			ar = []
	# 			ar1 = []
	# 	else:
	# 		for i in range(0,nodes):
	# 			value_area[i].config(state="disabled")
	# 			value_area1[i].config(state="disabled")
	# 			ar = []
	# 			ar1 = []

	button = tk.Button(root, text='Submit', padx=5, command=button_clicked)      
	button.place(relx = 0.5,  
					  rely = 0.9, 
					  anchor ='ne')
	# checkvar1 = tk.IntVar()
	# c1 = tk.Checkbutton(root, text = "Default AR Range", variable = checkvar1,onvalue = 1, offvalue = 0,command=clicked)
	# c1.place(relx = 0.85, rely = 0.9, anchor = 'ne')

	root.wait_window(root)
	print(min_width,max_width,min_height,max_height)
	return min_width,max_width,min_height,max_height
	

if __name__ == "__main__":
	gui_fnc(3)
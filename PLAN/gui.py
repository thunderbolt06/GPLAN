import ast
import json
import os
import pickle
import random
import sys
import tkinter as tk
import tkinter.ttk as ttk
import turtle
import warnings
from tkinter import ALL, EventType, Label, Menu, filedialog, messagebox
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageTk
import tablenoscroll
import numpy as np
import final
done = True
col = ["white","#9A8C98","light grey","white"]
font={'font' : ("lato bold",10,"")}
warnings.filterwarnings("ignore") 

class treenode:
    
    def __init__(self, parent, left, right, height, width, slice_type,d1,d2,d3,d4):
        self.parent = parent
        self.left = left
        self.right = right
        self.height = height
        self.width = width
        self.slice_type = slice_type
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.d4 = d4

class gui_class:
    def __init__(self):
        self.open = False
        self.command = "Null"
        self.value = []
        self.root =tk.Tk()
        
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.maxsize(screen_width,screen_height)
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
        self.e1 = tk.IntVar()
        self.e2 = tk.IntVar()

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
        self.dclass = None
        self.dissecting = 1
        self.root_window = self.ocan.getroot()   
        self.root.wait_variable(self.end)
        self.graph_ret()
        while((self.value[0] == 0) and done):
            # print(done)
            self.root.wait_variable(self.end)
            self.value=self.app.return_everything()
            tk.messagebox.showinfo("error","The graph is empty , please draw a graph")

    class Nodes:
        def __init__(self,id,x,y):
            self.circle_id=id
            self.pos_x=x
            self.pos_y=y
            self.radius=15
            self.adj_list=[]
        
        def clear(self):
            self.circle_id=-1
            self.pos_x=0
            self.pos_y=0
            self.radius=0
            self.adj_list=[]
    
    class PlotApp:

        def __init__(self, toframe,master):
            root = tk.Frame(toframe)
            root.grid(row=0,column=0)
            self.root = root
            self.l1 = tk.Label(root,text='Draw a test graph here',bg=col[2])
            self.l1.grid(row=0,column=0)
            self._root = root
            self.radius_circle=15
            self.rnames = []
            self.master = master
            self.command = "Null"
            self.table = tablenoscroll.Table(self.master.frame5,["Index", "Room Name"], column_minwidths=[None, None])
            self.table.pack(padx=10,pady=10)
            self.table.config(bg="#F4A5AE")
            self.createCanvas()

        colors = ['#4BC0D9','#76E5FC','#6457A6','#5C2751','#7D8491','#BBBE64','#64F58D','#9DFFF9','#AB4E68','#C4A287','#6F9283','#696D7D','#1B1F3B','#454ADE','#FB6376','#6C969D','#519872','#3B5249','#A4B494','#CCFF66','#FFC800','#FF8427','#0F7173','#EF8354','#795663','#AF5B5B','#667761','#CF5C36','#F0BCD4','#ADB2D3','#FF1B1C','#6A994E','#386641','#8B2635','#2E3532','#124E78']

        # colors = ['#edf1fe','#c6e3f7','#e1eaec','#e5e8f2','#def7fe','#f1ebda','#f3e2c6','#fff2de','#ecdfd6','#f5e6d3','#e3e7c4','#efdbcd','#ebf5f0','#cae1d9','#c3ddd6','#cef0cc','#9ab8c2','#ddffdd','#fdfff5','#eae9e0','#e0dddd','#f5ece7','#f6e6c5','#f4dbdc','#f4daf1','#f7cee0','#f8d0e7','#efa6aa','#fad6e5','#f9e8e2','#c4adc9','#f6e5f6','#feedca','#f2efe1','#fff5be','#ffffdd']
        nodes_data=[]
        id_circle=[]
        name_circle= []
        edge_count=0
        hex_list = []
        multiple_rfp = 0
        cir=0
        edges=[]
        random_list = []
        connection=[]
        oval = [] 
        rcanframe = []
        abc = 0
        xyz = 0
        elines = []

        def return_everything(self):
            return [len(self.nodes_data),self.edge_count,self.edges,self.command,self.master.checkvar1.get(),list(filter(None, [row[1].get() for row in self.table._data_vars])),self.hex_list]

        def createCanvas(self):
            self.id_circle.clear()
            self.name_circle.clear()
            for i in range(0,100):
                self.id_circle.append(i)
            for i in range(0,100):
                self.name_circle.append("Room "+ str(i))
            self.nodes_data.clear()
            self.edges.clear()
            self.table._pop_all()
            self.edge_count = 0
            self.oval.clear()
            self.rcanframe.clear()
            self.abc  =0
            self.hex_list.clear()
            self.xyz = 0
            self.elines.clear()
            # border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}
            self.canvas = tk.Canvas(self._root,bg=col[3], width=1000, height=370)
            self.canvas.grid(column=0,row =1, sticky='nwes')
            self.canvas.bind("<Button-3>",self.addH)
            self.connection=[]
            self.canvas.bind("<Button-1>",self.button_1_clicked)    
            self.canvas.bind("<Button-2>",self.remove_node)
            self.ButtonReset = tk.Button(self._root, text="Reset",fg='white',width=10,height=2 ,**font,relief = 'flat',bg=col[1] ,command=self.reset)
            self.ButtonReset.grid(column=0 ,row=1,sticky='n',pady=20,padx=40)
            
            self.instru = tk.Button(self._root, text="Instructions",fg='white',height=2 , **font ,relief = 'flat', bg=col[1] ,command=self.instructions)
            self.instru.grid(column=0 ,row=1,sticky='wn',pady=22,padx=40)

            self.lay = tk.Button(self._root, text="Switch to Layout",fg='white',height=2 ,**font,relief = 'flat',bg=col[1] ,command=self.switch)
            self.lay.grid(column=0 ,row=1,sticky='ne',pady=20,padx=40)

        def switch(self):
            self.master.root.quit()
            final.run()
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
            self.table.insert_row(list((id_node,self.rname.get())),self.table._number_of_rows)
            self.name_circle.pop(0)
            self.rframe.grid(row=0,column=1)
            self.oval.append(self.canvas.create_oval(x-self.radius_circle,y-self.radius_circle,x+self.radius_circle,y+self.radius_circle,width=3, fill=hex_number,tag=str(id_node)))
            # self.canvas.create_text(x,y-self.radius_circle-9,text=str(id_node),font=("Purisa",14))
            # self.buttonBG = self.canvas.create_rectangle(x-15,y-self.radius_circle-20, x+15,y-self.radius_circle, fill="light grey")
            # self.buttonTXT = self.canvas.create_text(x,y-self.radius_circle-9, text="click")
            self.rcanframe.append(self.canvas.create_window(x,y-self.radius_circle-12, window=self.rframe))
            # self.canvas.tag_bind(self.buttonBG, "<Button-1>", self.room_name) ## when the square is clicked runs function "clicked".
            # self.canvas.tag_bind(self.buttonTXT, "<Button-1>", self.room_name) ## same, but for the text.
            # def _on_configure(self, event):
                # self.entry.configure(width=event.width)
            self.entry = tk.Entry(self.rframe,textvariable=self.table._data_vars[self.id_circle[0]-1][1],relief='flat',justify='c',width=15,bg=col[2])
            # self.rframe.bind("<Configure>", _on_configure)
            
            # but =tk.Button(self.rframe)
            # but.grid()
            self.entry.grid()
            # print(self.rname.get())
        def button_1_clicked(self,event):
            if len(self.connection)==2:
                self.canvas.itemconfig(self.oval[self.xyz],outline='black')
                self.canvas.itemconfig(self.oval[self.abc],outline='black')
                self.connection=[]
            if len(self.nodes_data)<=1:
                tk.messagebox.showinfo("Connect Nodes","Please make 2 or more nodes")
                return
            x, y = event.x, event.y
            value=self.get_id(x,y)
            self.abc= self.xyz
            self.xyz= self.nodes_data[value].circle_id
            self.hover_bright(event)
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
            self.elines.append([self.canvas.create_line(node1_x,node1_y,node2_x,node2_y,width=3),connections])

        def get_id(self,x,y):
            for j,i in enumerate(self.nodes_data):
                distance=((i.pos_x-x)**2 + (i.pos_y-y)**2)**(1/2)
                if distance<=self.radius_circle:
                    return j
            tk.messagebox.showinfo("Connect Nodes","You have clicked outside all the circles. Please try again")
            return -1
        
        def remove_node(self,event):
            id = self.get_id(event.x,event.y)
            # id = self.nodes_data[id].circle_id
            self.canvas.delete(self.oval[id])
            self.canvas.delete(self.rcanframe[id])
            for i in self.elines:
                if i[1][0]==id or i[1][1]==id:
                    self.canvas.delete(i[0])
                    self.edges.remove(i[1])
                    self.edge_count-=1
            self.nodes_data[id].clear()
            self.nodes_data.pop(id)
            self.hex_list.pop(id)
            for j in range(self.table.number_of_columns):
                self.table._data_vars[id][j].set("")
            self.table._data_vars.pop(id)
            # self.edges.pop(id)
            # self.table.delete_row(id)
            i = id
            # while i < self.table._number_of_rows-1:
            #     row_of_vars_1 = self.table._data_vars[i]
            #     row_of_vars_2 = self.table._data_vars[i+1]

            #     j = 0
            #     while j <self.table._number_of_columns:
            #         row_of_vars_1[j].set(row_of_vars_2[j].get())
            #         j+=1
            #     i += 1

            # self.table._pop_n_rows(1)
            # self.table._number_of_rows-=1
            # self.table._data_vars.pop(id)
            for j in range(self.table.number_of_columns):
                self.table.grid_slaves(row=i+1, column=j)[0].destroy()
            self.table._number_of_rows -=1

            # if self.table._on_change_data is not None: self.table._on_change_data()
        
        def hover_bright(self,event):
            self.canvas.itemconfig(self.oval[self.xyz],outline='red')
        
        def reset(self):
            self.canvas.destroy()
            self.createCanvas()

    class dissected:
        def __init__(self,master,win=None):
            if(win is None):
                win=tk.Tk()
            tk.Entry(win)
            self.master = master
            self.current = None
            self.rootnode = None
            self.prev = None
            self.index = 0
            self.s = 25
            self.scale = 0
            self.startx = 10
            self.starty = 60
            self.rooms = 1
            self.ch = 0
            self.cir = []
            self.leaves = []
            self.mat = 0
            self.dim_mat = 0
            wd = 672
            ht = 345

            root = tk.Frame(win)
            
            tk.Label(win,text = 'Create a Dissected Floor Plan').grid(row=0)
            root.grid(row=1,column=0, sticky="n")
            self.root = root
            

            
            border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}
            
            self.canvas=tk.Canvas(root, width=wd, height=ht, background='white', **border_details)
            self.canvas.grid(row=1,column=0,columnspan=5)
            
            self.canvas2=tk.Canvas(root, width=340, height=ht, background='white',**border_details)
            self.canvas2.grid(row=1,column=5,columnspan=2)
            self.canvas2.create_text(120,30,fill='black',font="Times 16 italic bold",text=' Dimensions of Rooms ')
            self.canvas2.create_line(0,50,340,50)
            
            self.type = " Rectangular plot "
            self.popup = tk.Menu(root, tearoff=0)
            self.popup.add_command(label="Horizontal Slice", command=self.addH)
            self.popup.add_command(label="Vertical Slice", command=self.addV)
            self.popup.add_separator()
            self.popup.add_command(label="Make a Room",command=self.addLeaf)

            showButton = tk.Button(root, text=" Start a new dissection ",command=self.choice)
            showButton.grid(row=0, column=0)
            tk.Button(root, text=" Generate Spanning Circulation ",command=self.end).grid(row=2,column=0)
            
            tk.Button(root, text='Leave',command=self.forceExit).grid(row=0,column=1)
            tk.Button(root, text=' Change Entry Point ',command=self.changeentry).grid(row=2,column=1)
            tk.Label(root, text=" Dimensions of Total Plot ").grid(row=2,column=2)
            tk.Label(root, text=" Height  ").grid(row=2,column=3)
            tk.Label(root, text=" Width   ").grid(row=2,column=5)
            
            self.en1 = tk.Entry(root)
            self.en2 = tk.Entry(root)
        
            self.en1.grid(row=2, column=4)
            self.en2.grid(row=2, column=6)
            
            tk.Label(root, text=" Dimensions of Current block ").grid(row=0,column=2)
            tk.Label(root, text=" Height  ").grid(row=0,column=3)
            tk.Label(root, text=" Width   ").grid(row=0,column=5)
            
            self.ent1 = tk.Entry(root)
            self.ent2 = tk.Entry(root)

            self.master.e1 = tk.IntVar()
            self.master.e2 = tk.IntVar()
            self.master.e1.set(1)
            self.master.e2.set(2)

        
            self.ent1.grid(row=0, column=4)
            self.ent2.grid(row=0, column=6)
            self.done= True
            # if(self.done == 1):
            self.endvar = tk.IntVar()
            self.endvar.set(0)
            
            self.root.wait_variable(self.endvar)
            # while(self.done):
                # self.root.wait_variable(self.endvar)

        def forceExit(self):
            self.root.destroy()
            self.master.root.destroy()
            
        def changeentry(self):
            top = tk.Toplevel()
            
            tk.Label(top, text="Enter adjcent rooms to entry room"+self.type).grid(row=0,columnspan=2)
            tk.Label(top, text=" Left room  ").grid(row=1)
            tk.Label(top, text=" Right room   ").grid(row=2)
            
            entry1 = tk.Entry(top,textvariable = self.master.e1)
            entry2 = tk.Entry(top,textvariable = self.master.e2)
        
            entry1.grid(row=1, column=1)
            entry2.grid(row=2, column=1)
            # entry1.insert(0,1)
            # entry2.insert(0,2)
            
            but1 = tk.Button(top,text="   Submit   ", command=top.destroy)
            but1.grid(row=3, columnspan=2)

        def end(self):
            # self.done = 0
            if( self.rooms>1):
                self.submit()
                self.root.quit()
                self.done = False
                self.master.value.append(1)
                self.endvar.set(self.endvar.get()+1)
                self.master.end.set(self.master.end.get()+1)
            else :
                tk.messagebox.showinfo("error","Please make a dissection of two or more rooms")

        def start(self,canvas):
            global type
            global entry1
            global entry2
            global master
            global current
            current = self.current
            master = tk.Tk()
            tk.Label(master, text="Enter dimensions of "+self.type).grid(row=0,columnspan=2)
            tk.Label(master, text=" Height  ").grid(row=1)
            tk.Label(master, text=" Width   ").grid(row=2)
            
            entry1 = tk.Entry(master)
            entry2 = tk.Entry(master)
        
            entry1.grid(row=1, column=1)
            entry2.grid(row=2, column=1)
            entry1.insert(0,10)
            entry2.insert(0,10)
            but1 = tk.Button(master,text="   Save   ", command=lambda:self.saveDimsRect(canvas))
            but1.grid(row=3, columnspan=2)
            if current is not None and current.slice_type == "V" and current.height!=0:
                entry1.insert(0,current.height)
                entry1.configure(state = 'disabled')
                entry2.insert(0,current.width/2)
            if current is not None and current.slice_type == "H" and current.width!=0:
                entry2.insert(0,current.width)
                entry1.insert(0,current.height/2)
                entry2.configure(state = 'disabled')
                
                
            if current is not None and current.slice_type == 'L' and current.height!=0 and current.width!=0:
                entry2.insert(0,current.width)
                entry1.insert(0,current.height)
            
            master.mainloop()
            
        def error(self,h_val,w_val,rr):
            global master
            if rr == 0:
                master.destroy()
            if rr == 1:
                
                if w_val != 0:
                    box1 = tk.Tk()
                    if ch == 1:
                        tk.Label(box1, text="The width should be less than "+str(w_val)).grid(row=0)
                    if ch == 2:
                        tk.Label(box1, text="The width should be less than or equal to "+str(w_val)).grid(row=0)
                    but2 = tk.Button(box1,text="Okay", command=box1.destroy)
                    but2.grid(row=1)
                    box1.mainloop() 
                    
                if h_val != 0:
                    box2 = tk.Tk()
                    if ch == 1:
                        tk.Label(box2, text="The height should be less than "+str(h_val)).grid(row=0)
                    if ch == 2:
                        tk.Label(box2, text="The height should be less than or equal to "+str(h_val)).grid(row=0)
                    but2 = tk.Button(box2,text="Okay", command=box2.destroy)
                    but2.grid(row=1)
                    box2.mainloop()

        def disp(self,event,length,breadth):
            self.ent1.insert(0,length)
            self.ent2.insert(0,breadth)

        def remove(self,event,length,breadth):
            self.ent1.delete(0,"end")
            self.ent2.delete(0,"end")
            
        def saveDimsRect(self,canvas):
            global rootnode
            rootnode = self.rootnode
            global current
            current = self.current
            global v1
            global v2
            global temp1
            global temp2
            global fig1
            global fig2
            global master5
            global scale
            if rootnode is None:
            
                v1 = float(entry1.get())
                v2 = float(entry2.get())
                if v1 <=30 and v2 <=40:
                    scale = 20
                elif v1<=45 and v2 <=50:
                    scale = 15
                elif v1<=65 and v2<=90:
                    scale = 10
                elif v1<=110 and v2<=160:
                    scale = 6
                elif v1<=220 and v2<=320:
                    scale = 3
                elif v1<=680 and v2<= 980:
                    scale = 1
                elif v1<=1360 and v2<= 1960:
                    scale = 0.5
                elif v1<=2720 and v2<=3920:
                    scale = 0.25
                else:
                    box3 = tk.Tk()
                    tk.Label(box3, text="Try with smaller dimensions! ").grid(row=0)
                    but2 = tk.Button(box3,text="Okay", command=box3.destroy)
                    but2.grid(row=1)
                    box3.mainloop()
                    quit()
                self.error(0,0,0)
                self.en1.insert(0,v1)
                self.en2.insert(0,v2)
                fig1 = canvas.create_rectangle(10,10,(v2*scale)+10,(v1*scale)+10, fill="snow2")
                canvas.tag_bind(fig1, "<Enter>", lambda event, arg1 = v1, arg2 = v2: self.disp(event,arg1,arg2))        
                canvas.tag_bind(fig1, "<Leave>", lambda event, arg1 = v1, arg2 = v2: self.remove(event,arg1,arg2))        
            
                rootnode = treenode(None,None,None,v1,v2,None,10,10,(v2*scale)+10,(v1*scale)+10)
                current = rootnode
                self.current = current
                self.rootnode = rootnode
                canvas.tag_bind(fig1,"<Button-3>",self.do_popup)
            
            elif current.slice_type == 'H':        
                rem = 0
                u_v1 = float(entry1.get())
                u_v2 = float(entry2.get())
                
                if u_v1 >= current.height and current.height!=0:
                    rem = 1
                self.error(current.height,0,rem)
                
                temp1 = treenode(current,None,None,u_v1,u_v2,None,current.d1,current.d2,current.d3,current.d2+(u_v1*scale))
                current.left = temp1
                fig1 = canvas.create_rectangle(temp1.d1,temp1.d2,temp1.d3,temp1.d4,fill="snow2")
                canvas.tag_bind(fig1, "<Enter>", lambda event, arg1 = u_v1, arg2 = u_v2: self.disp(event,arg1,arg2))        
                canvas.tag_bind(fig1, "<Leave>", lambda event, arg1 = u_v1, arg2 = u_v2: self.remove(event,arg1,arg2))  
                
                l_v1 = current.height-u_v1
                l_v2 = u_v2
                temp2 = treenode(current,None,None,l_v1,l_v2,None,current.d1,current.d2+(u_v1*scale),current.d3,current.d4)
                current.right = temp2            
                fig2 = canvas.create_rectangle(temp2.d1,temp2.d2,temp2.d3,temp2.d4,fill="snow2")
                canvas.tag_bind(fig2, "<Enter>", lambda event, arg1 = l_v1, arg2 = l_v2: self.disp(event,arg1,arg2))        
                canvas.tag_bind(fig2, "<Leave>", lambda event, arg1 = l_v1, arg2 = l_v2: self.remove(event,arg1,arg2))  
                current = self.current

                canvas.create_line(temp2.d1,temp2.d2,temp1.d3,temp1.d4)
                canvas.tag_bind(fig1,"<Button-1>",lambda event, arg1 = temp1: self.shadeRectangle1(event,arg1))
                canvas.tag_bind(fig2,"<Button-1>",lambda event, arg2 = temp2: self.shadeRectangle2(event,arg2))
            
            elif current.slice_type == 'V':
                rem = 0
                l_v1 = float(entry1.get())
                l_v2 = float(entry2.get())
                
                if l_v2 >= current.width and current.width!=0:
                    rem = 1
                self.error(0,current.width,rem)
                
                temp1 = treenode(current,None,None,l_v1,l_v2,None,current.d1,current.d2,(l_v2*scale)+current.d1,current.d4)
                current.left = temp1
                fig1 = canvas.create_rectangle(temp1.d1,temp1.d2,temp1.d3,temp1.d4,fill="snow2")
                canvas.tag_bind(fig1, "<Enter>", lambda event, arg1 = l_v1, arg2 = l_v2: self.disp(event,arg1,arg2))        
                canvas.tag_bind(fig1, "<Leave>", lambda event, arg1 = l_v1, arg2 = l_v2: self.remove(event,arg1,arg2)) 
                
                r_v1 = l_v1
                r_v2 = current.width-l_v2
                temp2 = treenode(current,None,None,r_v1,r_v2,None,current.d1+(l_v2*scale),current.d2,current.d3,current.d4)
                current.right = temp2            
                fig2 = canvas.create_rectangle(temp2.d1,temp2.d2,temp2.d3,temp2.d4,fill="snow2")
                canvas.tag_bind(fig2, "<Enter>", lambda event, arg1 = r_v1, arg2 = r_v2: self.disp(event,arg1,arg2))        
                canvas.tag_bind(fig2, "<Leave>", lambda event, arg1 = r_v1, arg2 = r_v2: self.remove(event,arg1,arg2)) 
                current = self.current

                canvas.create_line(temp2.d1,temp2.d2,temp1.d3,temp1.d4)
                canvas.tag_bind(fig1,"<Button-1>",lambda event, arg1 = temp1: self.shadeRectangle1(event,arg1))
                canvas.tag_bind(fig2,"<Button-1>",lambda event, arg2 = temp2: self.shadeRectangle2(event,arg2))
            
            elif current.slice_type == 'L':
                v1 = float(entry1.get())
                v2 = float(entry2.get())
                rem = 0
                hh = 0
                ww = 0
                if v1 > current.height and current.height!=0:
                    rem = 1
                    hh = current.height
                if v2 > current.width and current.width !=0:
                    rem = 1
                    ww = current.width
                self.error(hh,ww,rem)
                
                global var1
                global var2
                global var3
                global var4
                global rooms
                global starty
                starty = self.starty
                rooms = self.rooms
                startx = self.startx

                if v1==0 or v2==0:
                    canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4, fill='white')
                    
                elif v1 != current.height and v1!=0 and current.height!=0:
                    self.getVerInfo("top","bottom","for the block")
                    if var1 == 0 and var2 == 1:
                        canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4-((current.height*scale)-(v1*scale)), fill='snow3')
                        canvas.create_rectangle(current.d1,current.d2+(v1*scale),current.d3,current.d4, fill='white')
                        current.d4 = current.d4-((current.height*scale)-(v1*scale))
                        canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                        print("Yes")
                        
                    elif var1 == 1 and var2 == 0:
                        canvas.create_rectangle(current.d1,current.d2+((current.height*scale)-(v1*scale)),current.d3,current.d4, fill='snow3')
                        canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4-(v1*scale), fill='white')
                        current.d2 = current.d2+((current.height*scale)-(v1*scale))
                        canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                        print("Yes")
                
                elif v2!=current.width and v2!=0 and current.width!=0:
                    self.getHorInfo('left','right','for the block')
                    if var3 == 0 and var4 == 1:
                        canvas.create_rectangle(current.d1,current.d2,current.d3-((current.width*scale)-(v2*scale)),current.d4, fill='snow3')
                        canvas.create_rectangle(current.d1+(v2*scale),current.d2,current.d3,current.d4, fill='white')
                        current.d3 = current.d3-((current.width*scale)-(v2*scale))
                        canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                        print("Yes")
                        
                    elif var3 == 1 and var4 == 0:
                        canvas.create_rectangle(current.d1+((current.width*scale)-(v2*scale)),current.d2,current.d3,current.d4, fill='snow3')
                        canvas.create_rectangle(current.d1,current.d2,current.d3-(v2*scale),current.d4, fill='white')
                        current.d1 = current.d1+((current.width*scale)-(v2*scale))
                        canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                        print("Yes")
                
                elif v1 == current.height and v2 == current.width:
                    canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4, fill='snow3')
                    canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                
                if v1!=0 and v2!=0:
                    txt = 'R'+str(rooms)+'  H = '+str(v1)+'  W = '+str(v2)
                    self.canvas2.create_text(startx+160,starty+15,fill="black",font="Times 14 italic bold", text=txt)
                    self.canvas2.create_line(0,starty+30,340,starty+30)
                    starty = starty+30
                    rooms = rooms+1
                
                if current is current.parent.left:
                    if current.parent.right.slice_type != 'L':
                        current = current.parent.right
                elif current is current.parent.right:
                    if current.parent.left.slice_type != 'L':
                        current = current.parent.left
                if current.parent.right.slice_type == 'L' and current.parent.left.slice_type == 'L':
                    if current.parent is not rootnode and current.parent is current.parent.parent.left:
                        current = current.parent.parent.right
                    elif current.parent is not rootnode and current.parent is current.parent.parent.right:
                        current = current.parent.parent.left

                self.current = current
                self.rooms = rooms
                self.starty = starty
                self.startx = startx
            
        def shadeRectangle1(self,event,temp):
            global current
            current = self.current
            canvas = self.canvas
            shade1 = canvas.create_rectangle(temp.d1,temp.d2,temp.d3,temp.d4, fill='grey')
            canvas.tag_bind(shade1, "<Enter>", lambda event, arg1 = temp.height, arg2 = temp.width: self.disp(event,arg1,arg2))        
            canvas.tag_bind(shade1, "<Leave>", lambda event, arg1 = temp.height, arg2 = temp.width: self.remove(event,arg1,arg2))
            current = temp
            self.current = current
            canvas.tag_bind(shade1,"<Button-3>",self.do_popup)
                
        def shadeRectangle2(self,event,temp):
            global current
            current = self.current
            canvas = self.canvas
            shade2 = canvas.create_rectangle(temp.d1,temp.d2,temp.d3,temp.d4, fill='grey')
            canvas.tag_bind(shade2, "<Enter>", lambda event, arg1 = temp.height, arg2 = temp.width: self.disp(event,arg1,arg2))        
            canvas.tag_bind(shade2, "<Leave>", lambda event, arg1 = temp.height, arg2 = temp.width: self.remove(event,arg1,arg2))
            current = temp
            self.current = current
            canvas.tag_bind(shade2,"<Button-3>",self.do_popup)
            
        def do_popup(self,event):
            try:
                self.popup.tk_popup(event.x_root, event.y_root, 0)
            finally:
                self.popup.grab_release()
                
        def addH(self):
            global type
            self.type = " Upper Block of the Dissection "
            self.current.slice_type = 'H'
            self.start(self.canvas)
        
        def addV(self):
            global type
            self.type = " Left Block of the Dissection "
            current.slice_type = 'V'
            self.start(self.canvas)       
            
        def addLeaf(self):
            global current
            global starty
            global rooms
            global type
            #global master4
            current = self.current
            canvas = self.canvas
            rooms = self.rooms
            startx = self.startx
            starty = self.starty
            canvas2 = self.canvas2
            leaves = self.leaves
            current.slice_type = 'L'
            if ch == 1:
                canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4, fill='snow3')
                canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                
                txt = 'R'+str(rooms)+'  H = '+str(current.height)+'  W = '+str(current.width)
                canvas2.create_text(startx+160,starty+15,fill="black",font="Times 14 italic bold", text=txt)
                canvas2.create_line(0,starty+30,340,starty+30)
                starty = starty+30
                rooms = rooms+1
                leaves.append(current)

                if current is current.parent.left:
                    if current.parent.right.slice_type != 'L':
                        current = current.parent.right
                elif current is current.parent.right:
                    if current.parent.left.slice_type != 'L':
                        current = current.parent.left
                if current.parent.right.slice_type == 'L' and current.parent.left.slice_type == 'L':            
                    if current.parent is not rootnode and current.parent is current.parent.parent.left:
                        current = current.parent.parent.right
                    elif current.parent is not rootnode and current.parent is current.parent.parent.right:
                        current = current.parent.parent.left
                self.rooms = rooms
                self.current = current
                self.startx = startx
                self.starty = starty
                self.leaves = leaves
            
            elif ch == 2:
                type = " room "
                self.start(canvas)
            return

        def getVerInfo(self,a,b,c):
            global var1
            global var2
            global master5
            master5 = tk.Tk()
            tk.Label(master5, text=" Specify the location of Vacant Space "+c).grid(row=0,columnspan=2)
            butt1 = tk.Button(master5,text=" At the "+a,command = lambda:[master5.destroy(),self.setVal1(),master5.quit()])
            butt1.grid(row=1, columnspan=2)
            butt2 = tk.Button(master5,text=" At the "+b,command = lambda:[master5.destroy(),self.setVal2(),master5.quit()])
            butt2.grid(row=2, columnspan=2)
            master5.mainloop()
            return

        def getHorInfo(self,a,b,c):
            global var3
            global var4
            global master6 
            master6 = tk.Tk()
            tk.Label(master6, text=" Specify the location of Vacant Space "+c).grid(row=0,columnspan=2)
            butt1 = tk.Button(master6,text=" At the "+a,command = lambda:[master6.destroy(),self.setVal3,master6.quit()])
            butt1.grid(row=1, columnspan=2)
            butt2 = tk.Button(master6,text=" At the "+b,command = lambda:[master6.destroy(),self.setVal4,master6.quit()])
            butt2.grid(row=2, columnspan=2)
            master6.mainloop()
            return

        def setVal1(self):
            global var1
            global var2
            var1 = 1
            var2 = 0

        def setVal2(self):
            global var1
            global var2
            var1 = 0
            var2 = 1

        def setVal3(self):
            global var3
            global var4
            var3 = 1
            var4 = 0

        def setVal4(self):
            global var3
            global var4
            var3 = 0
            var4 = 1

        def submit(self):
            #master2 = tk.Tk()\
            ch = 1
            if ch == 1:
                leaves = self.leaves
                rooms = self.rooms
                # Adjacency matrix Generation
                mat = np.zeros(([rooms-1,rooms-1]), dtype=int)
                dim_mat = np.zeros(([rooms-1,rooms-1]), dtype=int)
                for j in range(0,len(leaves)):
                    leaf1 = leaves[j]

                    for i in range(0,len(leaves)):
                        leaf2 = leaves[i]
                        if not i == j:
                            if (leaf1.d1 == leaf2.d1 and leaf1.d4 == leaf2.d2):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                if leaf1.width <= leaf2.width:
                                    dim_mat[i,j] = leaf1.width
                                    dim_mat[j,i] = leaf1.width
                                else:
                                    dim_mat[i,j] = leaf2.width
                                    dim_mat[j,i] = leaf2.width
                            if (leaf1.d3 == leaf2.d1 and leaf1.d4 == leaf2.d4):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                if leaf1.height <= leaf2.height:
                                    dim_mat[i,j] = leaf1.height
                                    dim_mat[j,i] = leaf1.height
                                else:
                                    dim_mat[i,j] = leaf2.height
                                    dim_mat[j,i] = leaf2.height
                            if (leaf1.d3 == leaf2.d1 and leaf1.d2 == leaf2.d2):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                if leaf1.height <= leaf2.height:
                                    dim_mat[i,j] = leaf1.height
                                    dim_mat[j,i] = leaf1.height
                                else:
                                    dim_mat[i,j] = leaf2.height
                                    dim_mat[j,i] = leaf2.height
                            if (leaf1.d3 == leaf2.d3 and leaf1.d4 == leaf2.d2):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                if leaf1.width <= leaf2.width:
                                    dim_mat[i,j] = leaf1.width
                                    dim_mat[j,i] = leaf1.width
                                else:
                                    dim_mat[i,j] = leaf2.width
                                    dim_mat[j,i] = leaf2.width
                            if ((leaf1.d1 < leaf2.d1 and leaf1.d3 > leaf2.d1) and leaf1.d4 == leaf2.d2):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                dim_mat[i,j] = (leaf1.d3-leaf2.d1)/scale
                                dim_mat[j,i] = (leaf1.d3-leaf2.d1)/scale
                            if ((leaf2.d1 < leaf1.d1 and leaf2.d3 > leaf1.d1) and leaf1.d4 == leaf2.d2):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                dim_mat[i,j] = (leaf2.d3-leaf1.d1)/scale
                                dim_mat[j,i] = (leaf2.d3-leaf1.d1)/scale
                            if ((leaf1.d2 < leaf2.d2 and leaf1.d4 > leaf2.d2) and leaf1.d3 == leaf2.d1):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                dim_mat[i,j] = (leaf1.d4-leaf2.d2)/scale
                                dim_mat[j,i] = (leaf1.d4-leaf2.d2)/scale
                            if ((leaf2.d2 < leaf1.d2 and leaf2.d4 > leaf1.d2) and leaf1.d3 == leaf2.d1):
                                mat[i,j] = 1
                                mat[j,i] = 1
                                dim_mat[i,j] = (leaf2.d4-leaf1.d2)/scale
                                dim_mat[j,i] = (leaf2.d4-leaf1.d2)/scale
                            if dim_mat[i,j] == 0:
                                dim_mat[i,j] = -1
                                dim_mat[j,i] = -1                            
                print(mat)
                # print(dim_mat)
                self.mat = mat
                self.dim_mat = dim_mat
                # self.done= TRUE
                # root.quit()
                return mat, dim_mat
            
        def choice(self):
            master3 = tk.Tk()
            tk.Label(master3, text=" Select the type of Floor Plan to be generated ").grid(row=0,columnspan=2)
            but3 = tk.Button(master3,text=" RFP without empty spaces ", command=lambda:[master3.destroy(),self.decideRect()])
            but3.grid(row=1, columnspan=2)
            but4 = tk.Button(master3,text=" RFP with empty spaces ", command=lambda:[master3.destroy(),self.decideNonRect()])
            but4.grid(row=2, columnspan=2)
            master3.mainloop()

        def decideRect(self):
            global ch
            ch = 1
            self.start(self.canvas)
            
        def decideNonRect(self):
            global ch
            ch = 2
            self.start(self.canvas)

    class Buttons:
        def __init__(self,root,master):
            
            button_details={'wraplength':'150','bg':col[1],'fg':'white','font':('lato','14') , 'padx':5 ,'pady':5,'activebackground' : col[2] }
            b1 = tk.Button(master.frame1,width=15,text='A Floor Plan',relief='flat',**button_details,command=master.single_floorplan)
            b1.grid(row=1,column=0,padx=5,pady=5)
            
            b2 = tk.Button(master.frame1,width=15, text='Multiple Floor Plans',relief='flat',**button_details,command=master.multiple_floorplan)
            b2.grid(row=2,column=0,padx=5,pady=5)
            
            c1 = tk.Checkbutton(master.frame1, text = "Dimensioned",relief='flat',**button_details,selectcolor='#4A4E69',width=13 ,variable = master.checkvar1,onvalue = 1, offvalue = 0)
            c1.grid(row=3,column=0,padx=5,pady=5)
           
            b3 = tk.Button(master.frame1,width=15, text='Circulation',relief='flat',**button_details,command=master.circulation)
            b3.grid(row=4,column=0,padx=5,pady=5)
            
            b4 = tk.Button(master.frame1,width=15, text='RFPchecker' ,relief='flat',**button_details,command=master.checker)
            b4.grid(row=5,column=0,padx=5,pady=5)

            b7 = tk.Button(master.frame1,width=15, text='Dissection' ,relief='flat',**button_details,command=master.dissection)
            b7.grid(row=5,column=0,padx=5,pady=5)
            
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
            self.root=root
            self.root_window = tk.PanedWindow(root)
            self.l1 = tk.Label(self.root_window, text= 'Rectangular Dual')
            self.root_window.grid(row=2,column=0,pady=5)
            self.tabs = []
            self.tabno = -1
            self.tabControl = ttk.Notebook(self.root_window)
            # self.tabs = ttk.Frame(self.tabControl)
            
            # self.tabControl.add(self.tabs, text='Tab 1')
            self.tabControl.pack(expand=1, fill="both")
            # tk.Label(tabs, text="Welcome to GeeksForGeeks").grid(column=0, row=0, padx=30, pady=30)
            # tk.Label(self.tab2, text="Lets dive into the world of computers").grid(column=0, row=0, padx=30, pady=30)
            self.add_tab()
            
        def add_tab(self):
            self.tabno+=1
            self.tabs.append( ttk.Frame(self.tabControl) )
            self.tabControl.add(self.tabs[self.tabno], text='Tab '+str(self.tabno+1))
            self.tabControl.select(self.tabno)
            self.canvas = turtle.ScrolledCanvas(self.tabs[self.tabno],width=970,height=350)
            self.canvas.bind("<Double-Button-1>",self.zoom)
            self.canvas.grid(column=0, row=1, padx=2, pady=2)
            self.tscreen = turtle.TurtleScreen(self.canvas)
            self.tscreen.screensize(50000,1000)
            self.tscreen.bgcolor(col[3])
            self.pen = turtle.RawTurtle(self.tscreen)
            self.pen.speed(10000000)

            self.canvas.bind("<MouseWheel>",  self.do_zoom)
            self.canvas.bind('<Button-1>', lambda event: self.canvas.scan_mark(event.x, event.y))
            self.canvas.bind("<B1-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
            imname = "./close1.png"
            im1 = Image.open(imname).convert("1")
            size = (im1.width // 4, im1.height // 4)
            # im1.resize(size)
            # # im1.show()
            # im1 = ImageTk.BitmapImage(im1.resize(size)) 
            im2 = ImageTk.PhotoImage(Image.open(imname).resize(size))
            
            ImageTk.PhotoImage(file="./close1.png")
            # flat, groove, raised, ridge, solid, or sunke
            # self.canvas.create_image(20,20,anchor='ne',image=butimg)
            self.closeb = tk.Button(self.tabs[self.tabno],relief='solid',bg=col[3],activebackground=col[2],image=im2,command=self.close)
            self.closeb.image=im2
            self.closeb.grid(row=1,column=0,sticky='ne',pady=20,padx=70)
        def do_zoom(self,event):
            factor = 1.001 ** event.delta
            self.canvas.scale(ALL, event.x, event.y, factor, factor)

        def getpen(self):
            return self.pen

        def getroot(self):
            return self.root_window
        
        def zoom(self,event):
            self.canvas.config(width=self.root.winfo_screenwidth(),height=self.root.winfo_screenheight())

        def close(self):
            self.tabno-=1
            self.tabs.pop()
            self.tabControl.forget(self.tabControl.select())
    
    class output_text:
        def __init__(self,root):
            self.root=root
            self.textbox = tk.Text(root,bg=col[3],fg='black',relief='flat',height=32,width=30,padx=5,pady=5,**font)
            self.textbox.grid(row=0,column=0 ,padx=10,pady=10)
            self.textbox.insert('insert',"\t         Output\n")

        def gettext(self):
            return self.textbox

        def clear(self):
            self.textbox.destroy()
            self.textbox = tk.Text(self.root,bg=col[3],fg='black',relief='flat',height=32,width=30,padx=5,pady=5,**font)
            self.textbox.grid(row=0,column=0 ,padx=10,pady=10)
            self.textbox.insert('insert',"\t         Output\n")
   
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

    def dissection(self):
        # graphplotter.destroy()
        # self.disframe = tk.Frame(self.root)
        # self.disframe.grid(row=0,column=1)
        if self.dissecting:
            self.app.root.destroy()
            
            self.dclass = self.dissected(self,self.frame2)
            self.app.command="dissection"
            self.command="dissection"
            self.end.set(self.end.get()+1)
            global done
            done= False
            self.dissecting = 0
        else:
            self.app.table.destroy()
            self.app = self.PlotApp(self.frame2,self)
            self.dclass.root.destroy()
            self.dissecting = 1

    def restart(self):
        os.execl(sys.executable, os.path.abspath(__file__), *sys.argv)
    
    def exit(self):
        global done

        self.app.command="end"
        self.command = "end"
        self.end.set(self.end.get()+10)
        done = False
        
        # self.dclass.root.destory()
        print("ending")
        # self.saver = tk.Toplevel()
        # saverlabel = tk.Label(self.saver,text="hwakeoa")
        # saverlabel.pack()

        # b1 = tk.Button(self.saver,text="No",command=sys.exit(0))
        # b1.pack()
        # self.saver.wait_window(self.saver)


        self.root.quit()

        
        
        
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

        # with open('config.dictionary', 'rb') as config_dictionary_file:
        #     cself = pickle.load(config_dictionary_file)
        # # After config_dictionary is read from file
        # print(cself.value)

    def save_file(self):
        # self.root.filename = self.value
        f = filedialog.asksaveasfile( defaultextension=".txt",title = "Select file",filetypes = (("text files","*.txt"),("all files","*.*")),initialfile="Rectangular Dual Graph.txt")
        if f is None:
            return


        f.write(str(self.value))
        f.close()

        # with open('config.dictionary', 'wb') as config_dictionary_file:
        #     pickle.dump(self.app, config_dictionary_file)
    # def copy_to_file(self):

        
if __name__ == '__main__':
    value=gui_class()
    print(value.value)

6# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:13:03 2020

@author: Dipam Goswami
"""
import tkinter as tk
from tkinter import *
import turtle
import numpy as np

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

def start(canvas):
    global type
    global entry1
    global entry2
    global master
    global current
        
    master = tk.Tk()
    tk.Label(master, text="Enter dimensions of"+type).grid(row=0,columnspan=2)
    tk.Label(master, text=" Height  ").grid(row=1)
    tk.Label(master, text=" Width   ").grid(row=2)
      
    entry1 = tk.Entry(master)
    entry2 = tk.Entry(master)
   
    entry1.grid(row=1, column=1)
    entry2.grid(row=2, column=1)
    but1 = Button(master,text="   Save   ", command=lambda:saveDimsRect(canvas)).grid(row=3, columnspan=2)
    
    if current is not None and current.slice_type == "V" and current.height!=0:
        entry1.insert(0,current.height)
    if current is not None and current.slice_type == "H" and current.width!=0:
        entry2.insert(0,current.width)
        
    if current is not None and current.slice_type == 'L' and current.height!=0 and current.width!=0:
        entry2.insert(0,current.width)
        entry1.insert(0,current.height)
    
    master.mainloop()
    
def error(h_val,w_val,rr):
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
            but2 = Button(box1,text="Okay", command=box1.destroy).grid(row=1)
            box1.mainloop() 
            
        if h_val != 0:
            box2 = tk.Tk()
            if ch == 1:
                tk.Label(box2, text="The height should be less than "+str(h_val)).grid(row=0)
            if ch == 2:
                tk.Label(box2, text="The height should be less than or equal to "+str(h_val)).grid(row=0)
            but2 = Button(box2,text="Okay", command=box2.destroy).grid(row=1)
            box2.mainloop()

def disp(event,length,breadth):
    global ent1
    global ent2
    ent1.insert(0,length)
    ent2.insert(0,breadth)

def remove(event,length,breadth):
    global ent1
    global ent2
    ent1.delete(0,"end")
    ent2.delete(0,"end")
    
def saveDimsRect(canvas):
    global rootnode
    global current
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
            but2 = Button(box3,text="Okay", command=box3.destroy).grid(row=1)
            box3.mainloop()
            quit()
        error(0,0,0)
        en1.insert(0,v1)
        en2.insert(0,v2)
        fig1 = canvas.create_rectangle(10,10,(v2*scale)+10,(v1*scale)+10, fill="snow2")
        canvas.tag_bind(fig1, "<Enter>", lambda event, arg1 = v1, arg2 = v2: disp(event,arg1,arg2))        
        canvas.tag_bind(fig1, "<Leave>", lambda event, arg1 = v1, arg2 = v2: remove(event,arg1,arg2))        
    
        rootnode = treenode(None,None,None,v1,v2,None,10,10,(v2*scale)+10,(v1*scale)+10)
        current = rootnode
        canvas.tag_bind(fig1,"<Button-3>",do_popup)
    
    elif current.slice_type == 'H':        
        rem = 0
        u_v1 = float(entry1.get())
        u_v2 = float(entry2.get())
        
        if u_v1 >= current.height and current.height!=0:
            rem = 1
        error(current.height,0,rem)
        
        temp1 = treenode(current,None,None,u_v1,u_v2,None,current.d1,current.d2,current.d3,current.d2+(u_v1*scale))
        current.left = temp1
        fig1 = canvas.create_rectangle(temp1.d1,temp1.d2,temp1.d3,temp1.d4,fill="snow2")
        canvas.tag_bind(fig1, "<Enter>", lambda event, arg1 = u_v1, arg2 = u_v2: disp(event,arg1,arg2))        
        canvas.tag_bind(fig1, "<Leave>", lambda event, arg1 = u_v1, arg2 = u_v2: remove(event,arg1,arg2))  
        
        l_v1 = current.height-u_v1
        l_v2 = u_v2
        temp2 = treenode(current,None,None,l_v1,l_v2,None,current.d1,current.d2+(u_v1*scale),current.d3,current.d4)
        current.right = temp2            
        fig2 = canvas.create_rectangle(temp2.d1,temp2.d2,temp2.d3,temp2.d4,fill="snow2")
        canvas.tag_bind(fig2, "<Enter>", lambda event, arg1 = l_v1, arg2 = l_v2: disp(event,arg1,arg2))        
        canvas.tag_bind(fig2, "<Leave>", lambda event, arg1 = l_v1, arg2 = l_v2: remove(event,arg1,arg2))  
        
        h_line = canvas.create_line(temp2.d1,temp2.d2,temp1.d3,temp1.d4)
        canvas.tag_bind(fig1,"<Button-1>",lambda event, arg1 = temp1: shadeRectangle1(event,arg1))
        canvas.tag_bind(fig2,"<Button-1>",lambda event, arg2 = temp2: shadeRectangle2(event,arg2))
    
    elif current.slice_type == 'V':
        rem = 0
        l_v1 = float(entry1.get())
        l_v2 = float(entry2.get())

        if l_v2 >= current.width and current.width!=0:
            rem = 1
        error(0,current.width,rem)
        
        temp1 = treenode(current,None,None,l_v1,l_v2,None,current.d1,current.d2,(l_v2*scale)+current.d1,current.d4)
        current.left = temp1
        fig1 = canvas.create_rectangle(temp1.d1,temp1.d2,temp1.d3,temp1.d4,fill="snow2")
        canvas.tag_bind(fig1, "<Enter>", lambda event, arg1 = l_v1, arg2 = l_v2: disp(event,arg1,arg2))        
        canvas.tag_bind(fig1, "<Leave>", lambda event, arg1 = l_v1, arg2 = l_v2: remove(event,arg1,arg2)) 
        
        r_v1 = l_v1
        r_v2 = current.width-l_v2
        temp2 = treenode(current,None,None,r_v1,r_v2,None,current.d1+(l_v2*scale),current.d2,current.d3,current.d4)
        current.right = temp2            
        fig2 = canvas.create_rectangle(temp2.d1,temp2.d2,temp2.d3,temp2.d4,fill="snow2")
        canvas.tag_bind(fig2, "<Enter>", lambda event, arg1 = r_v1, arg2 = r_v2: disp(event,arg1,arg2))        
        canvas.tag_bind(fig2, "<Leave>", lambda event, arg1 = r_v1, arg2 = r_v2: remove(event,arg1,arg2)) 
        
        v_line = canvas.create_line(temp2.d1,temp2.d2,temp1.d3,temp1.d4)
        canvas.tag_bind(fig1,"<Button-1>",lambda event, arg1 = temp1: shadeRectangle1(event,arg1))
        canvas.tag_bind(fig2,"<Button-1>",lambda event, arg2 = temp2: shadeRectangle2(event,arg2))
    
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
        error(hh,ww,rem)
        
        global var1
        global var2
        global var3
        global var4
        global rooms
        global starty
        if v1==0 or v2==0:
            shade4 = canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4, fill='white')
            
        elif v1 != current.height and v1!=0 and current.height!=0:
            getVerInfo("top","bottom","for the block")
            if var1 == 0 and var2 == 1:
                fig3 = canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4-((current.height*scale)-(v1*scale)), fill='snow3')
                fig4 = canvas.create_rectangle(current.d1,current.d2+(v1*scale),current.d3,current.d4, fill='white')
                current.d4 = current.d4-((current.height*scale)-(v1*scale))
                canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                print("Yes")
                
            elif var1 == 1 and var2 == 0:
                fig3 = canvas.create_rectangle(current.d1,current.d2+((current.height*scale)-(v1*scale)),current.d3,current.d4, fill='snow3')
                fig4 = canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4-(v1*scale), fill='white')
                current.d2 = current.d2+((current.height*scale)-(v1*scale))
                canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                print("Yes")
        
        elif v2!=current.width and v2!=0 and current.width!=0:
            getHorInfo("left","right","for the block")
            if var3 == 0 and var4 == 1:
                fig3 = canvas.create_rectangle(current.d1,current.d2,current.d3-((current.width*scale)-(v2*scale)),current.d4, fill='snow3')
                fig4 = canvas.create_rectangle(current.d1+(v2*scale),current.d2,current.d3,current.d4, fill='white')
                current.d3 = current.d3-((current.width*scale)-(v2*scale))
                canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                print("Yes")
                
            elif var3 == 1 and var4 == 0:
                fig3 = canvas.create_rectangle(current.d1+((current.width*scale)-(v2*scale)),current.d2,current.d3,current.d4, fill='snow3')
                fig4 = canvas.create_rectangle(current.d1,current.d2,current.d3-(v2*scale),current.d4, fill='white')
                current.d1 = current.d1+((current.width*scale)-(v2*scale))
                canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
                print("Yes")
        
        elif v1 == current.height and v2 == current.width:
            shade3 = canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4, fill='snow3')
            canvas.create_text((current.d1+current.d3)/2,(current.d2+current.d4)/2, fill = 'black', font="Times 14 bold",text='R'+str(rooms))
        
        if v1!=0 and v2!=0:
            txt = 'R'+str(rooms)+'  H = '+str(v1)+'  W = '+str(v2)
            canvas2.create_text(startx+160,starty+15,fill="black",font="Times 14 italic bold", text=txt)
            canvas2.create_line(0,starty+30,340,starty+30)
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
    
def shadeRectangle1(event,temp):
    global current
    shade1 = canvas.create_rectangle(temp.d1,temp.d2,temp.d3,temp.d4, fill='grey')
    canvas.tag_bind(shade1, "<Enter>", lambda event, arg1 = temp.height, arg2 = temp.width: disp(event,arg1,arg2))        
    canvas.tag_bind(shade1, "<Leave>", lambda event, arg1 = temp.height, arg2 = temp.width: remove(event,arg1,arg2))
    current = temp
    canvas.tag_bind(shade1,"<Button-3>",do_popup)
        
def shadeRectangle2(event,temp):
    global current
    shade2 = canvas.create_rectangle(temp.d1,temp.d2,temp.d3,temp.d4, fill='grey')
    canvas.tag_bind(shade2, "<Enter>", lambda event, arg1 = temp.height, arg2 = temp.width: disp(event,arg1,arg2))        
    canvas.tag_bind(shade2, "<Leave>", lambda event, arg1 = temp.height, arg2 = temp.width: remove(event,arg1,arg2))
    current = temp
    canvas.tag_bind(shade2,"<Button-3>",do_popup)
    
def do_popup(event):
    try:
        popup.tk_popup(event.x_root, event.y_root, 0)
    finally:
        popup.grab_release()
        
def addH():
    global type
    type = " Upper Block of the Dissection "
    current.slice_type = 'H'
    start(canvas)
 
def addV():
    global type
    type = " Left Block of the Dissection "
    current.slice_type = 'V'
    start(canvas)       
    
def addLeaf():
    global current
    global starty
    global rooms
    global e1
    global e2
    global type
    global cir
    #global master4
    current.slice_type = 'L'
    if ch == 1:
        shade3 = canvas.create_rectangle(current.d1,current.d2,current.d3,current.d4, fill='snow3')
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
    
    elif ch == 2:
        type = " room "
        start(canvas)
    return

def getVerInfo(a,b,c):
    global var1
    global var2
    global master5
    master5 = tk.Tk()
    tk.Label(master5, text=" Specify the location of Vacant Space "+c).grid(row=0,columnspan=2)
    butt1 = Button(master5,text=" At the "+a,command = lambda:[master5.destroy(),setVal1(),master5.quit()]).grid(row=1, columnspan=2)
    butt2 = Button(master5,text=" At the "+b,command = lambda:[master5.destroy(),setVal2(),master5.quit()]).grid(row=2, columnspan=2)
    master5.mainloop()
    return

def getHorInfo(a,b,c):
    global var3
    global var4
    global master6 
    master6 = tk.Tk()
    tk.Label(master6, text=" Specify the location of Vacant Space "+c).grid(row=0,columnspan=2)
    butt1 = Button(master6,text=" At the "+a,command = lambda:[master6.destroy(),setVal3(),master6.quit()]).grid(row=1, columnspan=2)
    butt2 = Button(master6,text=" At the "+b,command = lambda:[master6.destroy(),setVal4(),master6.quit()]).grid(row=2, columnspan=2)
    master6.mainloop()
    return

def setVal1():
    global var1
    global var2
    var1 = 1
    var2 = 0

def setVal2():
    global var1
    global var2
    var1 = 0
    var2 = 1

def setVal3():
    global var3
    global var4
    var3 = 1
    var4 = 0

def setVal4():
    global var3
    global var4
    var3 = 0
    var4 = 1

def submit():
    #master2 = tk.Tk()
    if ch == 1:
        leftmost = rootnode
        '''while leftmost.slice_type != 'L':
            leftmost = leftmost.left
        print(leftmost.height, leftmost.width)
        sibling = leftmost.parent.right
        print(sibling.height, sibling.width)
        if sibling.slice_type == 'L':
            circulate_entry(leftmost,sibling)
        else:
            circulate(sibling.right, sibling.left)
        while leftmost.parent != rootnode:
            leftmost = leftmost.parent
            if leftmost.right.slice_type != 'L':
                circulate(leftmost.right,leftmost.left)
                
        rightmost = rootnode
        while rightmost.slice_type != 'L':
            rightmost = rightmost.right
        print(rightmost.height, rightmost.width)
        sibling = rightmost.parent.left
        if sibling.slice_type == 'L':
            circulate(rightmost,sibling)
        else:
            circulate(sibling.right,sibling.left)
        while rightmost.parent != rootnode:
            rightmost = rightmost.parent
            if rightmost.left.slice_type != 'L':
                circulate(rightmost.right,rightmost.left)
        
        for i in range(1,len(cir)):
            sp1 = canvas.create_rectangle(cir[i],fill="white")'''
        
        # Adjacency matrix Generation
        mat = np.zeros([rooms-1,rooms-1])

        for j in range(0,len(leaves)):
            leaf1 = leaves[j]

            for i in range(0,len(leaves)):
                leaf2 = leaves[i]
                if not i == j:
                    if (leaf1.d1 == leaf2.d1 and leaf1.d4 == leaf2.d2) or (leaf1.d3 == leaf2.d1 and leaf1.d4 == leaf2.d4) or (leaf1.d3 == leaf2.d1 and leaf1.d2 == leaf2.d2) or (leaf1.d3 == leaf2.d3 and leaf1.d4 == leaf2.d2) or ((leaf1.d1 <= leaf2.d1 and leaf1.d3 >= leaf2.d1) and leaf1.d4 == leaf2.d2) or ((leaf2.d1 <= leaf1.d1 and leaf2.d3 >= leaf1.d1) and leaf1.d4 == leaf2.d2) or ((leaf1.d2 <= leaf2.d2 and leaf1.d4 >= leaf2.d2) and leaf1.d3 == leaf2.d1) or ((leaf2.d2 <= leaf1.d2 and leaf2.d4 >= leaf1.d2) and leaf1.d3 == leaf2.d1):
                        mat[i,j] = 1
                        mat[j,i] = 1
        print(mat)


    #tk.Label(master2, text="     ").grid(row=0,rowspan=2,columnspan=2)
    #but2 = Button(master2,text="   Save   ", command=master2.destroy).grid(row=2, columnspan=2)
    #master2.mainloop()
    
def circulate(right_sib, left_sib):
    space = []
    if right_sib.parent.slice_type == 'V' and right_sib.parent.parent.right == right_sib.parent :
        space = [left_sib.d1, left_sib.d2, left_sib.d3, left_sib.d2+(1*scale)]
        cir.append(space)
    elif right_sib.parent.slice_type == 'H' and right_sib.parent.parent.right == right_sib.parent:
        space = [left_sib.d1, left_sib.d2, left_sib.d1+(1*scale), left_sib.d4]
        cir.append(space)
    elif right_sib.parent.slice_type == 'V' and right_sib.parent.parent.left == right_sib.parent:
        space = [left_sib.d1, left_sib.d4-(1*scale), left_sib.d3, left_sib.d4]
        cir.append(space)
    elif right_sib.parent.slice_type == 'H' and right_sib.parent.parent.left == right_sib.parent:
        space = [left_sib.d3-(1*scale), left_sib.d2, left_sib.d3, left_sib.d4]
        cir.append(space)
        
def circulate_entry(leftmost,sibling):
    space1 = []
    space2 = []
    if sibling.parent.slice_type == 'V' and sibling.parent != rootnode and sibling.parent.parent.slice_type == 'H':
        if sibling.width >= leftmost.width:
            space1 = [sibling.d1, sibling.d2, sibling.d1+(1*scale), sibling.d4]
            cir.append(space1)
            
            space2 = [sibling.d1, sibling.d4-(1*scale), sibling.d3, sibling.d4]
            cir.append(space2)
            print(cir)
        else:
            space1 = [leftmost.d3-(1*scale), leftmost.d2, leftmost.d3, leftmost.d4]
            cir.append(space1)
            
            space2 = [leftmost.d1, leftmost.d4-(1*scale), leftmost.d3, leftmost.d4]
            cir.append(space2)
            print(cir)
    elif sibling.parent.slice_type == 'H' and sibling.parent != rootnode and sibling.parent.parent.slice_type == 'V':
        if sibling.height >= leftmost.height:
            space1 = [sibling.d1, sibling.d2, sibling.d3, sibling.d2+(1*scale)]
            cir.append(space1)
            
            space2 = [sibling.d3-(1*scale), sibling.d2, sibling.d3, sibling.d4]
            cir.append(space2)
            print(cir)
        else:
            space1 = [leftmost.d1, leftmost.d4-(1*scale), leftmost.d3, leftmost.d4]
            cir.append(space1)
            
            space2 = [leftmost.d3-(1*scale), leftmost.d2, leftmost.d3, leftmost.d4]
            cir.append(space2)
            print(cir)
    
def choice():
    master3 = tk.Tk()
    tk.Label(master3, text=" Select the type of Floor Plan to be generated ").grid(row=0,columnspan=2)
    but3 = Button(master3,text=" RFP without empty spaces ", command=lambda:[master3.destroy(),decideRect()]).grid(row=1, columnspan=2)
    but4 = Button(master3,text=" RFP with empty spaces ", command=lambda:[master3.destroy(),decideNonRect()]).grid(row=2, columnspan=2)
    master3.mainloop()

def decideRect():
    global ch
    ch = 1
    start(canvas)
    
def decideNonRect():
    global ch
    ch = 2
    start(canvas)
    
if __name__== '__main__':
        
    root=tk.Tk()
    entry = tk.Entry(root)
    
    current = None
    rootnode = None
    prev = None
    leftmost = None
    sibling = None
    index = 0
    s = 25
    scale = 0
    startx = 10
    starty = 60
    rooms = 1
    ch = 0
    cir = []
    leaves = []

    wd = 1000
    ht = 700
    frame = Frame(root)
    frame.grid(row=0,column=0, sticky="n")
    
    border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}
    
    canvas=tk.Canvas(root, width=wd, height=ht, background='white', **border_details)
    canvas.grid(row=1,column=0,columnspan=5)
    
    canvas2=tk.Canvas(root, width=340, height=ht, background='white',**border_details)
    canvas2.grid(row=1,column=5,columnspan=2)
    canvas2.create_text(120,30,fill='black',font="Times 16 italic bold",text=' Dimensions of Rooms ')
    canvas2.create_line(0,50,340,50)
    
    type = " Rectangular plot "
    popup = Menu(root, tearoff=0)
    popup.add_command(label="Horizontal Slice", command=addH)
    popup.add_command(label="Vertical Slice", command=addV)
    popup.add_separator()
    popup.add_command(label="Make a Room",command=addLeaf)
    
    showButton = Button(root, text=" Start a new dissection ",command=choice)
    showButton.grid(row=0, column=0)
    sub = Button(root, text=" Generate Spanning Circulation ",command=submit).grid(row=2,column=0)
    
    tk.Label(root, text=" Dimensions of Total Plot ").grid(row=2,column=2)
    tk.Label(root, text=" Height  ").grid(row=2,column=3)
    tk.Label(root, text=" Width   ").grid(row=2,column=5)
      
    en1 = tk.Entry(root)
    en2 = tk.Entry(root)
   
    en1.grid(row=2, column=4)
    en2.grid(row=2, column=6)
    
    tk.Label(root, text=" Dimensions of Current block ").grid(row=0,column=2)
    tk.Label(root, text=" Height  ").grid(row=0,column=3)
    tk.Label(root, text=" Width   ").grid(row=0,column=5)
      
    ent1 = tk.Entry(root)
    ent2 = tk.Entry(root)
   
    ent1.grid(row=0, column=4)
    ent2.grid(row=0, column=6)
        
    root.mainloop()
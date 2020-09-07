import final
import main
import tkinter as tk
from Dissected_RFP import dissected
from PIL import ImageTk,Image

colors = ["#B682F5","#2CC4B4"]
col = ["#788585","#9A8C98","#F2E9E4","#C9ADA7","#e1eaec"]
class plan:
    loader = tk.Tk()
    loader.title("Floor Plan Generator")
    loader.state('zoomed')
    h= loader.winfo_screenheight()
    w = loader.winfo_screenwidth()
    imname = "./wp.jpg"
    im1 = Image.open(imname).convert("1")
    size = (w,h )
    im2 = ImageTk.PhotoImage(Image.open(imname).resize(size))

    buta="./but.png"
    im1 = Image.open(imname).convert("1")
    size=(im1.width//8,im1.height//12)
    size1=( int(im1.width/2.5),im1.height //10)
    but=ImageTk.PhotoImage(Image.open(buta).resize(size))
    but1=ImageTk.PhotoImage(Image.open("./title.png").resize(size1))
    def __init__(self):
        self.canvas= tk.Canvas(self.loader,width=self.w,height=self.h)
        # self.canvas.create_image(self.w/2,self.h/2,image=self.im2,anchor='center')
        self.canvas.pack()
        self.canvas.create_rectangle(50,50,self.w-50,self.h-50,width=5)
        # a= self.canvas.create_image((self.w)*0.5,(self.h)*0.2,anchor='center',image=self.but1)
        self.canvas.create_text((self.w)*0.5,(self.h)*0.2,anchor='center',font=("Helvetica",40,"bold"),justify='center',text="GPLAN: Computer-Generated Dimensioned \nFloorplans for given Adjacencies")
        self.button(0.5,0.4,"Instructions",self.instructoins)
        self.button(0.3,0.6,"Draw an Adjacency Graph",self.run_GPLAN)
        self.button(0.7,0.6,"Draw a Layout",self.run_iFP)
        self.button(0.5,0.8,"Dissection method",self.run_dissected)
        self.loader.mainloop()
    
    def button(self,x,y,txt,func):
        a= self.canvas.create_image((self.w)*x,(self.h)*y,anchor='center',image=self.but)
        b= self.canvas.create_text((self.w)*x,(self.h)*y,font=("Helvetica",15,""),anchor='center',text=txt)
        self.canvas.tag_bind(a,"<Button-1>",func)
        self.canvas.tag_bind(b,"<Button-1>",func)
    def run_GPLAN(self,event):
        self.loader.destroy()
        main.run()

    def run_iFP(self,event):
        self.loader.destroy()
        final.run()

    def run_dissected(self,event):
        self.loader.destroy()
        dissected()


    def instructoins(self,event):
        tk.messagebox.showinfo("Instructions",
            "--------User Instructrions--------\n 1. Draw the input graph. \n 2. Use right mouse click to create a new room. \n 3. left click on one node then left click on another to create an edge between them. \n 4. You can give your own room names by clicking on the room name in the graph or the table on the right. \n 5. After creating a graph you can choose one of the option to create it's corresponding RFP or multiple RFPs with or without dimension. You can also get the corridor connecting all the rooms by selecting 'circultion' or click on 'RFPchecker' to check if RFP exists for the given graph. \n 6. You can also select multiple options .You can also add rooms after creating RFP and click on RFP to re-create a new RFP. \n 7.Reset button is used to clear the input graph. \n 8. Press 'Exit' if you want to close the application or Press 'Restart' if you want to restart the application")

p = plan()
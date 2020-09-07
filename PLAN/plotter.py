import networkx as nx
import matplotlib.pyplot as plt
import ptpg
import tkinter as tk
import turtle
def plot(cir,m):
    pos=nx.spring_layout(cir) # positions for all nodes
    nx.draw_networkx(cir,pos, labels=None,node_size=400 ,node_color='#4b8bc8',font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1, bbox=None, ax=None)
    nx.draw_networkx_edges(cir,pos)
    nx.draw_networkx_nodes(cir,pos,
                        nodelist=list(range(m,len(cir))),
                        node_color='r',
                        node_size=500,
                    alpha=1)
    plt.show()

def RFP_plot(spanned):
    
    G= ptpg.PTPG(spanned)

    root_window = tk.Tk()
    root_window.title('Rectangular Dual')
    root_window.geometry(str(1366) + 'x' + str(700))
    root_window.resizable(0, 0)
    root_window.grid_columnconfigure(0, weight=1, uniform=1)
    root_window.grid_rowconfigure(0, weight=1)

    border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}

    canvas = tk.Canvas(root_window, **border_details,width = 100000000000, height =2000)
    canvas.pack_propagate(0)
    canvas.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

    scroll_x = tk.Scrollbar(root_window, orient="horizontal", command=canvas.xview)
    scroll_x.grid(row=1, column=0, sticky="ew")

    scroll_y = tk.Scrollbar(root_window, orient="vertical", command=canvas.yview)
    scroll_y.grid(row=0, column=1, sticky="ns")

    canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
    canvas.configure(scrollregion=canvas.bbox("all"))

    pen = turtle.RawTurtle(canvas)
    pen.speed(100)

    G.create_single_dual(1,pen)

    # screenshot = pyautogui.screenshot()
    # screenshot.save("screen.png")

    root_window.mainloop()


import tkinter as tk
import turtle

def create_pen():
    # root = tk.Tk()
    root_window = tk.Tk()
    root_window.state('zoomed')
    # root_window = tk.Frame(root,width=1000,height=700)
    l1 = tk.Label(root_window, text = "Rectangular dual")

    l1.grid(row=1,column=0)
    
    # root_window.geometry(str(1366) + 'x' + str(700))
    # root_window.resizable(0, 0)
    root_window.grid_columnconfigure(0, weight=1, uniform=1)
    root_window.grid_rowconfigure(0, weight=1)
    border_details = {'highlightbackground': 'black', 'highlightcolor': 'black', 'highlightthickness': 1}

    canvas = tk.Canvas(root_window, **border_details,width = 100, height =200)
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
    # root_window.mainloop()
    return pen, root_window


if __name__ == '__main__':
    pen , root = create_pen()
    root.mainloop()
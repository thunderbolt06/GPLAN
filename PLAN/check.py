import networkx as nx
import tests
def print(string):
    box.insert('end',string)
    box.insert('end',"\n")
def checker(value,textbox):
    global box
    box = textbox
    print("Edge set")
    # textbox.insert('end',"Enter the edge set")
    edgeset = []
    edgeset=value[2]
    # for i in range(int(input("No. of Edges: "))):
        # edgeset.append(list(map(int, input().split())))
    g=nx.Graph()
    g.add_edges_from(edgeset)
    print(g.edges)
    # textbox.insert('end',g.edges)
    # textbox.insert('end',"\n")
    tests.tester(g,textbox)





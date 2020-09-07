import os
import matplotlib.pyplot as plt
import networkx as nx
import warnings
warnings.simplefilter("ignore")
# from generate import generation_next


def plot_graphs(init_len, graph_list):
    folder_path = f"RFP_Graph_Plots/Len {init_len}"
    os.makedirs(folder_path, exist_ok=True)
    graph_no = 1
    # _fig, _ax = plt.subplots()
    for graph in graph_list:
        plt.figure(graph_no)
        try:
            nx.draw_planar(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
        except:
            nx.draw(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
        # Inner Graph IG_<Initial Vertices>_<Graph_Size>_<Graph No.>.png
        plt.savefig(f'{folder_path}/graph_{init_len}_{graph.size()}_{graph_no}.png')
        plt.show()
        
        graph_no += 1
        plt.clf()


# g=nx.path_graph(5)
# plot_graphs(len(g.nodes),[g])

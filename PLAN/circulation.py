import networkx as nx
import matplotlib.pyplot as plt
from plotter import plot
def span(graph):
    n=len(graph)
    m = n

    # nx.draw_spring(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
    # plt.show()
    print("choose a door")
    k ,j = map(int, input().split())
    n=dfs(k,j,graph,n,-1,m)
    return graph

def dfs(i,j,graph,n,br,m):
    
    for ne in nx.common_neighbors(graph,i,j):
        if ne < m :
            graph.add_edges_from([(n,i),(n,j)])
            graph.remove_edge(i,j)
            graph.add_edge(n,ne)
            if br>0:
                graph.add_edge(n,br)
            # plot(graph,m)                  #can remove this line if not required to plot
            n+=1
            prev=n-1
            n=dfs( ne,i,graph,n,n-1,m)
            n=dfs( ne,j,graph,n,prev,m)

    return n
            

def BFS(graph,e1,e2):
    # nx.draw_spring(graph, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=1.0, bbox=None, ax=None)
    # plt.show()

    n = len(graph)
    m = n
    s = (e1-1 ,e2-1 , -1)

    # print("choose a door")
    # i ,j = map(int, input().split())
    # s[0] = i
    # s[1] = j
    queue = []
    queue.append(s)

    while ( queue ):
        s = queue.pop(0)
        for ne in nx.common_neighbors(graph,s[0],s[1]):
            if ne < m :
                graph.add_edge(s[0],n)
                graph.add_edge(s[1],n)
                graph.remove_edge(s[0],s[1])
                if s[2]>0:
                    graph.add_edge(n,s[2])
                graph.add_edge(n,ne)
                # plot(graph,m)  
                n+=1
                queue.append((ne,s[0],n-1))
                queue.append((ne,s[1],n-1))
    return graph
    


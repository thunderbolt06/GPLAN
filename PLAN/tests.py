import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import matplotlib.pyplot as plt
from plot_graphs import plot_graphs
from NESW import num_cips
def print(string):
    box.insert('end',string)
    box.insert('end',"\n")
def tester(graph,textbox):
    global box
    box = textbox
    check= True

    if not nx.is_biconnected(graph):
        print("Not Biconnected")
        return
    if not planarity_check(graph,textbox):
        check = False
    
    elif not complex_triangle_check(graph,textbox):
        check = False

    else:
        if not cip_rule_check(graph,textbox):
            check = False
        
        # if  not civ_rule_check(graph,textbox):
        #     check=False
        #     textbox.insert('end',"\n")
            # print("=> civ rule failed\n")

    if check:
        print("=> RFP exists\n")
        # plot_graphs(len(graph),[graph])
        # plot_graph([graph])
    else:
        print("=> RFP doesn't exist\n")
        # plot_graphs(len(graph),[graph])

def planarity_check(given_graph,textbox):
    global box
    box = textbox
    # Planarity Check
    if not nx.check_planarity(given_graph)[0]:
        print("=> The graph is non-planar\n")
        return 0
    return 1

def cip_rule_check(graph,textbox):
    global box
    box = textbox
    """
    Given A Graph
    Returns true if cip rule is satisfied

    CIP Rule = 2 CIP on outer biconnected components and no CIP on
    inner biconnected components
    """
    cip_check = True
    outer_comps, inner_comps, single_component = component_break(graph,textbox)
    if not single_component:
        # CIP Rule for Outer Components
        if len(outer_comps)>2:
            print("BNG is not a path graph\n")
            return False
        for comp in outer_comps:
            print(f"Checking biconnected component {list(comp)}")
            if num_cips(comp) > 2 :
                cip_check = False
                print(f"    Num cips ={num_cips(comp)}\n")
                print(f"    Maximum possible cip =2\n")
                print('Invalid')
            else:
                print('Valid')
        # CIP Rule for Inner Components
        for comp in inner_comps:
            print(f"Checking biconnected component {list(comp)}")
            if num_cips(comp) > 0 :
                cip_check = False
                print(f"    Num cips ={num_cips(comp)}\n")
                print(f"    Maximum possible cip =0\n")
                print('Invalid')
            else:
                print('Valid')
    else:
        # CIP Rule for single_component Components
        print(f"Checking biconnected component {list(single_component)}")
        if num_cips(single_component) > 4:
            cip_check = False
            print(f"    Num cips ={num_cips(single_component)}")
            print(f"    Maximum possible cip =4\n")
            print('Invalid')
        else:
            print('Valid')
    if not cip_check:
        print("=> cip rule failed\n")
    return cip_check

def component_break(given_graph,textbox):
    global box
    box = textbox
    """
    Given a graph,
    returns [list of the 2 outer components(1 articulation point) with 2cip],
    [list of other inner components(2 articulation points) with 0 cip]
    """
    test_graph = given_graph.copy()
    cutvertices = list(nx.articulation_points(test_graph))
    inner_components = []
    outer_components = []
    if len(cutvertices) == 0:
        single_component = test_graph
        return 0, 0, single_component
    for peice_edges in nx.biconnected_component_edges(test_graph):
        peice = nx.Graph()
        peice.add_edges_from(list(peice_edges))
        num_cutverts = 0
        for cutvert in cutvertices:
            if cutvert in peice.nodes():
                num_cutverts += 1

        if num_cutverts == 2:
            inner_components.append(peice)
        elif num_cutverts == 1:
            outer_components.append(peice)
        else:
            print("Not a PTPG\n")
            print(test_graph.edges())

        if len(outer_components) > 2:
            # print("BNG is not a path\n")
            print(test_graph)
    return outer_components, inner_components, 0

def complex_triangle_check(graph,textbox):
    global box
    box = textbox
    for compedges in nx.biconnected_component_edges(graph):
        comp=nx.Graph()
        comp.add_edges_from(compedges)
        H = comp.to_directed()
        all_cycles = list(nx.simple_cycles(H))
        all_triangles = []
        for cycle in all_cycles:
            if len(cycle) == 3:
                all_triangles.append(cycle)

        if (comp.size() - len(comp) +1) > (len(all_triangles)/2):
            print("=> Not triangled\n")
            return False
        elif (comp.size() - len(comp) +1) < (len(all_triangles)/2):
            print("=> complex triangle exists\n")
            return False
    return True

def civfinder(g,textbox):
    global box
    box = textbox
    
    # nx.draw_planar(g,labels=None)
    # print(g.edges)
    corner_set = []
    for ver in g.nodes:
        if nx.degree(g,ver)==2:
            corner_set.append(ver)
        else:
            if nx.degree(g,ver)==1:
                corner_set.append(ver)
                corner_set.append(ver)
    # print(f"    corner set = {corner_set}\n")
    return corner_set
def civ_rule_check(graph,textbox):
    global box
    box = textbox
    civ_check = True
    outer_comps, inner_comps, single_component = component_break(graph,textbox)
    # list, list, element
    # print(outer_comps, inner_comps, single_component)

    if not single_component:
        if len(outer_comps) >2:
            print("BNG is not a path graph\n")
            return False
        cut= nx.articulation_points(graph)
        cut= set(cut)
        # CIP Rule for Outer Components
        for comp in outer_comps:
            cut1= []
            for i in cut:
                if i in comp:
                    cut1.append(i)
            civ_check= eachcompciv(comp,cut1,2,textbox)
        # CIP Rule for Inner Components
        for comp in inner_comps:
            cut1= []
            for i in cut:
                if i in comp:
                    cut1.append(i)
            civ_check= eachcompciv(comp,cut1,0,textbox)
    else:
        # CIP Rule for single_component Components
        cut = []
        civ_check= eachcompciv(single_component,cut,4,textbox)
    return civ_check

def eachcompciv(graph,cut,maxciv,textbox):
    global box
    box = textbox
    # print(f"checking component {list(graph)}\n")
    set1= set(cut)
    # print(f"    cut set = {list(set1)}\n")
    y= [ i for i in list(civfinder(graph,textbox)) if i not in set1]
    x=len(y)
        # print(f"    num civs ={x}\n")
        # print(f"    maximum possible civ ={maxciv}\n")
    if x>maxciv :
        # print("    component failed\n")
        return False
        # print("    true\n")
    return True

# g=nx.Graph()
# g.add_edges_from([(0, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 5), (3, 4), (4, 5)]) #wrapped vertex attached
# g.add_edges_from([(0, 1), (1, 2),  (2, 3), (0,3),(1,3),(0,2)]) #complex triangle
# g.add_edges_from([(0, 1), (1, 2),  (2, 3), (3,4),(1,3)]) #cip rule
# g.add_edges_from([(0, 1), (1, 2),  (2, 3), (3,0),(2,5),(2,4),(3,5),(3,4),(4,5)]) #1 non-triangle with 1 ST

# g.add_edges_from([(0, 1), (1, 2), (0,2),(3,4),(4,5),(5,3),(0,3),(0,4),(0,5),(2,4),(1,4),(2,5)]) #k4 subgraph
# g.add_edges_from([(0, 1), (0,3),(1,2),(2,3),(1,3),(0,4),(4,1),(1,5),(2,5),(2,6),(3,6),(3,7),(0,7),(3,8),(6,8),(2,9),(6,9)]) #5 cips
# g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 4),
#  (1, 5), (2, 3), (2, 5), (3, 4), (3, 5), (4, 5)])#complex without K4 .. 2 lines
# g.add_edges_from([(0, 1), (1, 2),(1,3),(1,4)])
# RFPchecker(g)





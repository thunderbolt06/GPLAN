import networkx as nx
import warnings
import numpy as np
from NESW import num_cips

def gui_checker(matrix):
	# matrix = np.zeros((vertices, vertices), int)
	# for i in (edges):
	# 	matrix[i[0]][i[1]] = 1
	# 	matrix[i[1]][i[0]] = 1
	planarity = check_planarity(matrix)
	triangularity = check_triangularity(matrix)
	biconnectivity = isBiconnected(matrix)
	return planarity,triangularity,biconnectivity

def check_planarity(adjacencymatrix):
	graph = nx.from_numpy_matrix(adjacencymatrix)
	return nx.check_planarity(graph)[0]


def check_triangularity(adjacencymatrix):
	graph = nx.from_numpy_matrix(adjacencymatrix)
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
			print("Not triangled")
			return False
	return True

def isBiconnected(adjacencymatrix):
		h = nx.from_numpy_matrix(adjacencymatrix)
		return nx.is_biconnected(h)


def rfp_checker(adjacencymatrix):
	graph = nx.from_numpy_matrix(adjacencymatrix)
	check= True
	if not complex_triangle_check(graph):
	    check = False
	elif not cip_rule_check(graph):
	    check = False
	return check



def cip_rule_check(graph):
    """
    Given A Graph
    Returns true if cip rule is satisfied

    CIP Rule = 2 CIP on outer biconnected components and no CIP on
    inner biconnected components
    """
    cip_check = True
    outer_comps, inner_comps, single_component = component_break(graph)
    if not single_component:
        # CIP Rule for Outer Components
        if len(outer_comps)>2:
            return False
        for comp in outer_comps:
            if num_cips(comp) > 2 :
                cip_check = False
        # CIP Rule for Inner Components
        for comp in inner_comps:
            if num_cips(comp) > 0 :
                cip_check = False
    else:
        # CIP Rule for single_component Components
        if num_cips(single_component) > 4:
            cip_check = False
    return cip_check

def component_break(given_graph):
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
    return outer_components, inner_components, 0

def complex_triangle_check(graph):
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
            return False
        elif (comp.size() - len(comp) +1) < (len(all_triangles)/2):
            return False
    return True

def civfinder(g):
    
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
def civ_rule_check(graph):
    civ_check = True
    outer_comps, inner_comps, single_component = component_break(graph)
    # list, list, element
    # print(outer_comps, inner_comps, single_component)

    if not single_component:
        if len(outer_comps) >2:
            return False
        cut= nx.articulation_points(graph)
        cut= set(cut)
        # CIP Rule for Outer Components
        for comp in outer_comps:
            cut1= []
            for i in cut:
                if i in comp:
                    cut1.append(i)
            civ_check= eachcompciv(comp,cut1,2)
        # CIP Rule for Inner Components
        for comp in inner_comps:
            cut1= []
            for i in cut:
                if i in comp:
                    cut1.append(i)
            civ_check= eachcompciv(comp,cut1,0)
    else:
        # CIP Rule for single_component Components
        cut = []
        civ_check= eachcompciv(single_component,cut,4)
    return civ_check

def eachcompciv(graph,cut,maxciv):
    # print(f"checking component {list(graph)}\n")
    set1= set(cut)
    # print(f"    cut set = {list(set1)}\n")
    y= [ i for i in list(civfinder(graph)) if i not in set1]
    x=len(y)
        # print(f"    num civs ={x}\n")
        # print(f"    maximum possible civ ={maxciv}\n")
    if x>maxciv :
        # print("    component failed\n")
        return False
        # print("    true\n")
    return True



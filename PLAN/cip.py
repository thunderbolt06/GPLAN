import networkx as nx
import warnings
import numpy as np
import operations as opr
import shortcutresolver as sr

def find_cip(graph):
	ordered_boundary = opr.ordered_outer_boundary(graph)
	shortcuts = sr.get_shortcut(graph)
	shortcut_endpoints = []
	# print(shortcuts)
	for shortcut in shortcuts:
		shortcut_endpoints.append(shortcut[0])
		shortcut_endpoints.append(shortcut[1])
	cip = []
	for shortcut in shortcuts:
		pos_1 = ordered_boundary.index(shortcut[0])
		pos_2 = ordered_boundary.index(shortcut[1])
		if(pos_1 > pos_2):
			temp = pos_1
			pos_1 = pos_2
			pos_2 = temp
			temp1 = shortcut[0]
			shortcut[0] = shortcut[1]
			shortcut[1] = temp1
		path_1 = ordered_boundary[pos_1+1:pos_2]
		path_2 = ordered_boundary[pos_2+1:len(ordered_boundary)]
		path_2 = path_2 + ordered_boundary[0:pos_1]
		# print(path_1)
		# print(path_2)
		path_1_cip = 1
		path_2_cip = 1
		for i in path_1:
			if i in shortcut_endpoints:
				path_1_cip = 0
				break
		for i in path_2:
			if i in shortcut_endpoints:
				path_2_cip = 0
				break
		if(path_1_cip == 1):
			path_1.insert(0,shortcut[0])
			path_1.append(shortcut[1])
			cip.append(path_1)
		if(path_2_cip == 1):
			path_2.insert(0,shortcut[1])
			path_2.append(shortcut[0])
			cip.append(path_2)

	return cip

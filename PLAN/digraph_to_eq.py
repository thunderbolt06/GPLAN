from Convert_adj_equ import Convert_adj_equ
import numpy as np
import scipy.optimize
from solve_linear import solve_linear

# global N,f_VER, A_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, ar_max, ar_min

def digraph_to_eq(VER,HOR,inp_min,inp_height):
		
	#VER=[[0,1,1,0,0,0,1,0,0,1,1,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

	#HOR=[[0,1,0,1,0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0]]


	# digraph_to_eq(VER,HOR)
	N=len(VER)

	[f_VER,A_VER,Aeq_VER,Beq_VER]=Convert_adj_equ(VER,5)
	[f_HOR,A_HOR,Aeq_HOR,Beq_HOR]=Convert_adj_equ(HOR,5)

	# print(len(f_VER[0]))
	# print(len(A_VER[0]))

	# inp_min=[int(x) for x in input("Enter the minimum width of room: ").strip().split()]

	# inp_height=[int(x) for x in input("Enter the minimum height of each room: ").strip().split()]

	#ar_pmt = int(input("Enter '0' to proceed with default AR_Range (0.5 to 2) or '1' to enter custom range: "))

	#if ar_pmt == 0:
	#	ar_min = np.dot(np.ones([N,1],int),0.5)
	#	ar_max = np.dot(np.ones([N,1],int),2)
	#else:
	#	ar_min=[int(x) for x in input("Enter the minimum Aspect Ratio of room: ").strip().split()]

	#	ar_max=[int(x) for x in input("Enter the maximum Aspect Ratio of room: ").strip().split()]

	#inp_height = np.array(inp_area) / np.array(inp_min)
	#print(inp_height)
	#size_inp_min=len(inp_min)

	b_VER=np.dot(np.array(inp_min),-1)
	b_VER = np.transpose(b_VER)
	b_VER = b_VER.astype(float)

	dimensions = solve_linear(N,f_VER, A_VER, b_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, inp_height)

	# print('Height = ',dimensions[1])
	# print('\n Width = ',dimensions[0])

	return [dimensions[0],dimensions[1]]
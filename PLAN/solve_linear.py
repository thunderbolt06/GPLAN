import numpy as np
import scipy.optimize

# global N,f_VER, A_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, ar_max, ar_min

def solve_linear(N,f_VER, A_VER, b_VER, Aeq_VER, Beq_VER, f_HOR, A_HOR, Aeq_HOR, Beq_HOR, inp_height):

	# print(Aeq_VER)
	value_opti_ver = scipy.optimize.linprog(f_VER,A_ub=A_VER,b_ub=b_VER,A_eq=Aeq_VER,b_eq=Beq_VER, bounds=(1,None), method='interior-point', callback=None, options=None, x0=None)


	b_HOR=np.zeros([N-1,1],dtype=float)
	X1=value_opti_ver['x']
	X1 = np.array([X1])
	X1 = np.transpose(X1)
	# print(X1)

	b_HOR = np.dot(np.array(inp_height),-1)
	# print(b_HOR)

	value_opti_hor = scipy.optimize.linprog(f_HOR,A_ub=A_HOR,b_ub=b_HOR,A_eq=Aeq_HOR,b_eq=Beq_HOR, bounds=(1,None), method='interior-point', callback=None, options=None, x0=None)

	X2=value_opti_hor['x']
	X2 = np.array([X2])
	X2 = np.transpose(X2)
	W=np.dot(A_VER,X1)
	H=np.dot(A_HOR,X2)
	
	return [W, H]
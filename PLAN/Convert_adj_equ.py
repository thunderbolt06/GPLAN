import numpy as np
def Convert_adj_equ(DGPH,symm_rooms):
	N=len(DGPH)
	lineq_temp=np.zeros([N,N**2])
	# '''sys rrooms
	# code here
	# symm rooms'''

	#starting Liner equalities as a matrix
	for i in range(0,N):
		for j in range(0,N):
			if DGPH[i][j]==1:
				lineq_temp[i][N*i+j]=1
			if DGPH[j][i]==1:
				lineq_temp[i][N*j+i]= -1
	 
	#starting removing extra variables from matrix
	lineq_temp_np=np.array(lineq_temp)
	lineq_temp_np=lineq_temp_np.transpose()
	LINEQ = []
	for i in range(0,N):
		for j in range(0,N):
			if DGPH[i][j] == 1:
				LINEQ.append(lineq_temp_np[N*(i)+j])
	LINEQ=np.array(LINEQ)		
	#Starting Objective function
	LINEQ = np.transpose(LINEQ)
	
	n = len(LINEQ[0])

	f = np.zeros([1,n])

	z = np.sum(DGPH[0],dtype =int)

	for i in range(0,z):
		f[0][i] = 1
	# print(f)

	#Linear inequalities (Dimensional Constraints)
	def ismember(d, k):
		return [1 if (i == k) else 0 for i in d]

	A = []
	for i in range(0,N):
		A.append(ismember(LINEQ[i],-1))
	A = np.array(A)
	A = np.dot(A,-1)
	A = np.delete(A,0,0)
	Aeq = []
	# print(LINEQ)
	
	def any(A):
 		for i in A:
 			if i == 1:
 				return 1
 		return 0

	for i in range(0,N):
		if any(ismember(LINEQ[i],1)) != 0 and any(ismember(LINEQ[i],-1)) != 0:
			Aeq.append(LINEQ[i])
	Aeq = np.array(Aeq)
	
	Beq = np.zeros([1,len(Aeq)])

	return [f,A,Aeq,Beq]

# Convert_adj_equ([[0,1,1,1,0,0,0,0,0,0,0],
# 				 [0,0,0,0,1,0,0,0,0,0,0],
# 				 [0,0,0,0,0,0,0,1,0,0,0],
# 				 [0,0,0,0,0,1,1,1,0,0,0],
# 				 [0,0,0,0,0,0,0,0,0,0,0],
# 				 [0,0,0,0,0,0,0,0,1,0,0],
# 				 [0,0,0,0,0,0,0,0,0,1,0],
# 				 [0,0,0,0,0,0,0,0,0,1,1],
# 				 [0,0,0,0,0,0,0,0,0,0,0],
# 				 [0,0,0,0,0,0,0,0,0,0,0],
# 				 [0,0,0,0,0,0,0,0,0,0,0]],105)
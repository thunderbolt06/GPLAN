import numpy as np
from digraph_to_eq import digraph_to_eq
def floorplan_to_st(A,inp_min,inp_height):
	# A=[[1,2,6,9,10],[3,4,7,11,12],[5,5,8,13,13]]
	m=len(A)
	n=len(A[0])
	
	for i in range(0,m):
		for j in range(0,n):
			A[i][j] +=1
	
	A = np.array(A)

	len_dgph=np.amax(A)


	ver_dgph=np.zeros((len_dgph,len_dgph),int)
	north_adj=np.zeros((1,len_dgph),int)

	for i in range(0,n):
		north_adj[0][A[0][i]-1]=1
	for i in range(0,n):
		for j in range(0,m):
			if j==0:
				temp=A[j][i]
			if A[j][i]!=temp:
				ver_dgph[temp-1][A[j][i]-1]=1
				temp=A[j][i]
	VER=[]
	for i in north_adj:
		VER.append(i)

	for i in ver_dgph:
		VER = np.append(VER,[i],axis=0)

	VER = np.insert(VER,0,[0],axis=1)

	hor_dgph=np.zeros([len_dgph,len_dgph])
	west_adj=np.zeros([1,len_dgph])

	for i in range(0,m):
		west_adj[0][A[i][0]-1]=1
	for i in range(0,m):
		for j in range(0,n):
			if j==0:
				temp=A[i][j]
			if A[i][j]!=temp:
				hor_dgph[temp-1][A[i][j]-1]=1
				temp=A[i][j]
	HOR=[]

	for i in west_adj:
		HOR.append(i)


	for i in hor_dgph:
		HOR = np.append(HOR,[i],axis=0)

	HOR = np.insert(HOR,0,[0],axis=1)
	# print(HOR)

	[width,height] = digraph_to_eq(VER,HOR,inp_min,inp_height)

	return [(-1)*width,(-1)*height,hor_dgph]
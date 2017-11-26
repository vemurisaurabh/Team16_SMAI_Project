from __future__ import division
import numpy as np
import scipy
import random
import numpy as np
import numpy.matlib
from math import e, sqrt, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from generateexpo import generateExponential

def decideAorB(data,datasize,alphaA,betaA,alphaB,betaB):
	# This returns 0 for A and 1 for B
	decision=np.zeros(datasize)
	lognumerator = alphaA*np.log(betaA+data)
	logdenominator = alphaB*np.log(betaB+data)
	
	diff=lognumerator-logdenominator
	
	for i in range(0,datasize-1):
		if (diff[i] >=0):
			decision[i]=0
		else :
			decision[i]=1
	
	return decision

Nvals=np.array([100,500,1000,10000])
l1=len(Nvals)

diffvals=np.arange(1,50,1)#np.array([1,2,3,4,5,6,7,8])

l2=len(diffvals)

plt.figure()
#ax=fig.gca(projection='3d')

accvals =np.zeros(np.size(diffvals))
############################################################################
i=-1
j=-1
k=-1

lamb0A=1
lamb0B=8

# Class A parameters 

partitionSize=0.1

for i in range(0,np.size(Nvals)):
	for j in range(0,np.size(diffvals)):
		k=k+1
		N = Nvals[i]

		###################

		# Class B parameters 
		lamb0B=lamb0A+diffvals[j]
		##################

		traindataA, trainNA =generateExponential(lamb0A,N,partitionSize)
		traindataB, trainNB =generateExponential(lamb0B,N,partitionSize)

		alphaA=trainNA+1
		betaA=np.sum(traindataA)+lamb0A

		alphaB=trainNB+1
		betaB=np.sum(traindataB)+lamb0A


		testdataA, testNA =generateExponential(lamb0A,N,partitionSize)
		testdataB, testNB =generateExponential(lamb0B,N,partitionSize)

		temp1=decideAorB(data=testdataA,datasize=testNA,alphaA=alphaA,betaA=betaA,alphaB=alphaB,betaB=betaB)
		temp2=decideAorB(data=testdataB,datasize=testNB,alphaA=alphaA,betaA=betaA,alphaB=alphaB,betaB=betaB)

		tp=np.sum(temp2==1)
		tn=np.sum(temp1==0)
		fp=np.sum(temp1==1)
		fn=np.sum(temp2==0)

		acc=100*(tp+tn)/(tp+tn+fp+fn)
		accvals[k]=acc
	plt.plot(diffvals,accvals,label='N = %s'%(Nvals[i]))
	plt.xlabel('Difference in Lambda0A and Lambda0B')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.hold('true')
	k=-1

#for i in range(0,np.size(Nvals)):
#	string= "N ="+str(Nvals[i])
#	plt.legend(a[i],string)


# xx, yy= np.meshgrid(sig0vals,Nvals)
# a=np.matlib.repmat(Nvals,1,l2)
# b=np.matlib.repmat(sig0vals,1,l1)



plt.show()

#print acc

#print finalsigma2A
#print finalsigma2B

#plt.figure(1)
#plt.hist(traindataA,bins=30)

##plt.figure(2)
#plt.hist(traindataB,bins=30)
#plt.show(1)
##plt.show(2)


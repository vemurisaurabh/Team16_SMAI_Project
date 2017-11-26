from __future__ import division
import numpy as np
import scipy
import random
import numpy as np
import numpy.matlib
from math import e, sqrt, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from classdata import generatedata

def decideAorB(data,datasize,mewA,sigA,mewB,sigB):
	# This returns 0 for A and 1 for B
	decision=np.zeros(datasize)
	lognumerator = np.log(1/sqrt(2*pi*sigA))-((data-mewA)**2)/(2*sigA)
	logdenominator = np.log(1/sqrt(2*pi*sigB))-((data-mewB)**2)/(2*sigB)
	
	diff=lognumerator-logdenominator
	
	for i in range(0,datasize-1):
		if (diff[i] >=0):
			decision[i]=0
		else :
			decision[i]=1
	
	return decision

Nvals=np.array([100,500,1000,10000])
l1=len(Nvals)

diffvals=np.array([200,100,90,80,70,60,50,40,30,20,10])

sig0vals=np.array([200,400,500,1000])
l2=len(sig0vals)

plt.figure()
#ax=fig.gca(projection='3d')

accvals =np.zeros(np.size(diffvals))
############################################################################
i=-1
j=-1
k=-1

sig0A=sig0vals[j]
sig0B=sig0vals[j]

# Class A parameters 

partitionSize=1
mean0A=100
#sig0A=25
sigma2A=500



for i in range(0,np.size(Nvals)):
	for j in range(0,np.size(diffvals)):
		k=k+1
		N = Nvals[i]
		

		###################

		# Class B parameters 
		mean0B=mean0A+diffvals[j]
		#sig0B=25
		sigma2B=500

		###################


		traindataA, traindataB, trainNA, trainNB =generatedata(mean0A,sig0A,sigma2A,mean0B,sig0B,sigma2B,N,partitionSize)

		classAxbar=np.average(traindataA)
		classBxbar=np.average(traindataB)

		updatedmewA=((trainNA*sig0A)/(trainNA*sig0A+sigma2A))*classAxbar+(sigma2A/(trainNA*sig0A+sigma2A))*mean0A
		updatedmewB=((trainNB*sig0B)/(trainNB*sig0B+sigma2B))*classBxbar+(sigma2B/(trainNB*sig0B+sigma2B))*mean0B

		finalsigma2A=sigma2A+(sigma2A*sig0A)/(trainNA*sig0A+sigma2A)
		finalsigma2B=sigma2B+(sigma2B*sig0B)/(trainNB*sig0B+sigma2B)

		testdataA, testdataB, testNA, testNB =generatedata(mean0A,sig0A,sigma2A,mean0B,sig0B,sigma2B,N,partitionSize)

		temp1=decideAorB(data=testdataA,datasize=testNA,mewA=updatedmewA,sigA=finalsigma2A,mewB=updatedmewB,sigB=finalsigma2B)
		temp2=decideAorB(data=testdataB,datasize=testNB,mewA=updatedmewA,sigA=finalsigma2A,mewB=updatedmewB,sigB=finalsigma2B)

		tp=np.sum(temp2==1)
		tn=np.sum(temp1==0)
		fp=np.sum(temp1==1)
		fn=np.sum(temp2==0)

		acc=100*(tp+tn)/(tp+tn+fp+fn)
		accvals[k]=acc
	plt.plot(diffvals,accvals,label='N = %s'%(Nvals[i]))
	plt.xlabel('Difference in Mu0A and Mu0B')
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


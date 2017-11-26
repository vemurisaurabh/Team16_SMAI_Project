from __future__ import division
import numpy as np
import scipy
import random
import matplotlib
from math import e, sqrt, pi

def expo(lamb,theta):
	pois = lamb*e**(-lamb*theta)
	return pois

def generateExponential(lamb0=4,N=10000,partitionSize=0.1):

	sd=float(1/lamb0)
	numPartitions=np.int(np.floor((7*sd)/partitionSize))

	v=np.linspace(0,7*sd,numPartitions)

	# numSamples in each muVals
	numSamples=np.zeros(np.size(v),dtype='int')
    
	thetaVals=np.zeros(np.size(v))

	for i in range (0,len(v)-1):
	    mid_point_theta = (v[i] + v[i+1])/2
	    mid_pointy = expo(lamb0,mid_point_theta) 
	    prob_val=mid_pointy*(v[i+1]-v[i])
	    numSamples[i]=np.int(np.floor(N*prob_val))
	    thetaVals[i]=mid_point_theta
	    #print numSamples[i]

	actualN= np.sum(numSamples)

    # Data generation


	xVals=np.array([])

	for i in range (0,numPartitions-1):
		xVals=np.append(xVals,np.random.exponential(thetaVals[i],numSamples[i]))
    
    
    ## Function ends
    
	return xVals,actualN


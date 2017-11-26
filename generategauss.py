from __future__ import division
import numpy as np
import scipy
import random
import matplotlib
from math import e, sqrt, pi


def gaussian(x,mu,s):
	gauss = (1/(sqrt(2*pi)*s))*e**(-0.5*(float(x-mu)/s)**2)
	return gauss


def generategauss(mew=100,sig0=900,sigma2=1000,N=100000,partitionSize=0.1):

    #for mew    mean    sig0

    #for x   sigma2   N

    sd= np.sqrt(sig0)

    numPartitions=np.int(np.floor((6*sd) /partitionSize))

    v=np.linspace(mew-3*sd,mew+3*sd,numPartitions)

	# numSamples in each muVals
    numSamples=np.zeros(np.size(v),dtype='int')
    
    muVals=np.zeros(np.size(v))

    for i in range (0,len(v)-1):
	    mid_pointx = (v[i] + v[i+1])/2
	    mid_pointy = gaussian(mid_pointx,mew,sd) 
	    prob_val=mid_pointy*(v[i+1]-v[i])
	    numSamples[i]=np.int(np.floor(N*prob_val))
	    muVals[i]=mid_pointx
	    #print numSamples[i]

    actualN= np.sum(numSamples)

    # Data generation


    xVals=np.array([])

    for i in range (0,numPartitions-1):
        xVals=np.append(xVals,np.random.normal(muVals[i],sqrt(sigma2),numSamples[i]))
    
    
    ## Function ends
    
    return xVals,actualN


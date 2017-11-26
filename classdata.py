from __future__ import division
import numpy as np
import scipy
import random
import numpy as np
from math import e, sqrt, pi
import matplotlib.pyplot as plt

from generategauss import generategauss

def generatedata(mean0A,sig0A,sigma2A,mean0B,sig0B,sigma2B,N,partitionSize):

# mean0A for mew0 
# sig0A for mew0 
# sigma2A for x given mew for class A
	dataA,NA=generategauss(mew=mean0A,sig0=sig0A,sigma2=sigma2A,N=N,partitionSize=partitionSize)

# For B same as A
	dataB,NB=generategauss(mew=mean0B,sig0=sig0B,sigma2=sigma2B,N=N,partitionSize=partitionSize)

	return dataA, dataB, NA, NB

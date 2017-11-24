from __future__ import division
import numpy as np
import scipy
import random
import numpy as np
from math import e, sqrt, pi
import matplotlib.pyplot as plt

from generategauss import generategauss


# For  mew pdf 
mew0=100
sig0=400

# For the data condtional pdf

sigma2=10
N=200
partitionSize=1
####################

data, N =generategauss(mew=mew0,sig0=sig0,sigma2=sigma2,N=N,partitionSize=partitionSize)

deno=N*sig0+sigma2

r1=np.float(N*sig0/deno)
r2=np.float(sigma2/deno)

xnbar=np.average(data)
mewnew= (r1*xnbar)+(r2*mew0)
varnew = sigma2+(sig0*sigma2)/deno

print "xnbar = ", xnbar
print "data variance = " , np.var(data)

print "New Mew = ", mewnew
print "New variance = ", varnew

print "r1 = ", r1
print "r2 = ", r2

plt.plot(data)
plt.show()

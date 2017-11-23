import  numpy as nm
import random
import math
def normpdf(x, mean, sd):
    var = float(sd)**2
    pi = 3.1415926
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom
X=[]
Z=[]
lamda=[]
mu1=1
sigma1=1
mu2=2
sigma2=2
for x in range(0,1000000):
    y=random.uniform(0,1)
    if(y>=0.75):
        s=nm.random.normal(mu1,sigma1,1)
        X.append(s.item())
    else:
	s=nm.random.normal(mu2,sigma2,1)
        Z.append(s.item())
print nm.mean(X)
print nm.var(X)
print nm.mean(Z)
print nm.var(Z)
#############
l=nm.mean(X)
m=nm.var(X)
mean2=nm.mean(Z)
var2=nm.var(Z)
#######
H=[]
S=[]
for x in range(0,1000):
    y=random.uniform(0,1)
    if(y>=0.75):
        s=nm.random.normal(mu1,sigma1,1)
        H.append(s.item())
    else:
        s=nm.random.normal(mu2,sigma2,1)
        S.append(s.item())

for x in range(0,len(H)):
        lamda.append(H[x])
for x in range(0,len(S)):
        lamda.append(S[x])
print lamda
print len(H)
print len(S)
W1=[]
W2=[]
for x in range(0,1000):
        g=lamda[x]
        g1=normpdf(g,l,m)
        g2=normpdf(g,mean2,var2)
        W1.append(g1)
        W2.append(g2)
W=[]
for x in range(0,1000):
        if (W1[x]>=(W2[x])*3):
            W.append(0)
        else:
            W.append(1) 
print W
s=0
r=0
for x in range(0,len(H)):
             if(W[x]==1):
                 s=s+1
for x in range(len(H),len(H)+len(S)):
              if(W[x]==0):
                 r=r+1           
print s 
print r
print float(s+r)/10





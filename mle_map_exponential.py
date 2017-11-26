from __future__ import division
import numpy as nm
import scipy
import random
import matplotlib.pyplot as plt
import math
from math import e, sqrt, pi

def expopdf(lamda,x):
    num=math.exp(-(float(x)*float(lamda)))
    num=num*lamda
    return num

def source(l1,l2,pB):
    A=[]
    B=[]

    ## Generating Training data to compute the mean and variance ##
    for x in  range(0,10000):
        y=random.uniform(0,1)
        if(y>=pB):
            s=nm.random.exponential(1/l1,1)
            A.append(s.item())
        else:
            s=nm.random.exponential(1/l2,1)
            B.append(s.item())

    A=nm.array(A)
    B=nm.array(B)
    return A,B 


lam1=1
lam2=nm.arange(0.1,1,0.05)
diff=(1/lam2)-1
pB=nm.arange(0.1,0.6,0.1)

Acc=nm.zeros(nm.size(lam2))

for p in range(0,nm.size(pB)):
    for q in range(0,nm.size(lam2)):
        A,B=source(lam1,lam2[q],pB[p])

        ## Computing the Likelihood Estimates ##
        lamda1=1/nm.mean(A)
        lamda2=1/nm.mean(B)

        # print lamda1
        # print lamda2
        #######

        ## Testing ##
        a=[]
        b=[]
        N_test=100
        lamda=[]
        for x in range(0,N_test):
            y=random.uniform(0,1)
            if(y>=pB[p]):
                s=nm.random.exponential((1/lamda1),1)
                a.append(s.item())
            else:
                s=nm.random.exponential((1/lamda2),1)
                b.append(s.item())

        for i in range(0,len(a)):
                lamda.append(a[i])


        for i in range(0,len(b)):
                lamda.append(b[i])


        # print len(a)
        # print len(b)

        La=[]
        Lb=[]

        for i in range(0,N_test):
                g=lamda[i]
                g1=expopdf(lamda1,g)
                g2=expopdf(lamda2,g)
                La.append(g1)
                Lb.append(g2)
        Wmap=[]
        Wml=[]

        for i in range(0,N_test):
                if (La[i] >= Lb[i]):
                    Wml.append(0)
                else:
                    Wml.append(1) 


        for i in range(0,N_test):
                if ((La[i]*(1-pB[p])) >= (Lb[i]*pB[p])):
                    Wmap.append(0)
                else:
                    Wmap.append(1) 


        # print Wml
        # print Wmap

        s_ml=0
        r_ml=0
        s_map=0
        r_map=0

        for i in range(0,len(a)):
            if(Wml[i]==1):
                s_ml=s_ml+1
            if(Wmap[i]==1):
                s_map=s_map+1

        for i in range(len(a),len(a)+len(b)):
            if(Wml[i]==0):
                r_ml=r_ml+1
            if(Wmap[i]==0):
                r_map=r_map+1


        Acc_ml=100-(100*float((s_ml+r_ml)/N_test))
        Acc_map=100-(100*float((s_map+r_map)/N_test))

        Acc[q]=Acc_map-Acc_ml

    plt.plot(diff,Acc,label='Prior = %s'%(pB[p]))
    plt.xlabel('Difference in Means')
    plt.ylabel('Difference between MAP and MLE Accuracies')
    plt.legend()
    plt.hold('true')

plt.show()




from __future__ import division
import numpy as nm
import scipy
import random
import matplotlib.pyplot as plt
import math
from math import e, sqrt, pi

def normpdf(x, mean, var):
    denom = (2*pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom


def source(mu1, sigma1, mu2, sigma2, pB):
    A=[]
    B=[]
    ## Generating Training data to compute the mean and variance ##
    for x in  range(0,10000):
        y=random.uniform(0,1)
        if(y>=pB):
            s=nm.random.normal(mu1,sqrt(sigma1),1)
            A.append(s.item())
        else:
            s=nm.random.normal(mu2,sqrt(sigma2),1)
            B.append(s.item())

    A=nm.array(A)
    B=nm.array(B)
    return A,B 


mu1=0
sigma1=25
mu2=nm.arange(1,20,2)
sigma2=sigma1
pB=nm.arange(0.1,0.6,0.1)

Acc=nm.zeros(nm.size(mu2))

for p in range(0,nm.size(pB)):
    for q in range(0,nm.size(mu2)):

        A,B=source(mu1,sigma1,mu2[q],sigma2,pB[p])

        ## Computing the Likelihood Estimates ##
        mean1=nm.mean(A)
        var1=nm.var(A)
        mean2=nm.mean(B)
        var2=nm.var(B)
        #######

        ## Testing ##
        a=[]
        b=[]
        N_test=1000
        lamda=[]
        for x in range(0,N_test):
            y=random.uniform(0,1)
            if(y>=pB[p]):
                s=nm.random.normal(mu1,sigma1,1)
                a.append(s.item())
            else:
                s=nm.random.normal(mu2[q],sigma2,1)
                b.append(s.item())

        for i in range(0,len(a)):
                lamda.append(a[i])


        for i in range(0,len(b)):
                lamda.append(b[i])

        La=[]
        Lb=[]

        for i in range(0,N_test):
                g=lamda[i]
                g1=normpdf(g,mean1,var1)
                g2=normpdf(g,mean2,var2)
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
        # print '\n'
        # print Wmap

        a_ml_tn=0
        a_map_tn=0

        a_ml_fp=0
        a_map_fp=0

        b_ml_tp=0
        b_map_tp=0

        b_ml_fn=0
        b_map_fn=0
        for i in range(0,len(a)):
            if(Wml[i]==0):
                a_ml_tn=a_ml_tn+1
            else:
                a_ml_fp=a_ml_fp+1
            if(Wmap[i]==0):
                a_map_tn=a_map_tn+1
            else:
                a_map_fp=a_map_fp+1


        for i in range(len(a),len(a)+len(b)):
            if(Wml[i]==1):
                b_ml_tp=b_ml_tp+1
            else:
                b_ml_fn=b_ml_fn+1
            if(Wmap[i]==1):
                b_map_tp=b_map_tp+1
            else:
                b_map_fn=b_map_fn+1

        # print 'No. elements in Class A is: %f'%len(a)
        # print 'No. elements in Class B is: %f'%len(b)


        # print '\n'
        # print 'True Negative for ml: %f'%(a_ml_tn/N_test)
        # print 'True Positive for ml: %f'%(b_ml_tp/N_test)
        # print 'False Positive for ml: %f'%(a_ml_fp/N_test)
        # print 'False Negative for ml: %f'%(b_ml_fn/N_test)

        acc_ml=(a_ml_tn+b_ml_tp)*100/N_test

        # print 'MLE accuracy : %f'%(acc_ml)
        # print '\n'

        # print 'True Negative for map: %f'%(a_map_tn/N_test)
        # print 'True Positive for map: %f'%(b_map_tp/N_test)
        # print 'False Positive for map: %f'%(a_map_fp/N_test)
        # print 'False Negative for map: %f'%(b_map_fn/N_test)

        acc_map=(a_map_tn+b_map_tp)*100/N_test
        Acc[q]=acc_map-acc_ml

        # print 'Map accuracy: %f'%(acc_map)
        # print '\n'
    plt.plot(mu2,Acc,label='Prior = %s'%(pB[p]))
    plt.xlabel('Difference in Means')
    plt.ylabel('Difference between MAP and MLE Accuracies')
    plt.legend()
    plt.hold('true')


plt.show()





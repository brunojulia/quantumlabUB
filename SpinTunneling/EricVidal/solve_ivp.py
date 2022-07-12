# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 11:30:22 2022

@author: Eric Vidal Marcos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def odes(t, x, D, h, B):
    
    
    a_1 = x[0]
    a0 = x[1]
    a1 = x[2]
    
    da_1dt=-1j*(a_1*(-D+h*t)+a1*B)
    da0=0
    da1dt=-1j*(a1*(-D-h*t)+a_1*B)
    
    return [da_1dt,da0,da1dt]

#IC
x0=[1+0j, 0+0j, 0+0j]
D=1
h=0.1
B=0.1

#test the defined function odes
print(odes(0, x0, D, h, B))


#declare a time vector (time window) and parameters
p=(D, h, B)

#solve
x=solve_ivp(odes,[-10,10], x0, args=p)

t=x.t[:]
a_1 = x.y[0,:]
a0 = x.y[1,:]
a1 = x.y[2,:]
a_12 = np.abs(a_1)**2
a02 = np.abs(a0)**2
a12 = np.abs(a1)**2
sumat=np.zeros((3,np.size(a1)))
sumat[0,:]=a_12[:]
sumat[1,:]=a02[:]
sumat[2,:]=a12[:]
norm = np.sum(sumat, axis=0)

plt.title('Test')
plt.xlabel("t")
plt.ylabel('a^2')
plt.axhline(y=1.0,linestyle='--',color='grey')
plt.plot(t, a_12,'-',label='m=-1')
plt.plot(t, a02,'-',label='m=0')
plt.plot(t, a12,'-',label='m=1')
plt.plot(t, norm,'-',label='norma')
plt.legend()

plt.show()
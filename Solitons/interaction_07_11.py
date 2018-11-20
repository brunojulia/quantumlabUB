# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 17:03:49 2018

@author: Rosa
"""

import numpy as np
import scipy.sparse
from scipy.sparse import diags
from scipy.sparse import linalg
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

start_time = time.time()

#define the spacing and time interval
limits=15
dz=0.1 #spacing
Nz=(limits-(-limits))/dz #number of points
z=np.linspace(-limits,limits,Nz) #position vector, from -10 to 10 Nz points
dt=0.009 #time interval

#parameters of the solutions of the solitons
v=0.9 #velocity (goes from 0 to 1)
n=5 #density, n_inf for grey solitons, n_0 for bright solitons
z0=0 #initial position 

g=-1 #interaction term, -1 for bright solitons, 1 for grey solitons

#function to evolve
def gaussian(x,mu,sigma):
    return np.exp(-(x - mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi*sigma**2))

def bright(z,t,v,n,z0):
    "solution for a bright soliton whith Vext=0"
    imag= 0.0 + 1j
    arg1=((z-z0)*v)
    arg2=(t*0.5*(1 - v**2))
    psi=np.sqrt(n)*(1/np.cosh((z-z0) -v*t))*np.exp(arg1*imag)*np.exp(arg2*imag)
    return psi
    
def grey(z,t,v,n,z0):
    "solution for a dark soliton whith Vext=0"
    imag=0.0 +1j
    psi=np.sqrt(n)*(imag*v + np.sqrt(1-v**2)*np.tanh((z-z0)/np.sqrt(2)-v*t)*np.sqrt(1-v**2))
    return psi
    

#normalitzation function
def Normalitzation(array,h):
    """
    Computes the normalitzation constant using the simpson's method for a 1D integral
    the function to integrate is an array, h is the spacing between points 
    and returns a float
    """
    constant=0.0
    for i in range(len(array)):
        if (i == 0 or i==len(array)):
            constant+=array[i]*array[i].conjugate()
        else:
            if (i % 2) == 0:
                constant += 2.0*array[i]*array[i].conjugate()
            else:
                constant += 4.0*array[i]*array[i].conjugate()
    constant=(h/3.)*constant
    return np.sqrt(1/np.real(constant))
    
def interact(g,n,funct):
    """
    Interaction term of the GP equation g is g/|g|=1 for grey solitons, +1
    for bright solitons, n is the density of the infinity for grey solitons and
    the central density n_0 for bright solitons. funct is an array with 
    the state of the system at a certain time.
    """
    return g*np.real(funct*funct.conjugate())/n
    
#sytem at time t, it has to include all the boundary conditions
func_0=[]
if g==-1:
    for position in z:
        func_0.append(bright(position,0,v,n,z0))
else:
    for position in z:
        func_0.append(gaussian(position,0,1)*grey(position,0,v,n,0.5))

func_0=np.asanyarray(func_0) #turs to an ndarray (needed for the tridiag solver)

#we store the norm at t=0 as it will be useful for cheking if it is preserved during
#the time evolution.
norm_0=Normalitzation(func_0,dz)
print('n0',max(np.real(func_0*func_0.conjugate())))


#system at time t+1
func_1=func_0

#matrixs for the Crank-Nicholson method
#first [] the numbers to plug in the diagonals, second [] position of the 
#diagonals (0 is the main one), shape: matrix's shape
#we compute the main diagonals of the matrices, which in general will depend 
#on the position z
r= (1j*dt)/(4*dz**2) #parameter of the method 
middleA=[1+2*r +2*r*dz**2*interact(g,n,func_0)] #main diagonal of the matrix A
middleB=[1-2*r -2*r*dz**2*interact(g,n,func_0)] #main diagonal for the matrix B
A=diags([-r,middleA,-r],[-1,0,1], shape=(len(func_0),len(func_0)))
A=scipy.sparse.csr_matrix(A) #turs to sparse csr matrix (needed for the tridiag solver)
B=diags([r,middleB,r],[-1,0,1], shape=(len(func_0),len(func_0)))

#plot of the square modulus of phy at t=0
plt.ylabel('$|\psi(\~z)|^2$/n')
plt.xlabel('$\~z$')
#plt.xlim(-10,10)
#plt.ylim(-0.2,1.2)
plt.plot(z,np.real(func_0*func_0.conjugate())/n)

energy=0
t=0
dif_norm=0
ev_time= 10
counter=0
print('evolution time:',ev_time)

while t < ev_time:
    #ndarray b product of B and the system's state at time t
    prod=B.dot(func_0)
    #solver of a tridiagonal problem
    func_1=linalg.spsolve(A,prod)
    #we store the maximum diference between the norm_0 and the evolved one
    dif_norm=max(dif_norm,abs(norm_0 - Normalitzation(func_1,dz)))
    #redefine each matrix
    func_0=func_1
    t += dt
    if counter% 200 ==0:
        plt.plot(z,np.real(func_1*func_1.conjugate())/n)
    counter+=1

#plot the evolved function
print('norm_diference:',dif_norm)
print('r',r)
print('n_0 evolved',max(np.real(func_1*func_1.conjugate())))
# your code
elapsed_time = time.time() - start_time
print('computing time=',elapsed_time)


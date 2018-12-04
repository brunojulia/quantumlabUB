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
import warnings

warnings.filterwarnings("ignore",".*GUI is implemented.*")

start_time = time.time()

#functions to evolve
def gaussian(x,t,mu,sigma,w):
    return (np.exp(-(x - mu)**2/(2*sigma**2))*np.exp(-0.5j*w*t))/(np.sqrt(2*np.pi*sigma**2))

def bright(z,t,v,n,z0):
    "solution for a bright soliton whith Vext=0"
    imag= 0.0 + 1j
    arg1=((z-z0)*v)
    arg2=(t*0.5*(1 - v**2))
    arg3=(-(z+z0)*v)
    psi=np.sqrt(n)*(1/np.cosh((z-z0) -v*t))*np.exp(arg1*imag)*np.exp(arg2*imag)
    psi2=np.sqrt(n)*(1/np.cosh((z+z0) +v*t))*np.exp(arg3*imag)*np.exp(arg2*imag)
    return psi + psi2
    
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
    
#simpsion method
def Simpson(array,h):
    """
    Simpson's method for a 1D integral. Takes the function to integrate as
    an array. h is the spacing between points. Returns a float
    """
    suma=0.0
    for i in range(len(array)):
        if (i==0 or i==len(array)):
            suma+=array[i]
        else:
            if (i % 2) == 0:
                suma+= 2.0*array[i]
            else:
                suma+= 4.0*array[i]
    suma=(h/3.)*suma
    return suma
    
#interaction term of the GP equation    
def interact(g,n,funct):
    """
    Interaction term of the GP equation g is g/|g|=1 for grey solitons, +1
    for bright solitons, n is the density of the infinity for grey solitons and
    the central density n_0 for bright solitons. funct is an array with 
    the state of the system at a certain time.
    """
    return g*np.real(funct*funct.conjugate())/n
    
#External potential of the GP
def potential(x,dx,x0):
    """
    External potential, typically it will depend on the positon.
    """
    pot=10*(np.exp(-(x)**2/(2*0.5**2)))/(np.sqrt(2*np.pi*0.5**2))*0
    return (pot + 0*0.5*x**2)
    

#define the spacing and time interval
limits=10
dz=0.022 #spacing
Nz=(limits-(-limits))/dz #number of points
z=np.linspace(-limits,limits,Nz) #position vector, from -10 to 10 Nz points
dt=0.01 #time interval

#parameters of the solutions of the solitons
v=0.7#velocity (goes from 0 to 1)
n=3 #density, n_inf for grey solitons, n_0 for bright solitons
z0=-3 #initial position 
g=-1 #interaction term, -1 for bright solitons, 1 for grey solitons 0 harmonic
w=1#harmonic oscillator

    
#sytem at time t, it has to include all the boundary conditions
func_0=[]
if g == -1:
    for position in z:
        func_0.append(bright(position,0,v,n,z0))
elif g == 1:
    for position in z:
        func_0.append(grey(position,0,v,n,0.5))
else:
    for position in z:
        func_0.append(gaussian(position,0,z0,1,w))


func_0=np.asanyarray(func_0) #turns to an ndarray (needed for the tridiag solver)
#we store the norm at t=0 as it will be useful for cheking if it is preserved during
#the time evolution.
norm_0=Normalitzation(func_0,dz)
if g==0:
    constant=1/np.sqrt(Simpson(np.real(func_0*func_0.conjugate()),dz))
    func_0=func_0*constant
    norm_0=Simpson(np.real(func_0*func_0.conjugate()),dz)
    print('norm_0', norm_0)
#system at time t+1
func_1=func_0


#plot of the square modulus of phy at t=0
#plt.ylabel('$|\psi(\~z)|^2')
#plt.xlabel('$\~z$')
#plt.plot(z,np.real(func_0*func_0.conjugate()))


t=0
dif_norm=0
ev_time= 7
counter=0
r= (1j*dt)/(4*dz**2) #parameter of the method 
print('r', r)
print('evolution time:',ev_time)

V=[]
for position in z:
    V.append(2*r*dz**2*potential(position,0,dz))
V=np.array(V)

#to enable interactive plotting
plt.ion()
plt.hold(False)


expected_z=[]
time_evol=[]
while t < ev_time:
    #matrixs for the Crank-Nicholson method
    #first [] the numbers to plug in the diagonals, second [] position of the 
    #diagonals (0 is the main one), shape: matrix's shape
    #we compute the main diagonals of the matrices, which in general will depend 
    #on the position z
    mainA=[1+2*r +2*r*dz**2*interact(g,n,func_0)] #main diagonal of A matrix (time t+ 1)
    mainB=[1-2*r -2*r*dz**2*interact(g,n,func_0)] #main diagonal of B matrix (time t)
    mainA= np.array(mainA) + V
    mainB= np.array(mainB) - V
    A=diags([-r,mainA,-r],[-1,0,1], shape=(len(func_0),len(func_0)))
    A=scipy.sparse.csr_matrix(A) #turs to sparse csr matrix (needed for the tridiag solver)
    B=diags([r,mainB,r],[-1,0,1], shape=(len(func_0),len(func_0)))
    #ndarray b product of B and the system's state at time t
    prod=B.dot(func_0)
    #solver of a tridiagonal problem
    func_1=linalg.spsolve(A,prod)
    #we store the maximum diference between the norm_0 and the evolved one
    dif_norm=max(dif_norm,abs(norm_0 - Normalitzation(func_1,dz)))
    #redefine each matrix
    func_0=func_1
    t += dt
#    expected_z.append(Simpson(np.real(z*func_1*func_1.conjugate()),dz))
#    time_evol.append(t)
#    if counter% 200 ==0:
#        plt.plot(z,np.real(func_1*func_1.conjugate()))
    counter+=1
    if counter%6 ==0:
        plt.cla()
        plt.plot(z,np.real(func_1*func_1.conjugate())/n)
    plt.pause(0.00000001)

plt.show(block=True)
check=[]
for position in z:
    check.append(bright(position,ev_time-dt,v,n,z0))

check=np.asanyarray(check)


"""
#plot the evolved function
plt.ylabel('$|\psi(\~z)|^2$')
plt.xlabel('$\~z$')
plt.plot(z,np.real(func_1*func_1.conjugate()))

"""

#scalar product to check
check2=[]
for position in z:
    check2.append(bright(position,ev_time - dt,v,n,z0))
check2=np.asanyarray(check2)
initial=Simpson(check2*check2.conjugate(),dz)
dot_product=Simpson(check*check.conjugate(),dz)
print(dot_product,initial)
print(Simpson(func_1*check.conjugate(),dz))

"""
#plot the expected value of z
plt.ylabel('$<\~z>$')
plt.xlabel('$\~t$')
plt.plot(time_evol,expected_z)

"""

#plt.plot(z,np.real(check*check.conjugate()))

print('norm_diference:',dif_norm)
#print('n_0 evolved',max(np.real(func_1*func_1.conjugate())))
# your code
elapsed_time = time.time() - start_time
print('computing time=',elapsed_time)



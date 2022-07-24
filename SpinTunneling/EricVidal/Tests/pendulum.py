# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 16:42:22 2022

@author: Eric Vidal Marcos
"""
import numpy as np
import matplotlib.pyplot as plt

dim=2   #num equa's
#parametres del problema
g=10
l=0.1

#CI
t0=0
tf=5

#RK4 steps
nstep = 2000   #number
h = (tf-t0)/nstep   #step

#CI
x0=np.zeros(dim, dtype='float')
y0=np.zeros(dim, dtype='float')

p0=2
theta0=np.pi/4

dtheta0=p0

y0[0]=theta0
y0[1]=dtheta0

def pendul(k, x, y):
    if (k==0):
        dypas=y[1]
    if (k==1):
        dypas=-(g/l)*np.sin(y[0])
    return dypas


#RK4 algorithm for solve 1 step of ODE         
def RK4(t, am, fun_dif):
    '''Inputs: s (int) total spin, t (int or float) time, am (array (dim,1))
    coefficients of each state, h (int or float) step.
    Outputs: ak (complex array dim 2s+1) 1 step.
    This function returns every differential equation solution for coefficients
    time evolution.'''
    for k in range(dim):   #apliquem un pas de RK4 per a totes les ED acoblades
            k1 = h*fun_dif(k, t, am)
            k2 = h*fun_dif(k, t + h/2, am + k1/2)
            k3 = h*fun_dif(k, t + h/2, am + k2/2)
            k4 = h*fun_dif(k, t + h, am + k3)
            am[k] = am[k] + (k1 + 2*k2 + 2*k3 + k4)/6
    return am

#Array to save ys and time
ys=np.zeros((dim, nstep+1), dtype='float')
ti=np.zeros(nstep+1, dtype='float')

#IC
ti[0]=t0
for i in range(dim):
    ys[i,0]=y0[i]

#System resolution
t=t0
for n in range(nstep):
    print(n)    #print step number (PODRIA SER UN CARGANDO)
    y0=RK4(t, y0, pendul)  #RK4 step
    
    #Save every value to afterwards plot evo in time, and continue RK4 
    for i in range(dim):
        ys[i, n+1]=y0[i]
    ti[n+1]=t
    
    #Input time for next step
    t=t0+n*h

plt.title('Pendulum test RK4: '+'N='+str(nstep))
plt.xlabel('t (s)')
plt.ylabel('y')
plt.axhline(y=0,linestyle='--',color='grey')
for i in range(dim):
    if i==0:
        lab=r'$\theta$'+' (rad)'
    elif i==1:
        lab='d/dt ('+r'$\theta$'+') (rad/s)'
    plt.plot(ti, ys[i,:],'-',label=lab)
plt.legend()

plt.show()
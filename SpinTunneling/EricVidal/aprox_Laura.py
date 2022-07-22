# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:31:51 2021

@author: Laura Guerra
"""
import numpy as np
from matplotlib import pyplot as plt


s=1     #Spin
dim=int(2*s+1)
#Parametres Hamiltonia
hbar = 1.
a=0.1
alpha=0.1

#Operadors Ham
Sx = hbar/np.sqrt(2.)*np.array([[0,1,0],[1,0,1],[0,1,0]])
Sy = hbar/np.sqrt(2.)*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])
Sz = hbar*np.array([[1,0,0],[0,0,0],[0,0,-1]])

Sxy2 = np.dot(Sx,Sx)-np.dot(Sy,Sy)
Sz2 = np.dot(Sz,Sz)

# Hamiltonia depenent del temps amb unitats d'energia entre D
def Hamiltonia(t):
    Ham = -Sz2 - a * hbar * t * Sz + alpha * (Sxy2)
    return Ham

# Base de vectors propis de spin 1: |1>, |0> i |-1>
mm=[]    #Conjunt dels 3 vects propis
for i in range(dim):
    m_i = [0]*(dim-1-i)+[1]+[0]*(i) #Corre l'1 des del final fins al principi
    mm.append(m_i)

           
#Condicions inicials dels coeficients de la funcio d'ona
a_m=np.zeros(shape=(int(2*s+1),1),dtype=complex)
a_m[0]=1+0j


# Equacions diferencials de primer ordre associades a les diferents a_m
def dam(i, t, am):
    m_i=np.array([mm[i]])   #element de la base corresponent al coef am_i
    
    sum_H=0.    #sera equa dif a falta dun factor
    for k in range (dim):       #recorrem per tots els coef
        m_k=np.array([mm[k]])
        m_k=np.transpose(m_k)   #element del coef k
        sum_H += am[k]*np.dot(m_i,(np.dot(Hamiltonia(t),m_k)))
        if (k==i):
            eigenterm=am[k]*np.dot(m_i,(np.dot(Hamiltonia(t),m_k)))
    if (n<10):
        sumterm=sum_H-eigenterm
        print(k,sumterm)
    dam_i=-1j*sum_H   #equa dif
    
    return dam_i


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
#Evolution time frame
t0=-10
tf=10

#RK4 steps
nstep = 20000   #number
h = (tf-t0)/nstep   #step

t=t0    #Initial time

#Array to save coefficients probabilities, time, and states norm
asave=np.zeros((dim, nstep+1), dtype='float')
ti=np.zeros(nstep+1, dtype='float')
norm=np.zeros(nstep+1, dtype='float')

#IC
ti[0]=t0
for i in range(dim):
    print(type(np.abs(a_m[i])**2))
    asave[i,0]=np.abs(a_m[i])**2
    norm[0] = norm[0]+asave[i,0]


#System resolution 
for n in range(nstep):
    print(n)    #print step number (PODRIA SER UN CARGANDO)
    a_m=RK4(t, a_m, dam)  #RK4 step
    
    #Save every value to afterwards plot evo in time, and continue RK4 
    for i in range(dim):
        asave[i,n+1]=np.abs(a_m[i])**2
        norm[n+1] = norm[n+1]+asave[i,n+1]
    ti[n+1]=t
    
    #Input time for next step
    t=t0+n*h
    

plt.title('Straight forward operators RK4: '+'N='+str(nstep))
plt.xlabel('t')
plt.ylabel('$|a|^2$')
plt.axhline(y=1.0,linestyle='--',color='grey')
for i in range(dim):
    plt.plot(ti, asave[i,:],'-',label='m='+str(i-s))
plt.plot(ti, norm,'-',label='norma')
plt.legend()

plt.show()


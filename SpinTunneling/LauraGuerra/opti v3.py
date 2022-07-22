# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:31:51 2021

@author: Laura Guerra
"""
import kivy
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
import numpy as np
from matplotlib import pyplot as plt

''' Definicio de les funcions'''
# Hamiltonia depenent del temps amb unitats d'energia entre D
def H(alpha,t):
    Ham = -Sz2 + a * hbar * t * Sz + alpha * (Sxy2)
    return Ham


# Equacions diferencials de primer ordre associades a les diferents a_m
def dam(t,am,i):
    sum_H=0.
    m_i=np.array([m[i]])
    
    for k in range (2*s+1):
        m_k=np.array([m[k]])
        m_k=np.transpose(m_k)
        sum_H=sum_H+am[k]*np.dot(m_i,(np.dot(H(alpha,t),m_k)))
        
    dam_res=-1j*sum_H
    
    return dam_res


# Funcio que utilitza el metode Runge Kutta4 per resoldre les equacions 
# diferencials de primer ordre            
def RK4(t,h,am,s):
    for i in range(int(2*s+1)):
            k0=h*dam(t,am,i)
            k1=h*dam(t+0.5*h,am+0.5*k0,i)
            k2=h*dam(t+0.5*h,am+0.5*k1,i)
            k3=h*dam(t+0.5*h,am+k2,i)
            am[i]=am[i]+(k0+2*k1+2*k2+k3)/6.
    return a_m

#Definicio de parametres generals per descriure l,Hamiltonia
hbar=1.
Sx=hbar/np.sqrt(2.)*np.array([[0,1,0],[1,0,1],[0,1,0]])
Sy=hbar/np.sqrt(2.)*np.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]])
Sz=hbar*np.array([[1,0,0],[0,0,0],[0,0,-1]])
Sxy2=np.dot(Sx,Sx)-np.dot(Sy,Sy)
Sz2=np.dot(Sz,Sz)


""" PROGRAMA """


# Creacio dels vectors propis de l'spin
m=[]
s=1
dam_res=[0]*(2*s+1)
alpha=0.1
a=0.01
t0=-10.
tf=10.

for i in range(int(2*s+1)):
    m_i=[0]*(int(2*s)-i)+[1]+[0]*(i)
    m.append(m_i)

           
#Condicions inicials
a_m=np.zeros(shape=(int(2*s+1),1),dtype=complex)
a_m[0]=1+0j


#Execucio de les funcions           

nstep=2000
h=(tf-t0)/nstep
t=t0
a1=[0]*(nstep+1)
a2=[0]*(nstep+1)
a3=[0]*(nstep+1)
ti=[0]*(nstep+1)
mod=[0]*(nstep+1)

ti[0]=t0
a1[0]=abs(a_m[0])*abs(a_m[0])
a2[0]=abs(a_m[1])*abs(a_m[1])
a3[0]=abs(a_m[2])*abs(a_m[2])
mod[0]=a1[0]+a2[0]+a3[0]

for n in range(nstep):
    print(n)
    
    an=RK4(t,h,a_m,s)
    a1[n+1]=abs(an[0])*abs(an[0])
    a2[n+1]=abs(an[1])*abs(an[1])
    a3[n+1]=abs(an[2])*abs(an[2])
    ti[n+1]=t
    mod[n+1]=a1[n]+a2[n]+a3[n]
    t=t+h
    a_m=an

plt.title('N='+str(nstep))
plt.xlabel("t'")
plt.ylabel('a^2')
plt.axhline(y=1.0,linestyle='--',color='grey')
plt.plot(ti, a1,'-',label='m=-1')
plt.plot(ti, a2,'-',label='m=0')
plt.plot(ti, a3,'-',label='m=1')
plt.plot(ti, mod,'-',label='norma')
plt.legend()

plt.show()



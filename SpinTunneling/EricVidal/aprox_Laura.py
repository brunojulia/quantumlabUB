# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 19:31:51 2021

@author: Laura Guerra
"""
import numpy as np
from matplotlib import pyplot as plt

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
s=1     #Spin
dim=int(2*s+1)

for i in range(dim):
    m_i = [0]*(dim-1-i)+[1]+[0]*(i) #Corre l'1 des del final fins al principi
    mm.append(m_i)

           
#Condicions inicials dels coeficients de la funcio d'ona
a_m=np.zeros(shape=(int(2*s+1),1),dtype=complex)
a_m[0]=1+0j


# Equacions diferencials de primer ordre associades a les diferents a_m
def dam(t, am, i):
    m_i=np.array([mm[i]])   #element de la base corresponent al coef am_i
    
    sum_H=0.    #sera equa dif a falta dun factor
    for k in range (dim):       #recorrem per tots els coef
        m_k=np.array([mm[k]])
        m_k=np.transpose(m_k)   #element del coef k
        sum_H += am[k]*np.dot(m_i,(np.dot(Hamiltonia(t),m_k)))
        
    dam_i=-1j*sum_H   #equa dif
    
    return dam_i


# Funcio que utilitza el metode Runge Kutta4 per resoldre les equacions 
# diferencials de primer ordre            
def RK4(t, h, am, dim_s):
    for i in range(dim_s):   #apliquem un pas de RK4 per a totes les ED acoblades
            k0 = h*dam(t,am,i)
            k1 = h*dam(t+0.5*h,am+0.5*k0,i)
            k2 = h*dam(t+0.5*h,am+0.5*k1,i)
            k3 = h*dam(t+0.5*h,am+k2,i)
            am[i] = am[i] + (k0+2*k1+2*k2+k3)/6.
    return am

#Parametres de RK
t0=-10
tf=10
nstep=2000
h=(tf-t0)/nstep

t=t0    #Valor inicial del temps

#Llistes buides per a probabilitat dels coeficients, temps i
#probabilitat de la funcio d'ona (normalitzada en teoria)
a1=[0]*(nstep+1)
a2=[0]*(nstep+1)
a3=[0]*(nstep+1)
ti=[0]*(nstep+1)
mod=[0]*(nstep+1)

#CI
ti[0]=t0
a1[0]=abs(a_m[0])**2
a2[0]=abs(a_m[1])**2
a3[0]=abs(a_m[2])**2
mod[0]=a1[0]+a2[0]+a3[0]

for n in range(nstep):
    print(n)    #print del numero d'operacio que porta (PODRIA SER UN CARGANDO)
    
    a_m=RK4(t, h, a_m, dim)  #pas de RK4
    
    #Guardem valor de prob, t, norma a llista
    a1[n+1]=abs(a_m[0])**2
    a2[n+1]=abs(a_m[1])**2
    a3[n+1]=abs(a_m[2])**2
    
    mod[n+1]=a1[n]+a2[n]+a3[n]
    
    #Canviem de pas per inputs del proxim pas de RK4
    t=t+h
    ti[n+1]=t

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



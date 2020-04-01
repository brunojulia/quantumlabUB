# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

import numpy as np
import matplotlib.pyplot as plt

def trapezis(xa,xb,ya,yb,dx,fun):
    Nx=np.int((xb-xa)/dx)
    Ny=np.int((yb-ya)/dx)    
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*(dx**2)/2
        for j in range(1,Ny):
            funsum=funsum+fun[i,j]*dx**2
    return funsum

normas=np.load('normasl3.npy')
tvec=np.load('tvecl3.npy')
dades=np.load('dadesl3.npy')
psi=np.load('psivecl3.npy')
L=dades[0]
Nx=np.int(dades[1])
Nt=np.int(dades[2])
#Calculem totes les normas

    
#Matriu d'energia
print(np.shape(psi[:,:,0]))

def Ham(psi,hbar,m,dx,Vvec):
    Ham=np.zeros((np.shape(psi)),dtype=complex)
    s=(hbar**2/(2*m*dx))
    for i in range(1,Nx):
        for j in range(1,Nx):
            Ham[i,j]=np.real((-s*(psi[i+1,j]+psi[i-1,j]+psi[i,j-1]
                +psi[i,j+1]-4*psi[i,j])+psi[i,j]*Vvec[i,j])*np.conj(psi[i,j]))
    #Derivades del contorn
    for i in range(0,Nx):
        Ham[0,i]=np.real((-s*(psi[1,i]+psi[0,i-1]
                +psi[0,i+1]-4*psi[0,i])+psi[0,i]*Vvec[0,i])*np.conj(psi[0,i]))
        Ham[i,0]=np.real((-s*(psi[i+1,0]+psi[i-1,0]
                +psi[i,1]-4*psi[i,0])+psi[i,0]*Vvec[i,0])*np.conj(psi[i,0]))
        Ham[Nx,i]=np.real((-s*(psi[Nx,i]+psi[Nx,i-1]
                +psi[Nx,i+1]-4*psi[Nx,i])+psi[Nx,i]*Vvec[Nx,i])*np.conj(psi[Nx,i]))
        Ham[i,Nx]=np.real((-s*(psi[i+1,Nx]+psi[i-1,Nx]+psi[i,Nx]
                -4*psi[i,Nx])+psi[i,Nx]*Vvec[i,Nx])*np.conj(psi[i,j]))
        
    return Ham

def dispersiox(xa,xb,Nx,fun):
    #Fun es la norma
    dx=(xb-xa)/Nx
    #Valor esperat de x:
    fun1=np.array([[fun[i,j]*(xa+dx*j) for j in range(Nx+1)] 
        for i in range(Nx+1)])
    xesp=trapezis(xa,xb,xa,xb,dx,fun1)
    #Valor esperat de x**2:
    fun2=np.array([[fun[i,j]*(xa+dx*j)**2 for j in range(Nx+1)] 
        for i in range(Nx+1)])
    xesp2=trapezis(xa,xb,xa,xb,dx,fun2)
    
    s2=xesp2-xesp**2
    return s2




Vvec=np.zeros((np.shape(psi[:,:,0])))
hbar=1
m=1
dx=(2*L)/Nx
H=np.zeros((len(tvec)))

devec=np.zeros((Nt+1))
for i in range(Nt+1):
    a=Ham(psi[:,:,i],hbar,m,dx,Vvec)
    H[i]=trapezis(-L,L,-L,L,dx,a)
    devec[i]=dispersiox(-L,L,Nx,normas[:,:,i])

plt.figure()
plt.suptitle('Energia de paquet lliure(L=3,p0=50,s0**2=0.25)')
plt.plot(tvec,H)
plt.ylim((4.455,4.457))
plt.xlabel('t')
plt.ylabel('E')
plt.savefig('Energial3')

plt.figure()
plt.suptitle('sigmax**2(t) (p0=50,L=3,s0**2=0.25)')
plt.plot(tvec,devec)
plt.xlabel('t')
plt.ylabel('sigma**2')
plt.savefig('Devestl3')

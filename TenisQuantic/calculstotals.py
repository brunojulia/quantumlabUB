# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

#Aquí realitzarem diversos calculs importants o simplement gràficarem alguns
#resultats


dxvec=np.array([0.05,0.075,0.10,0.15])
dtvec=np.array([0.003,0.006,0.01])

#Funcions que utilitzarem

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


def Ham(psi,hbar,m,dx,Vvec):
#Matriu que ens calcula la energia total.
    Ec=np.zeros((np.shape(psi)),dtype=complex)
    Ep=np.zeros((np.shape(psi)),dtype=complex)
    Nx=np.int(len(psi[0,:]))-1
    s=(1/2.)*(1./dx**2)
    
    for i in range(1,Nx):
        for j in range(1,Nx):
            Ec[i,j]=-s*(psi[i+1,j]+psi[i-1,j]+psi[i,j-1]
                +psi[i,j+1]-4*psi[i,j])
            Ec[i,j]=np.real(Ec[i,j]*np.conj(psi[i,j]))
            Ep[i,j]=psi[i,j]*Vvec[i,j]
            Ep[i,j]=np.real(Ep[i,j]*np.conj(psi[i,j]))
        
#Derivades del contorn-
#    for i in range(0,Nx):
#        Ham[0,i]=np.real((-s*(psi[1,i]+psi[0,i-1]
#                +psi[0,i+1]-4*psi[0,i])+psi[0,i]*Vvec[0,i])*np.conj(psi[0,i]))
#        Ham[i,0]=np.real((-s*(psi[i+1,0]+psi[i-1,0]
#                +psi[i,1]-4*psi[i,0])+psi[i,0]*Vvec[i,0])*np.conj(psi[i,0]))
#        Ham[Nx,i]=np.real((-s*(psi[Nx,i]+psi[Nx,i-1]
#                +psi[Nx,i+1]-4*psi[Nx,i])+psi[Nx,i]*Vvec[Nx,i])*np.conj(psi[Nx,i]))
#        Ham[i,Nx]=np.real((-s*(psi[i+1,Nx]+psi[i-1,Nx]+psi[i,Nx]
#                -4*psi[i,Nx])+psi[i,Nx]*Vvec[i,Nx])*np.conj(psi[i,Nx]))
        
    return Ec,Ep

def dispersiox(xa,xb,dx,fun):
    #Fun es la norma
    Nx=np.int((xb-xa)/dx)
    #Valor esperat de x:
    fun1=np.array([[fun[i,j]*(xa+dx*j) for j in range(Nx+1)] 
        for i in range(Nx+1)])
    xesp=trapezis(xa,xb,xa,xb,dx,fun1)
    #Valor esperat de x**2:
    fun2=np.array([[fun[i,j]*(xa+dx*j)**2 for j in range(Nx+1)] 
        for i in range(Nx+1)])
    xesp2=trapezis(xa,xb,xa,xb,dx,fun2)
    
    s2=xesp2-xesp**2
    return s2,xesp

def dispersioy(xa,xb,dx,fun):
    #Fun es la norma
    Nx=np.int((xb-xa)/dx)
    #Valor esperat de x:
    fun1=np.array([[fun[i,j]*(xa+dx*i) for i in range(Nx+1)] 
        for j in range(Nx+1)])
    yesp=trapezis(xa,xb,xa,xb,dx,fun1)
    #Valor esperat de x**2:
    fun2=np.array([[fun[i,j]*(xa+dx*i)**2 for i in range(Nx+1)] 
        for j in range(Nx+1)])
    yesp2=trapezis(xa,xb,xa,xb,dx,fun2)
    
    s2=yesp2-yesp**2
    return s2,yesp



#%% Temps de càlcul

#El temps de càlcul ve donat pel discretit
#zat utilitzat. 
tcalcul=np.zeros((len(dtvec),len(dxvec)))
for i in range(len(dtvec)):
    for j in range(len(dxvec)):
        dades=np.load('dvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        tcalcul[i,j]=dades[2]

fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Temps de càlcul (dx,dt)')

for i in range(len(dtvec)):
    plt.plot(dxvec,tcalcul[i,:],'.',label='dt={}'.format(dtvec[i]))
    
plt.ylabel('temps de càclul(s)')
plt.xlabel('dx')
plt.legend()
plt.savefig('Temps de càlcul')
plt.show()


#%%Energia i norma càlcul

def Hamt(dx,hbar,m,Vvec,psi):
    #Torna la energia en funció del temps
    Ht=np.zeros((len(psi[1,1,:])))
    Nx=len(psi[0,:,1])-1

    for j in range(len(psi[1,1,:])):
        a,b=Ham(psi[:,:,j],hbar,m,dx,Vvec)
        suma=np.complex128(0.)
        for n in range(0,Nx+1):
            for k in range(0,Nx+1):
                        suma=suma+a[n,k]
        Ht[j]=suma*dx**2

    return Ht

def normavec(norma,dx,L):
    #Calcula la norma a cada temps
    normalitza=np.zeros( (len(norma[1,1,:])))
    
    for j in range(len(norma[1,1,:])):
        normalitza[j]= trapezis(-L,L,-L,L,dx,norma[:,:,j])

    return normalitza

def dispersioxvec(xa,xb,dx,normas):
    Nt=len(normas[1,1,:])
    
    dispersioxvec=np.zeros((len(normas[1,1,:])))
    xespvec=np.zeros(len(normas[1,1,:]))
    
    for i in range(Nt):
         dispersioxvec[i],xespvec[i]=dispersiox(xa,xb,dx,normas[:,:,i])
    
    return dispersioxvec,xespvec

def dispersioyvec(xa,xb,dx,normas):
    Nt=len(normas[1,1,:])
    
    dispersioyvec=np.zeros((len(normas[1,1,:])))
    yespvec=np.zeros(len(normas[1,1,:]))
    
    for i in range(Nt):
         dispersioyvec[i],yespvec[i]=dispersioy(xa,xb,dx,normas[:,:,i])
    
    return dispersioyvec,yespvec
         

L=3.
for i in range(len(dtvec)):
    for j in range(len(dxvec)):
        
        Vvec=np.load('Vvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        normas=np.load('normadx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        psivec=np.load('psivecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        
        dx=dxvec[j]
        #Calcul de la energia        
#        Ec=Hamt(dx,1.,1.,Vvec,psivec)
#        np.save('Ecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]),Ec)
        #Calcul de la norma
#        normavect=normavec(normas,dx,L)
#        np.save('normatotdx{}dt{}.npy'.format(dxvec[j],dtvec[i]),normavect)
        #Calcul de la dispersió i el valor esperat
#        dispersioxvect,xespvect=dispersioxvec(-L,L,dx,normas)
#        np.save('dispersioxdx{}dt{}.npy'.format(dxvec[j],dtvec[i]),dispersioxvect)
#        np.save('xespxdx{}dt{}.npy'.format(dxvec[j],dtvec[i]),xespvect)
        #Ara també en y
        dispersioyvect,yespvect=dispersioyvec(-L,L,dx,normas)
        np.save('dispersioydx{}dt{}.npy'.format(dxvec[j],dtvec[i]),dispersioyvect)
        np.save('yespydx{}dt{}.npy'.format(dxvec[j],dtvec[i]),yespvect)




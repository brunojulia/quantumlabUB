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


def eteo(p0x,p0y,s2):
    energia=(1/2.)*(p0x**2+p0y**2 +1/(2.*s2))
    return energia


#c=['g','b','r']
#for i in range(len(dtvec)):
#    #Calcul de la energia
#    fig=plt.figure(figsize=[10,8])
#    ax=plt.subplot(111)
#    plt.suptitle('Energia paquet(p0=10)(dx,dt={})'.format(dtvec[i]))    
#    for j in range(len(dxvec)):       
#        tvec=np.load('tvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
#        H=np.load('Ecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
#        plt.plot(tvec,H,label='dx={}'.format(dxvec[j]))
    
#    energvec=np.array([eteo(10,0,0.25) for i in range(len(tvec))])        
#    plt.plot(tvec,energvec,label='Valor teòric')
    
    # Shrink current axis by 20%
#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
#    plt.xlabel('t')
#    plt.ylabel('E')
#    plt.savefig('energiadiscretitzatdt{}'.format(i))
#    plt.show()



#for i in range(len(dtvec)):
    #Calcul de la norma:
#    fig=plt.figure(figsize=[10,8])
#    ax=plt.subplot(111)
#    plt.suptitle('norma(dx,dt={})'.format(dtvec[i]))
#    for j in range(len(dxvec)):       
#        normavector=np.load('normatotdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
#        tvec=np.load('tvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
#        plt.plot(tvec,normavector,'.',label='dx={}'.format(dxvec[j]))
    
    # Shrink current axis by 20%
#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
#    plt.xlabel('t')
#    plt.ylabel('norma')
#    plt.show()
#    plt.savefig('normadiscretizatdt{}'.format(i))

#Dispersió i valor sperat per x
for i in range(len(dtvec)):
    #Calcul de la norma:
    fig=plt.figure(figsize=[10,8])
    ax=plt.subplot(111)
    plt.suptitle('Dispersiox(dx,dt={})'.format(dtvec[i]))
    for j in range(len(dxvec)):       
        normavector=np.load('dispersioxdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        tvec=np.load('tvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        plt.plot(tvec,normavector,'.',label='dx={}'.format(dxvec[j]))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('t')
    plt.ylabel('sigmax**2')
    plt.show()
    plt.savefig('dispersioxdiscretizatdt{}'.format(i))

for i in range(len(dtvec)):
    #Calcul de la norma:
    fig=plt.figure(figsize=[10,8])
    ax=plt.subplot(111)
    plt.suptitle('Valor esperat x(dx,dt={})'.format(dtvec[i]))
    for j in range(len(dxvec)):       
        normavector=np.load('xespxdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        tvec=np.load('tvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        plt.plot(tvec,normavector,'.',label='dx={}'.format(dxvec[j]))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()
    plt.savefig('xespxdiscretizatdt{}'.format(i))


#dipsersio en y    
for i in range(len(dtvec)):
    #Calcul de la norma:
    fig=plt.figure(figsize=[10,8])
    ax=plt.subplot(111)
    plt.suptitle('Dispersioy(dx,dt={})'.format(dtvec[i]))
    for j in range(len(dxvec)):       
        normavector=np.load('dispersioydx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        tvec=np.load('tvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        plt.plot(tvec,normavector,'.',label='dx={}'.format(dxvec[j]))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('t')
    plt.ylabel('sigmay**2')
    plt.show()
    plt.savefig('dispersioydiscretizatdt{}'.format(i))

for i in range(len(dtvec)):
    #Calcul de la norma:
    fig=plt.figure(figsize=[10,8])
    ax=plt.subplot(111)
    plt.suptitle('Valor esperat y(dx,dt={})'.format(dtvec[i]))
    for j in range(len(dxvec)):       
        normavector=np.load('yespydx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        tvec=np.load('tvecdx{}dt{}.npy'.format(dxvec[j],dtvec[i]))
        plt.plot(tvec,normavector,'.',label='dx={}'.format(dxvec[j]))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()
    plt.savefig('yespydiscretizatdt{}'.format(i))



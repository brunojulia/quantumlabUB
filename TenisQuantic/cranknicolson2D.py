# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

#Funció triadag
#range(n) n element pero comencem a contar desde 0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from time import time

def tridiag(a, b, c, d):
    n = len(d)  # número de filas

    # Modifica los coeficientes de la primera fila
    cp= np.zeros(n,dtype=complex)
    dp=np.zeros(n,dtype=complex)
    cp[0]=c[0]/b[0]  # Posible división por cero
    dp[0]=d[0]/ b[0]

    for i in range(1, n):
        ptemp = b[i] - (a[i] * cp[i-1])
        cp[i]= c[i]/ptemp
        dp[i] = (d[i] - a[i] * dp[i-1])/ptemp

    # Sustitución hacia atrás
    x = np.zeros(n,dtype=complex)
    x[-1] = dp[-1] #-1 vol dir l'ultim element

    for i in range(-2, -n-1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x

#Pas de CrannkNicolson
def Crannk_step(psi,avec,bxvec,byvec,cvec,r,V):
    """Realitza un pas de Crank-Nicolson per un dt."""
    
    #Primer itera per psi[i,:], és a dir, prenent psi[i,:] com
    #únic element de psi i aplicant-li un pas de Crannk-Nicolson.
    #Tot seguit segueix iterant per tot i.
    for i in range(Nx+1):
        dvec=dx(psi[i,:],i,r,V)
        psi[i,:]=tridiag(avec,bxvec[i],cvec,dvec)
    #Fa el mateix per les psi[j,:]
    for j in range(Nx+1):
        dvec=dy(psi[:,j],j,r,V)
        psi[:,j]=tridiag(avec,byvec[j],cvec,dvec)    
    return psi
    

def Crannk2D(xa,xb,ya,yb,ta,tb,Nx,Ny,Nt,V,hbar,m,psi):
    "Crank-Nicolson 2D"
    
    
    #Generem tots els números que utilitzarem
    dx=(xb-xa)/np.float(Nx)
    dy=(yb-ya)/np.float(Ny)
    #Dividim entre dos per què cada pas de Crannk Nicolson requereix
    #primer mig pas dt/2, i després un altre dt/2
    dt=(tb-ta)/(2*np.float(Nt))
    xvec=np.array([xa+i*dx for i in range(Nx+1)])
    yvec=np.array([ya+i*dy for i in range(Ny+1)])
    tvec=np.array([ta+dt*i*2 for i in range(Nt+1)])
    #Matriu de potencial
    Vvec=np.array([[V(xvec[i],yvec[j],xb,yb) for i in range(Nx+1)]
        for j in range(Ny+1)])
    
    r=(dt/(4*dx**2))*(hbar**2/m)
    #Generem vectors bx i by:
    bxvec=np.array([bx(i,r,Vvec) for i in range(Nx+1)])
    byvec=np.array([by(i,r,Vvec) for i in range(Ny+1)])
    avec=np.insert(ac(r,Vvec),0,0)
    cvec=np.append(ac(r,Vvec),0)
    
    #Vectors on guardarem les dades:
    psivec=np.zeros((Nx+1,Nx+1,Nt+1),dtype=complex)
    normas=np.zeros((Nx+1,Nx+1,Nt+1),dtype=float)
    psivec[:,:,0]=psi
    normas[:,:,0]=norma(psi)
    
    for i in range(Nt):        
        psivec[:,:,i+1]=Crannk_step(psivec[:,:,i],avec,bxvec,byvec,cvec,r,Vvec)
        normas[:,:,i+1]=norma(psivec[:,:,i+1])
    
    return psivec,normas,tvec,Vvec
    
    
    
    
def psi0f(x,y,s2,p0):
    p0=1j*p0
    n=1./((2*np.pi*s2)**(1/2))
    a= n*np.exp(-((x)**2+(y)**2)/(4*s2))*np.exp(p0*x)
    return a






def Vfree(x,y,xb,yb):
    if abs(x)>=xb or abs(y)>=yb:
        V=10000000.
    else:
        V=0.
        
    return V
def Vbarrera(x,y,t):
    if abs(x)>=(L-t*(3.5/20.)) or abs(y)>=(L-t*(3.5/20)):
        V=1000000        
    else:
        V=0.
    return V
        

def Hx(n,r,V):
    #Generem matriu Hx
    H=np.zeros((Nx+1,Nx+1),dtype=complex)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=1j*(V[n,j]+r*2)
            elif abs(i-j)==1:
                H[i,j]=-1j*r
    return H    

def Hy(n,r,V):
    #Generem matriu Hy
    H=np.zeros((Nx+1,Nx+1),dtype=complex)
    for i in range(Nx+1):
        for j in range(Nx+1):
            if i==j:
                H[i,j]=1j*(V[i,n]+r*2)
            elif abs(i-j)==1:
                H[i,j]=-1j*r    
    return H
    

def bx(n,r,V):
    Hamp=Hx(n,r,V)+np.eye(Nx+1,dtype=complex)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])
def by(n,r,V):
    Hamp=Hy(n,r,V)+np.eye(Nx+1,dtype=complex)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if i==j])

def ac(r,V):
    Hamp=Hx(0,r,V)+np.eye(Nx+1,dtype=complex)
    return np.array([Hamp[i,j] for i in range(Nx+1) for j in range(Nx+1)
            if (i-j)==1])
    

def dx(psi,n,r,V):
    Hamm=(np.eye(Nx+1,dtype=complex))-Hx(n,r,V)
    psi=np.dot(Hamm,psi)
   
    
    return psi

def dy(psi,n,r,V):
    Hamm=(np.eye(Nx+1,dtype=complex))-Hy(n,r,V)
    psi=np.dot(Hamm,psi)    
    
    return psi



def norma(psi):
    return np.array([[np.real((psi[i,j])*np.conj(psi[i,j])) for j in range(Nx+1)] 
             for i in range(Nx+1)])

def trapezis(xa,xb,ya,yb,dx,fun):
    Nx=np.int((xb-xa)/dx)
    Ny=np.int((yb-ya)/dx)    
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*((dx**2)/2)
        
        for j in range(1,Ny):
            funsum=funsum+fun[i,j]*dx**2
    return funsum

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

    
    



L=3
tb=1.8
ta=0
hbar=1
m=1

dxvec=np.array([0.05,0.075,0.10,0.15])
dtvec=np.array([0.003,0.006,0.01])
#Generacion rapida de animación

#Nx=40
#dex=(2*L)/Nx
#Nt=np.int(tb/0.02)
#psi0=np.array([[psi0f(-L+i*dex,-L+j*dex,0.25,10) for i in range(Nx+1)]
#                    for j in range(Nx+1)])
#psivec,normas,tvec,Vvec=Crannk2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,
#                                         Vfree,hbar,m,psi0)
#dades=[L,Nx,Nt]
#np.save('normasprov.npy',normas)
#np.save('dadesprov.npy',dades)
#np.save('Vvec.npy',Vvec)


for i in range(len(dxvec)):
    for j in range(len(dtvec)):
        dex=dxvec[i]
        dt=dtvec[j]
        Nx=np.int(2*L/dex)
        Nt=np.int((tb-ta)/dt)
        
        psi0=np.array([[psi0f(-L+i*dex,-L+j*dex,0.25,10) for i in range(Nx+1)]
                    for j in range(Nx+1)])
        t_ini=time()
        psivec,normas,tvec,Vvec=Crannk2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,
                                         Vfree,hbar,m,psi0)
        t_final=time()
        tejec=t_final-t_ini
        devec=np.array([dex,dt,tejec])
        np.save('normadx{}dt{}.npy'.format(dex,dt),normas)
        np.save('psivecdx{}dt{}.npy'.format(dex,dt),psivec)
        np.save('tvecdx{}dt{}.npy'.format(dex,dt),tvec)
        np.save('Vvecdx{}dt{}.npy'.format(dex,dt),Vvec)
        np.save('dvecdx{}dt{}.npy'.format(dex,dt),devec)



#s2vec=np.array([0.4,0.6,0.8,1,1.2,1.4])
#p0vec=np.array([10,50,100,200,500,1000,10000,1000000])
#for x in range(0,6):
#    for y in range(0,8):    
#        psi0=np.array([[psi0f(-L+i*dex,-L+j*dex,s2vec[x],p0vec[y]) for i in range(Nx+1)]
#              for j in range(Nx+1)])
#        psivec,normas,tvec=Crannk2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,Vfree,hbar,m,psi0)
#        for i in range(Nt+1):
#            devec[i,x,y]=dispersiox(-L,L,Nx,normas[:,:,i])
            
#np.save('devec.npy',devec)
#np.save('p0vec.npy',p0vec)
#np.save('s2vec.npy',s2vec)
#np.save('tvec.npy',tvec)

#for i in range(0,4):
#    Nx=np.int(20)+np.int(20)*i
#    dex=np.real((2*L)/Nx)
#    psi0=np.array([[psi0f(-L+i*dex,-L+j*dex,0.25,p0vec[1]) for i in range(Nx+1)]
#              for j in range(Nx+1)])
#    psivec,normas,tvec=Crannk2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,Vfree,hbar,m,psi0)
#    np.save('psivector{}.npy'.format(i),psivec)

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
    
    r=(dt/(4*dx**2))*(hbar**2/m)
    
    #Vectors on guardarem les dades:
    psivec=np.zeros((Nx+1,Nx+1,Nt+1),dtype=complex)
    normas=np.zeros((Nx+1,Nx+1,Nt+1),dtype=float)
    Vvecgran=np.zeros((Nx+1,Nx+1,Nt+1),dtype=float)
    psivec[:,:,0]=psi
    normas[:,:,0]=norma(psi)
    Vvec=np.array([[V(xvec[i],yvec[j],xb,ta) for i in range(Nx+1)]
            for j in range(Ny+1)])
    Vvecgran[:,:,0]=Vvec[:,:]
    
    keepk=np.int(0.)
    for k in range(Nt):
        #Matriu de potencial                        
        
        #Generem vectors bx i by:
        bxvec=np.array([bx(i,r,Vvec) for i in range(Nx+1)])
        byvec=np.array([by(i,r,Vvec) for i in range(Ny+1)])
        avec=np.insert(ac(r,Vvec),0,0)
        cvec=np.append(ac(r,Vvec),0)
        
        psivec[:,:,k+1]=Crannk_step(psivec[:,:,k],avec,bxvec,byvec,cvec,r,Vvec)
        normas[:,:,k+1]=norma(psivec[:,:,k+1])
        #Valor esperat de x:
      
        s2,xesp=dispersiox(xa,xb,Nx,normas[:,:,k+1])
        
        Vvecmax=np.where(Vvec[0,:]==np.amax(Vvec[0,:]))        
        Vvecmaxs=Vvecmax[0]        
        xespV=Vvecmaxs[0]*dx+xa 

        
        if (abs(xesp)+2.9*np.sqrt(s2))>=abs(xespV):             
            Vvec=Vvec
            Vvecgran[:,:,k+1]=Vvec
            keepk=keepk+1
        else:
            Vvec=np.array([[V(xvec[i],yvec[j],xb,tvec[k+1-keepk]) 
            for i in range(Nx+1)] for j in range(Ny+1)])
            Vvecgran[:,:,k+1]=Vvec
    
            
        
        
        
        
    
    return psivec,normas,tvec,Vvecgran
    
    
    
    
def psi0f(x,y):
    p0=50j
    n=1./((2*np.pi*0.25)**(1/2))
    a= n*np.exp(-((x-1.5)**2+(y)**2)/(4*0.25))*np.exp(p0*x)
    return a






def Vfree(x,y,xb,yb):
    V=0.
    return V
def Vbarrera(x,y,L,t):
    
    if t<10:    
        if abs(x)>=(L-t*(2.25/10.)):
            V=100000000        
            else:
                V=0.
    else:
        if abs(x)>=0.75:
            V=100000000
    
    
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
        funsum=funsum+(fun[i,0]+fun[i,-1])*(dx**2)/2
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
    fun2=np.array([[fun[i,j]*((xa+dx*j)**2) for j in range(Nx+1)] 
        for i in range(Nx+1)])
    xesp2=trapezis(xa,xb,xa,xb,dx,fun2)
    
    s2=xesp2-xesp**2
    return s2,xesp



L=3
tb=20
Nt=450
Nx=40
ta=0
hbar=1
m=1
dex=np.real((2*L)/Nx)
psi0=np.array([[psi0f(-L+i*dex,-L+j*dex) for i in range(Nx+1)]
              for j in range(Nx+1)])
psi,normas,tvec,Vvec=Crannk2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,Vbarrera,hbar,m,psi0)
zero=np.zeros((Nx+1,Nx+1))
dades=np.array([L,Nx,Nt])
np.save('normasprov.npy',normas)
np.save('psiprov.npy',psi)
np.save('tvecprov.npy',tvec)
np.save('dadesprov.npy',dades)
np.save('Vvec.npy',Vvec)

#normax=np.where(normas[:,:,k+1]==np.amax(normas[:,:,k+1])) 
#        Vvecmax=np.where(Vvec==np.amax(Vvec))
#        normaxs=normax[0]+np.array(5)
#        normaxi=normax[0]-np.array(10)
#        Vvecmaxs=Vvecmax[0]
        
#        if abs(normaxs[0])>abs(Vvecmaxs[0]):
#            Vvec=Vvec
#            suma=+np.int(1)+suma
            
                        
#        Vvec=np.array([[V(xvec[i],yvec[j],xb,tvec[k+1-suma]) 
#            for i in range(Nx+1)] for j in range(Ny+1)])
            
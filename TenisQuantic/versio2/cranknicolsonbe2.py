# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

#Funció triadag
#range(n) n element pero comencem a contar desde 0
from numba import jit
import numpy as np
from time import time


@jit(nopython=True)
def tridiag(a, b, c, d):
    n = len(d)  # número de filas

    # Modifica los coeficientes de la primera fila
    cp= np.zeros(n,dtype=np.complex128)
    dp=np.zeros(n,dtype=np.complex128)
    cp[0]=c[0]/b[0]  # Posible división por cero
    dp[0]=d[0]/ b[0]

    for i in range(1, n):
        ptemp = b[i] - (a[i] * cp[i-1])
        cp[i]= c[i]/ptemp
        dp[i] = (d[i] - a[i] * dp[i-1])/ptemp

    # Sustitución hacia atrás
    x = np.zeros(n,dtype=np.complex128)
    x[n-1] = dp[n-1] #-1 vol dir l'ultim element

    for j in range(1,n):
        i = (n-1)-j
        x[i] = dp[i]-cp[i]*x[i+1]

    return x


#Modifiquem el Crankk_Step per la part d'animació. Fem una funció que nomes retorni
#mitj pas....
@jit(nopython=True)
def Crannk_stepm(psi,avec,cvec,r,V,dt,Nx,l):
    """Realitza un pas de Crank-Nicolson per un dt."""
    
    #Primer itera per psi[i,:], és a dir, prenent psi[i,:] com
    #únic element de psi i aplicant-li un pas de Crannk-Nicolson.
    #Tot seguit segueix iterant per tot i.
    psip=np.copy(psi)
   
    if (l%2)==0:    
        for k in range(0,Nx+1):
            psi[k,:]=tridiag(avec,bx(k,r,V,dt,Nx),cvec,dvecy(psip,k,r,V,dt,Nx))
       
    #Fa el mateix per les psi[:,j]
    else:   
        for k in range(0,Nx+1):
            psi[:,k]=tridiag(avec,by(k,r,V,dt,Nx),cvec,dvecx(psip,k,r,V,dt,Nx)) 
                
    return psi


    
def psi0f(x,y,s2,p0x,p0y,x0,y0):
    n=1./((2*np.pi*s2)**(1/2))
    a= n*np.exp(-((x-x0)**2+(y-y0)**2)/(4*s2))*np.exp(1j*p0x*(x-x0)+1j*p0y*(y-y0))
    return a



@jit
def bx(n,r,V,dt,Nx):
    Hamp=np.zeros((Nx+1),dtype=np.complex128)
    Hamp=1j*((dt/4.)*V[n,:]+r*2)+1.

    return Hamp
@jit
def by(n,r,V,dt,Nx):
    Hamp=np.zeros((Nx+1),dtype=np.complex128)
    Hamp=1j*((dt/4.)*V[:,n]+r*2)+1.

    return Hamp
@jit
def ac(r,V,Nx):
    a=np.full(Nx+1,-1j*r,dtype=np.complex128)
    a[0]=0.
    c=np.full(Nx+1,-1j*r,dtype=np.complex128)
    c[Nx]=0.
    return a,c
@jit
def dvecx(psi,n,r,V,dt,Nx):
    #Aquí li assignarem un vector de la matriu psi, de fet, el psi[:,n],
    #el vector columna(variant x, deixant y invariant). n indicara de quina
    #columna estem parlant.
    Hamm=np.zeros((Nx+1),dtype=np.complex128)
    #Cream la matriu que el contendra:
    if (n>0 and n<(Nx)):  
        Hamm=(1.-1j*((dt/4.)*V[:,n] + 2.*r))*psi[:,n]+1j*r*(psi[:,-1+n]+psi[:,1+n])            
    elif n==0:
        Hamm=(1.-1j*((dt/4.)*V[:,n] + 2.*r))*psi[:,n]+1j*r*(psi[:,1+n])
    else:
        Hamm=(1.-1j*((dt/4.)*V[:,n] + 2.*r))*psi[:,n]+1j*r*(psi[:,-1+n])
   
   #Tot seguït, multipliquem els psi
    return Hamm

@jit
def dvecy(psi,n,r,V,dt,Nx):
     #Aquí li assignarem un vector de la matriu psi, de fet, el psi[n,:],
    #el vector fila(variant y, deixant x invariant). n indicara de quina
    #columna estem parlant.
    Hamm=np.zeros((Nx+1),dtype=np.complex128)
    #Cream la matriu que el contendra:
    if (n>0 and n<(Nx)):  
        Hamm=(1.-1j*((dt/4.)*V[n,:] + 2.*r))*psi[n,:] +1j*r*(psi[-1+n,:]+psi[1+n,:])
    
    elif n==0:        
        Hamm=(1.-1j*((dt/4.)*V[n,:] + 2.*r))*psi[n,:]+1j*r*(psi[1+n,:])
    else:      
        Hamm=(1.-1j*((dt/4.)*V[n,:] + 2.*r))*psi[n,:]+1j*r*(psi[-1+n,:])
   #Tot seguït, multipliquem els psi

    return Hamm



@jit
def norma(psi,Nx):
    norma=np.real(psi*np.conj(psi))
    #return np.array([[np.real((psi[i,j])*np.conj(psi[i,j])) for j in range(Nx+1)] 
    #         for i in range(Nx+1)])
    return norma

@jit
def trapezis(xa,xb,dx,fun):
    Nx=np.int((xb-xa)/dx)  
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*(dx**2)/2
        for j in range(1,Nx):
            funsum=funsum+fun[i,j]*dx**2
    return funsum
@jit
def Ham(xa,xb,dx,psi):
    #Matriu que ens calcula la energia cinetica total.
    Ec=np.zeros((np.shape(psi)),dtype=np.complex128)
    Nx=np.int((xb-xa)/dx)  
    s=(1/2.)*(1./dx**2)
    #Generem la matriu del potencial
    for i in range(1,Nx):
        for j in range(1,Nx):
            Ec[i,j]=-s*(psi[i+1,j]+psi[i-1,j]+psi[i,j-1]
                +psi[i,j+1]-4*psi[i,j])
            Ec[i,j]=np.real(Ec[i,j]*np.conj(psi[i,j]))
        
    Ecfinal=trapezis(xa,xb,dx,Ec)
    Ecfinal=np.real(Ecfinal)
    return Ecfinal
    


    




def psi0harm(x,y,w,m,hbar):
    c=np.sqrt((m*w)/(hbar*np.pi))
    psi0=c*np.exp(-0.5*(w*m)*(1/hbar)*((x-1.5)**2+y**2))

    return psi0

def Vharm(x,y,xb,yb):
   
   V=0.

   return V

def Vbarrera(x,y,xb,yb):
    
    #if abs(x)>=xb or abs(y)>=yb:
    #    V=10000000.
    
    if x<=0.1 and abs(y)<=0.3:
        V=10000000

    else:
        V=0.
    return V
    
    

if __name__=='__main__':
    
        pass

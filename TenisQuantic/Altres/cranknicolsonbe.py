# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

#Funció triadag
#range(n) n element pero comencem a contar desde 0

import numpy as np
from time import time
from numba import jit

@jit
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
    x[n-1] = dp[n-1] #-1 vol dir l'ultim element

    for j in range(1,n):
        i = (n-1)-j
        x[i] = dp[i]-cp[i]*x[i+1]

    return x

#Pas de CrannkNicolson
@jit
def Crannk_step(psi,avec,bxvec,byvec,cvec,r,V):
    """Realitza un pas de Crank-Nicolson per un dt."""
    
    #Primer itera per psi[i,:], és a dir, prenent psi[i,:] com
    #únic element de psi i aplicant-li un pas de Crannk-Nicolson.
    #Tot seguit segueix iterant per tot i.    
    psip=np.copy(psi)
    for k in range(0,Nx+1):
        dvec=dvecy(psip,k,r,V)
        psi[k,:]=tridiag(avec,bxvec[k],cvec,dvec)
    

    #Fa el mateix per les psi[:,j]
        
    for k in range(0,Nx+1):
        dvec=dvecx(psi,k,r,V)
        psip[:,k]=tridiag(avec,byvec[k],cvec,dvec)    
    return psip


    
#@jit
def Crannk2D(xa,xb,ya,yb,ta,tb,Nx,Ny,Nt,V,hbar,m,psi):
    "Crank-Nicolson 2D"
    
    
    #Generem tots els números que utilitzarem
    dx=(xb-xa)/np.float(Nx)
    dy=(yb-ya)/np.float(Ny)
    #Dividim entre dos per què cada pas de Crannk Nicolson requereix
    #primer mig pas dt/2, i després un altre dt/2
    dt=(tb-ta)/(np.float(Nt))
    xvec=np.array([xa+i*dx for i in range(Nx+1)])
    yvec=np.array([ya+i*dy for i in range(Ny+1)])
    tvec=np.array([ta+dt*i for i in range(Nt+1)])
    #Matriu de potencial
    Vvec=np.array([[V(xvec[i],yvec[j],xb,yb) for i in range(Nx+1)]
        for j in range(Ny+1)],dtype=np.float64)

#Crec que aquí falta un 4...., ho provo
    #He de pensar perque... no se que passa
    
    r=(dt/(4*dx**2))*(hbar/m)

    #Generem vectors bx i by:
    bxvec=np.array([bx(i,r,Vvec) for i in range(Nx+1)])
    byvec=np.array([by(i,r,Vvec) for i in range(Ny+1)])
    print(np.shape(bxvec))
    avec,cvec=ac(r,Vvec)
    print(np.shape(avec))
    #Vectors on guardarem les dades:
    psivec=np.zeros((Nx+1,Nx+1,Nt+1),dtype=np.complex128)
    normas=np.zeros((Nx+1,Nx+1,Nt+1),dtype=np.float64)
    psivec[:,:,0]=psi
    normas[:,:,0]=norma(psi)
    
    for i in range(Nt): 
        #Això aplica 2dt, es a dir, fa un pas de temps i després un altre.
        #El conjunt sera un dt.
        psivecnew=Crannk_step(psivec[:,:,i],avec,bxvec,byvec,cvec,r,Vvec)
        psivec[:,:,i+1]=psivecnew
        normas[:,:,i+1]=norma(psivecnew)
    
    return psivec,normas,tvec,Vvec
    
    
    
    
def psi0f(x,y,s2,p0x,p0y,*largs):
    n=1./((2*np.pi*s2)**(1/2))
    a= n*np.exp(-((x)**2+(y)**2)/(4*s2))*np.exp(1j*p0x*x+1j*p0y*y)
    return a




def Vfree(x,y,xb,yb):
    if abs(x)>=xb or abs(y)>=yb:
        V=0
    else:
        V=0.
        
    return V
def Vbarrera(x,y,t):
    if abs(x)>=(L-t*(3.5/20.)) or abs(y)>=(L-t*(3.5/20)):
        V=1000000        
    else:
        V=0.
    return V
        
@jit
def bx(n,r,V):
    Hamp=np.zeros((Nx+1),dtype=np.complex128)
    Hamp=1j*((dt/4.)*V[n,:]+r*2)+1.

    return Hamp
@jit
def by(n,r,V):
    Hamp=np.zeros((Nx+1),dtype=np.complex128)
    Hamp=1j*((dt/4.)*V[:,n]+r*2)+1.

    return Hamp
@jit
def ac(r,V):
    a=np.full(Nx+1,-1j*r,dtype=np.complex128)
    a[0]=0.
    c=np.full(Nx+1,-1j*r,dtype=np.complex128)
    c[Nx]=0.
    return a,c
@jit
def dvecx(psi,n,r,V):
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
def dvecy(psi,n,r,V):
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


def psi0harm(x,y,w,m,hbar):
    c=np.sqrt((m*w)/(hbar*np.pi))
    psi0=c*np.exp(-0.5*(w*m)*(1/hbar)*((x)**2+y**2))

    return psi0

def Vharm(x,y,xb,yb):
   if abs(x)>=xb or abs(y)>=yb:
       V=10000000.
   else:
        
        w=2
        m=1.
        V=0.5*(w**2)*(x**2+y**2)
        V=0.

   return V
    

if __name__=='__main__':

    L=3.
    tb=0.5
    ta=0
    hbar=1.
    m=1.
    
    dxvec=np.array([0.03])
    dtvec=np.array([0.01])
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
    
    
    #for i in range(len(dxvec)):
    #    for j in range(len(dtvec)):
    #        dex=dxvec[i]
    #        dt=dtvec[j]
    #        Nx=np.int(2*L/dex)
    #        Nt=np.int((tb-ta)/dt)
    #       
    #        psi0=np.array([[psi0f(-L+i*dex,-L+j*dex,0.25,10) for i in range(Nx+1)]
    #                    for j in range(Nx+1)])
    #        t_ini=time()
    #        psivec,normas,tvec,Vvec=Crannk2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,
    #                                         Vfree,hbar,m,psi0)
    ##        t_final=time()
    #        tejec=t_final-t_ini
    #        devec=np.array([dex,dt,tejec])
    #        np.save('normadx{}dt{}.npy'.format(dex,dt),normas)
    #        np.save('psivecdx{}dt{}.npy'.format(dex,dt),psivec)
    #        np.save('tvecdx{}dt{}.npy'.format(dex,dt),tvec)
    #        np.save('Vvecdx{}dt{}.npy'.format(dex,dt),Vvec)
    #        np.save('dvecdx{}dt{}.npy'.format(dex,dt),devec)
    
    dex=dxvec[0]
    dt=dtvec[0]
    Nx=np.int(2*L/dex)
    Nt=np.int((tb-ta)/dt)
    xb=L
    yb=L
    psi0=np.array([[psi0f(-L+i*dex,-L+j*dex,0.25,5.) for i in range(Nx+1)]
                        for j in range(Nx+1)],dtype=np.complex128)
    Vvece=np.array([[Vharm(-L+i*dex,-L+j*dex,xb,yb) for i in range(Nx+1)]
            for j in range(Nx+1)])
    r=dt/(4*dex**2)
    dxvector=dvecx(psi0,2,r,Vvece)   
    
        
    t_ini=time()
    psivec,normas,tvec,Vvec=Crannk2D(-L,L,-L,L,ta,tb,Nx,Nx,Nt,
                                             Vharm,hbar,m,psi0)
    t_final=time()
    tejec=t_final-t_ini
    devec=np.array([dex,dt,tejec])
    np.save('normaharmdx{}dt{}.npy'.format(dex,dt),normas)
    np.save('psiharmdx{}dt{}.npy'.format(dex,dt),psivec)
    np.save('tvecharmdx{}dt{}.npy'.format(dex,dt),tvec)
    np.save('Vvecharmdx{}dt{}.npy'.format(dex,dt),Vvec)
    np.save('dvecharmdx{}dt{}.npy'.format(dex,dt),devec)
    
    print(trapezis(-L,L,-L,L,dex,normas[:,:,0]))
    print(trapezis(-L,L,-L,L,dex,normas[:,:,40]))
    
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
    

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 18:23:28 2021

@author: llucv
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 07:45:28 2021

@author: llucv
"""

from numba import jit
import numpy as np


hbar=1
pi=np.pi


@jit(nopython=True)
def tridiag(A,B,C,D):
    N=len(D)
    x=np.zeros((N),dtype=np.complex128)
    
    for i in range(1,N):
        W=A[i]/B[i-1]
        B[i] = B[i] - W * C[i - 1]
        D[i] = D[i] - W * D[i - 1]
    
    x[N-1]=D[N-1]/B[N-1]
    
    for j in range(1,N):
        i=N-1-j
        x[i] = (D[i] - C[i] * x[i + 1]) / B[i]
        
    return x

@jit
def psi_ev_ck(psip,V,r,dl,dt):
    box_shape=(np.shape(psip))
    psi=np.zeros((box_shape[0],box_shape[1],3),dtype=np.complex128)
    psi[:,:,0]=psip[:,:]
    
    """primer mig pas de temps"""        
    #calcul de la psi passat el primer mig pas
    #vector D
    D=np.zeros((box_shape[0],box_shape[1]),dtype=np.complex128)
    
    D[:,0]=psi[:,0,0]*(1-2*1j*r-1j*dt*V[:,0]/4)+1j*r*(psi[:,1,0])
    
    D[:,box_shape[1]-1]=psi[:,box_shape[1]-1,0]*(1-1j*2*r-1j*dt*\
                            V[:,box_shape[1]-1]/4)+\
                            1j*r*(psi[:,box_shape[1]-2,0])
                            
    D[:,1:box_shape[1]-1]=psi[:,1:box_shape[1]-1,0]*\
        (1-1j*2*r-1j*dt*V[:,1:box_shape[1]-1]/4)+\
        1j*r*(psi[:,0:box_shape[1]-2,0]+psi[:,2:box_shape[1],0])
    
    #altres vectors (coeficients)
    A=np.zeros((box_shape[0]),dtype=np.complex128)
    A[1:]=-1j*r
    A[0]=0.
    
    C=np.zeros((box_shape[0]),dtype=np.complex128)
    C[0:-1]=-1j*r
    C[box_shape[0]-1]=0.
    
    B=np.zeros((box_shape[0],box_shape[1]),dtype=np.complex128)
    B[:,:]=1+1j*2*r+1j*dt*V[:,:]/4
    
    for j in range(box_shape[1]):
        psi[:,j,1]=tridiag(A,B[:,j],C,D[:,j])
        
    
    """segon mig pas de temps"""
    
    #calcul de la psi passat el segon mig pas
    #vector D
    D=np.zeros((box_shape[0],box_shape[1]),dtype=np.complex128)
    
    D[0,:]=psi[0,:,1]*(1-1j*2*r-1j*dt*V[0,:]/4)+1j*r*(psi[1,:,1])
    
    D[box_shape[0]-1,:]=psi[box_shape[0]-1,:,1]*(1-1j*2*r-1j*dt*\
                        V[box_shape[0]-1,:]/4)+\
                        1j*r*(psi[box_shape[0]-2,:,1])

    D[1:box_shape[0]-1,:]=psi[1:box_shape[0]-1,:,1]*\
            (1-1j*2*r-1j*dt*V[1:box_shape[0]-1,:]/4)+\
            1j*r*(psi[0:box_shape[0]-2,:,1]+psi[2:box_shape[0],:,1])
    
    #altres vectors (coeficients A,B i C)
    A=np.zeros((box_shape[1]),dtype=np.complex128)
    A[1:]=-1j*r
    A[0]=0.
    
    C=np.zeros((box_shape[1]),dtype=np.complex128)
    C[0:-1]=-1j*r
    C[box_shape[1]-1]=0.
    
    B=np.zeros((box_shape[0],box_shape[1]),dtype=np.complex128)
    B[:,:]=1+1j*2*r+1j*dt*V[:,:]/4
    
    for i in range(box_shape[0]):
        psi[i,:,2]=tridiag(A,B[i,:],C,D[i,:])
    
    psip[:,:]=psi[:,:,2]
    return psip


def psi_0(Nx,Ny,x0,y0,px0,py0,dev,dl,dt):
    gauss_x=np.zeros((Nx),dtype=np.complex128)
    gauss_y=np.zeros((Ny),dtype=np.complex128)
    x=np.linspace(0,Nx,Nx,endpoint=False)
    y=np.linspace(0,Ny,Ny,endpoint=False)
    
    x=x*dl
    y=y*dl
    
    x0=x0*dl
    y0=y0*dl
    
    dev=dev*dl
    
    const=(1/(2*pi*dev**2))**(1/4)
    gauss_x[:]=const*np.exp(-(((x[:]-x0)**2)/(dev**2))/4+\
                               1j*px0*(x[:]-x0/2)/hbar)
    gauss_y[:]=const*np.exp(-(((y[:]-y0)**2)/(dev**2))/4+\
                               1j*py0*(y[:]-y0/2)/hbar)
    
    psi=np.tensordot(gauss_x,gauss_y,axes=0)
    
    return psi

@jit
def prob_dens(psi):
    prob=np.real(psi*np.conj(psi))
    return prob


def Potential_slits_gauss(max_V,x_wall,h_slits,
                          separation,w_slt,dev,Nslt,dl,Nx,Ny):
    slt_i=np.zeros((Nslt+2),dtype=int)
    slt_f=np.zeros((Nslt+2),dtype=int)
    slt_n=np.linspace(1,Nslt,Nslt,dtype=int)
    dev=dev*dl
    
    if Nslt==2:
        #posicio del final i l'inici de cada escletxa, ara amb separació variable
        slt_i[1]=int(Ny/2)-int(separation/2)-\
            int(w_slt/2)
        slt_i[2]=int(Ny/2)+int(separation/2)-\
            int(w_slt/2)
        slt_f[1]=int(Ny/2)-int(separation/2)+\
            int(w_slt/2)
        slt_f[2]=int(Ny/2)+int(separation/2)+\
            int(w_slt/2)
    
        slt_f[0]=0
        slt_f[Nslt+1]=Ny-1
        slt_i[Nslt+1]=Ny-1
    
    else:
        #posicio del final i l'inici de cada escletxa
        slt_i[1:Nslt+1]=int(Ny/2)-int(h_slits/2)-\
            int(w_slt/2)+slt_n[:]*int(h_slits/(1+Nslt))
        slt_f[1:Nslt+1]=int(Ny/2)-int(h_slits/2)+\
            int(w_slt/2)+slt_n[:]*int(h_slits/(1+Nslt))
            
        slt_f[0]=0
        slt_f[Nslt+1]=Ny-1
        slt_i[Nslt+1]=Ny-1
        
    x=np.linspace(0,Nx,Nx,endpoint=False,dtype=np.complex128)
    y=np.linspace(0,Ny,Ny,endpoint=False,dtype=np.complex128)
    
    x=x*dl
    y=y*dl
    
    beta=dev*np.sqrt(2*pi)
    alpha=beta*np.sqrt(max_V)
    
    V_y=np.zeros((Ny),dtype=np.complex128)
    V_x=np.zeros((Nx),dtype=np.complex128)
    
    x0=x_wall*dl
    V_x[:]=alpha*np.exp(-(((x[:]-x0)/dev)**2)/2)/beta
    
    for n in range(1,Nslt+2):
        V_y[slt_f[n-1]:slt_i[n]]=alpha/beta
        
        V_y[slt_i[n]:slt_f[n]]=\
            alpha\
   *np.exp(-(((y[slt_i[n]:slt_f[n]]-y[slt_i[n]])/dev)**2)/2)\
            /beta+\
            alpha\
   *np.exp(-(((-y[slt_i[n]:slt_f[n]]+y[slt_f[n]])/dev)**2)/2)\
            /beta
        
        V_y[Ny-1]=alpha/beta
        V_y[Ny-2]=alpha/beta

    V=np.tensordot(V_x,V_y,axes=0)


    return V


def mirror_pot(Nx,Ny,x0,y0,dl,pot_max,lenght):
    x=np.linspace(0,Nx,Nx,endpoint=False,dtype=np.complex128)
    y=np.linspace(0,Ny,Ny,endpoint=False,dtype=np.complex128)
    
    x=x*dl
    y=y*dl
    dev=dl
    
    x0=x0*dl
    y0=y0*dl
    
    beta=dev*np.sqrt(2*pi)
    alpha=beta*np.sqrt(pot_max)
    
    V_y=np.zeros((Ny),dtype=np.complex128)
    V_x=np.zeros((Nx),dtype=np.complex128)
    
    V_y[:]=alpha*np.exp(-(((y[:]-y0)/dev)**2)/2)/beta
    
    x0i=int(x0/dl)
    V_x[x0i-int(lenght/2):x0i+int(lenght/2)]=alpha/beta
    V_x[:x0i-int(lenght/2)]=alpha*np.exp(-(((x[:x0i-int(lenght/2)]-x0)\
                                            /dev)**2)/2)/beta
    V_x[x0i+int(lenght/2):]=alpha*np.exp(-(((x[x0i+int(lenght/2):]-x0)\
                                            /dev)**2)/2)/beta
    
    V=np.tensordot(V_x,V_y,axes=0)
    return V

@jit
def trapezis_2D(f,h):
    f_shape=np.shape(f)
    suma=0
    for i in range(f_shape[0]-1):
        suma=((np.sum(f[i+1,0:-1])+np.sum(f[i+1,1:]))*h/2\
             +(np.sum(f[i,0:-1])+np.sum(f[i,1:]))*h/2)*h/2+\
                 suma
        
    return suma

@jit
def normalize(psi,box_shape,dl):
    prob=np.zeros((box_shape[0],box_shape[1]),dtype=np.complex128)
    prob[:,:]=prob_dens(psi[:,:])
    psi=psi/trapezis_2D(psi,1)
    return psi
    
def coulomb_pot(Nx,Ny,dl,x0,y0,Z,sign):
    V=np.zeros((Nx,Ny),dtype=np.complex128)

    x=np.linspace(0,Nx,Nx,endpoint=False,dtype=np.complex128)
    y=np.linspace(0,Ny,Ny,endpoint=False,dtype=np.complex128)
    
    x=x*dl
    y=y*dl
    
    alpha=1
    
    r_a=3.11*(10**(-3))*((2*Z)**(1/3))
    
    for i in range(Nx):
        for j in range(Ny):
            r=np.sqrt((x[i]-x0)**2+(y[j]-y0)**2)
            if r<r_a:
                V[i,j]=1000
            else:
                V[i,j]=sign*alpha*Z/r
    
    return V

@jit
def young_slit_wall(Nslt,w_slt,separation,h_display,Ny):
    slt_i=np.zeros((Nslt+2))
    slt_f=np.zeros((Nslt+2))
    slt_n=np.linspace(1,Nslt,Nslt)
    wall_presence=np.zeros((Ny))
    
    if Nslt==2:
        #posicio del final i l'inici de cada escletxa, ara amb separació variable
        slt_i[1]=int(Ny/2)-int(separation/2)-\
            int(w_slt/2)
        slt_i[2]=int(Ny/2)+int(separation/2)-\
            int(w_slt/2)
        slt_f[1]=int(Ny/2)-int(separation/2)+\
            int(w_slt/2)
        slt_f[2]=int(Ny/2)+int(separation/2)+\
            int(w_slt/2)

        slt_f[0]=0
        slt_f[Nslt+1]=Ny
        slt_i[Nslt+1]=Ny
    
    else:
        #posicio del final i l'inici de cada escletxa
        slt_i[1:Nslt+1]=int(Ny/2)-int(h_display/2)-\
            int(w_slt/2)+slt_n[:]*int(h_display/(1+Nslt))
        slt_f[1:Nslt+1]=int(Ny/2)-int(h_display/2)+\
            int(w_slt/2)+slt_n[:]*int(h_display/(1+Nslt))
        slt_f[0]=0
        slt_f[Nslt+1]=Ny
        slt_i[Nslt+1]=Ny
    
           
    #les escletxes van de splt_i a splt_f-1, en aquests punts no hi ha paret,
    # a slpt_f ja hi ha paret
    for n in range(1,Nslt+2):
        wall_presence[slt_f[n-1]:slt_i[n]]=1
        wall_presence[slt_i[n]:slt_f[n]]=0
    
    return wall_presence


def young_slit_sgm(wall_presence,w_wall,x_wall,sgm_max,m,Nx,Ny):
    sgm_wall=np.zeros((Nx,Ny))
    # matriu que, amb el gruix de la paret com a nombre de files, ens diu si 
    # hi ha paret o escletxes a cada una de les y(representades en les columnes)
    wall_presence=np.tile(np.array([wall_presence]),
                          (w_wall,1))

    #matriu que diu com de "dins" som a la paret
    wall_n=np.linspace(1,w_wall,w_wall)
    wall_ny=np.tile(np.array([wall_n]).transpose(),(1,Ny))

    #valors de coeficient d'absorció a les parets
    sgm_wall[x_wall-w_wall:x_wall,:]=wall_presence[:,:]\
                *sgm_max*((wall_ny[:,:])/w_wall)**m
    
            
    #llista per a l'última capa de la paret, on l'amplitud d'ona és 0
    wave_presence=np.ones((Nx,Ny))
    wave_presence[x_wall,:]=(1-wall_presence[0,:])
                
    return sgm_wall,wave_presence
    
    
        
    

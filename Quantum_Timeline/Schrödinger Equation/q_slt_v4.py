# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 07:45:28 2021

@author: llucv
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

Nx=200
Ny=200

h_display=int(Ny)
w_display=int(Nx)
h_slits=int(Ny/3)

hbar=1
pi=np.pi

psi=np.zeros((Nx,Ny,3),dtype=np.complex128)
V=np.zeros((Nx,Ny))

dt=1/100
dl=7*dt
r=dt/(4*dl*dl)


@jit(nopython=True)
def tridiag (a,b,c,d):
    n = len(a)

    cp = np.zeros(n, dtype = np.complex128)
    dp = np.zeros(n, dtype = np.complex128)
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    for i in range(1,n):
        m = (b[i]-a[i]*cp[i-1])
        cp[i] = c[i]/m
        dp[i] = (d[i] - a[i]*dp[i-1])/m

    x = np.zeros(n, dtype = np.complex128)
    x[n-1] = dp[n-1]

    for j in range(1,n):
        i = (n-1)-j
        x[i] = dp[i]-cp[i]*x[i+1]

    return x

@jit
def psi_ev_ck(psi,V,r,dl,dt):
    box_shape=(np.shape(psi))
    
    """primer mig pas de temps"""        
    #calcul de la psi passat el primer mig pas
    for j in range(box_shape[1]):
        #vector D
        D=np.zeros((box_shape[0],box_shape[1]),dtype=np.complex128)
        
        D[:,0]=psi[:,0,0]*(1-2*1j*r-1j*dt*V[:,0]/4)+1j*r*(psi[:,1,0])
        
        D[:,box_shape[1]-1]=psi[:,box_shape[1]-1,0]*(1-1j*2*r-1j*dt*\
                                V[:,box_shape[1]-1]/4)+\
                                1j*r*(psi[:,box_shape[1]-1,0])
                                
        D[:,1:box_shape[1]-1]=psi[:,1:box_shape[1]-1,0]*\
            (1-1j*2*r-1j*dt*V[:,1:box_shape[1]-1]/4)+\
            1j*r*(psi[:,0:box_shape[1]-2,0]+psi[:,2:box_shape[1],0])
        
        #altres vectors (coeficients)
        A=np.zeros((box_shape[0]),dtype=np.complex128)
        A[1:]=-1j*r
        
        C=np.zeros((box_shape[0]),dtype=np.complex128)
        C[0:-1]=-1j*r
        
        B=np.zeros((box_shape[0]),dtype=np.complex128)
        B[:]=1+1j*2*r+1j*dt*V[:,j]/4
        
        psi[:,j,1]=tridiag(A,B,C,D[:,j])
        
    """segon mig pas de temps"""
    
    #calcul de la psi passat el segon mig pas
    for j in range(box_shape[0]):
        #vector D
        D=np.zeros((box_shape[0],box_shape[1]),dtype=np.complex128)
        
        D[0,:]=psi[0,:,1]*(1-1j*2*r-1j*dt*V[0,:]/4)+1j*r*(psi[1,:,1])
        
        D[box_shape[0]-1,:]=psi[box_shape[0]-1,:,1]*(1-1j*2*r-1j*dt*\
                            V[box_shape[0]-1,:]/4)+\
                            1j*r*(psi[box_shape[0]-1,:,1])

        D[1:box_shape[0]-1,:]=psi[1:box_shape[0]-1,:,1]*\
                (1-1j*2*r-1j*dt*V[1:box_shape[0]-1,:]/4)+\
                1j*r*(psi[0:box_shape[0]-2,:,1]+psi[2:box_shape[0],:,1])
        
        #altres vectors (coeficients A,B i C)
        A=np.zeros((box_shape[1]),dtype=np.complex128)
        A[1:]=-1j*r
        A[0]=0
        
        C=np.zeros((box_shape[1]),dtype=np.complex128)
        C[0:-1]=-1j*r
        C[box_shape[1]-1]=0
        
        B=np.zeros((box_shape[1]),dtype=np.complex128)
        B[:]=1+1j*2*r+1j*dt*V[j,:]/4
        
        psi[j,:,2]=tridiag(A,B,C,D[j,:])
    
    psi[:,:,0]=psi[:,:,2]
    psi[:,:,1]=0
    psi[:,:,2]=0
    return psi


def psi_0(Nx,Ny,x0,y0,px0,py0,dev,dl,dt):
    gauss_x=np.zeros((Nx),dtype=np.complex128)
    gauss_y=np.zeros((Ny),dtype=np.complex128)
    x=np.linspace(0,Nx,Nx,endpoint=False)
    y=np.linspace(0,Ny,Ny,endpoint=False)
    
    x=x*dl
    y=y*dl
    
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


def Potential_slits_gauss(max_V,x_wall,separation,w_slt,dev,Nslt,dl,Nx,Ny):
    slt_i=np.zeros((Nslt+2),dtype=int)
    slt_f=np.zeros((Nslt+2),dtype=int)
    slt_n=np.linspace(1,Nslt,Nslt,dtype=int)
    
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

    
    plt.plot(np.real(V_y))
    plt.savefig('V_y.png')
    V=np.tensordot(V_x,V_y,axes=0)
    print(V[x_wall,10])

    return V
        

Nt=400
prob=np.zeros((Nx,Ny,Nt))
V=np.zeros((Nx,Ny))
x0_i=int(Nx/6)
y0_j=int(Ny/2)
x0=x0_i*dl
y0=y0_j*dl
px0=10
py0=0
dev=6*dl

print('som-hi')
    


Nslt=0
w_slt=15
separation=32
x_wall=int(w_display/2)
max_V=10
max_V_det=100
devV=dl*3

V=Potential_slits_gauss(max_V,x_wall,separation,w_slt,devV,Nslt,dl,Nx,Ny)
V=np.zeros((Nx,Ny),dtype=np.complex128)


print("let's animate!")
start=time.time()

psi[:,:,0]=psi_0(Nx,Ny,x0,y0,px0,py0,dev,dl,dt)
prob[:,:,0]=prob_dens(psi[:,:,0])
print(np.sum(prob[:,:,0]))

for k in range(Nt):
    print(k)
    psi=psi_ev_ck(psi,V,r,dl,dt)
    prob[:,:,k]=prob_dens(psi[:,:,0])
    print(np.sum(prob[:,:,k]))
    
print(time.time()-start)
    
comap = plt.get_cmap('Reds')
comap.set_under('k', alpha=0)
Pot=np.real(V)
Pot_max=np.max(Pot)

print("let's animate!")
def update9(frame):
    k=frame*4
    print(frame)
    normk=prob[:,:,k]
    plt.imshow(normk.transpose()[int((Ny-h_display)/2):\
                                int((Ny+h_display)/2),
                                0:w_display],origin='lower',cmap="Blues",
               vmax=prob[x0_i,y0_j,0]/10,vmin=0,alpha=1,
        extent=(0,int(w_display*dl),0,int(h_display*dl)))
        
    plt.imshow(Pot.transpose()[int((Ny-h_display)/2):\
                                int((Ny+h_display)/2),
                                0:w_display],
               origin='lower',cmap=comap,vmin=0.5*Pot_max,alpha=0,
        extent=(0,int(w_display*dl),0,int(h_display*dl)))
    

fig9 = plt.figure()
ax1 = plt.subplot()

Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim9 = ani.FuncAnimation(fig9, update9, 
                               frames = int(Nt/4)-1, 
                               blit = False, interval=200)

guarda9=1
if guarda9==1:
    anim9.save(str(Nslt)+'_slits_Sch_v4_2r.mp4', writer=writer)
    
V=np.real(V)
print(V)
plt.imshow(np.transpose(V))
plt.savefig(str(Nslt)+"V_slits_2r.png")

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
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

Nx=400
Ny=200

h_display=int(Ny)
w_display=int(Nx)
h_slits=int(Ny/3)

hbar=1
pi=np.pi

central=1

dt=1/(30*(abs(central-1)+central*10))
dl=np.sqrt(dt/2)
print(dl)
Nr=8
r=dt/(Nr*dl**2)


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
            


psi=np.zeros((Nx,Ny),dtype=np.complex128)
V=np.zeros((Nx,Ny))

Nt=10000
prob=np.zeros((Nx,Ny,Nt))
x0_i=int(0.45*Nx)
y0_j=int(Ny/2)
x0=x0_i*dl
y0=y0_j*dl
px0=0.5
py0=0
Ndev=10
dev=Ndev*dl

print('som-hi')
    


Nslt=2
w_slt=15
separation=32
x_wall=int(w_display/2)

#potencial
xv0_i=int(0.55*Nx)
yv0_j=int(Ny/2)
xv0=xv0_i*dl
yv0=yv0_j*dl
max_V=100
Z=10
sign=-1
devV=dl

if central==0:
    opac_k=1
    V=Potential_slits_gauss(max_V,x_wall,separation,w_slt,devV,Nslt,dl,Nx,Ny)

if central==1:
    opac_k=0.05
    V=coulomb_pot(Nx,Ny,dl,xv0,yv0,Z,sign)


print("let's compute!")
start=time.time()

psi[:,:]=psi_0(Nx,Ny,x0,y0,px0,py0,dev,dl,dt)
prob[:,:,0]=prob_dens(psi[:,:])
print(np.sum(prob[:,:,0]))
prob_k=np.zeros((Nx,Ny))

for k in range(Nt):
    print(k)
    psi=psi_ev_ck(psi,V,r,dl,dt)
    prob[:,:,k]=prob_dens(psi[:,:])
    prob_k[:,:]=prob[:,:,k]
    print(trapezis_2D(prob_k,dl))
    
print(time.time()-start)

Pot=np.real(V)

"""
for i in range(Nx):
    for j in range(Ny):
        if Pot[i,j]-1000==0:
            Pot[i,j]=0.
"""

print('dl')
print(dl)

print('potencial0')
print(np.max(Pot))
print(np.min(Pot))

if central==0:   
    comap = plt.get_cmap('Reds')
    comap.set_under('k', alpha=0)
    Pot_max=np.max(Pot)

    
if central==1:
    
    if np.max(Pot)<=0:
        comap = plt.get_cmap('Reds')
        Pot=-Pot
        Pot_max=np.max(Pot)
    
    else:
        comap = plt.get_cmap('Reds')
        Pot_max=np.max(Pot)



print('potencial')
print(Pot_max)
if Pot_max > 0.1:
    opac=0.5/1*opac_k
    
if Pot_max < 0.1:
    opac=0/1

print("let's animate!")

Nvisu=1
Nden=50
def update9(frame):
    k=frame*Nden
    print(frame)
    normk=prob[:,:,k]
    plt.imshow(normk.transpose()[int((Ny-h_display)/2):\
                                int((Ny+h_display)/2),
                                0:w_display],origin='lower',cmap="Blues",
               vmax=prob[x0_i,y0_j,0]/Nvisu,vmin=0,alpha=1,
        extent=(0,int(w_display*dl),0,int(h_display*dl)))
    
    plt.imshow(Pot.transpose()[int((Ny-h_display)/2):\
                                int((Ny+h_display)/2),
                                0:w_display],
               origin='lower',vmax=Pot_max*opac_k,cmap=comap,alpha=opac,
        extent=(0,int(w_display*dl),0,int(h_display*dl)))
    

fig9 = plt.figure()
ax1 = plt.subplot()

Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim9 = ani.FuncAnimation(fig9, update9, 
                               frames = int(Nt/Nden)-1, 
                               blit = False, interval=200)
save=1
if save==1:
    if central==0:
        anim9.save('Nx'+str(Nx)+'_Ny'+str(Ny)+'_maxV'+str(max_V)\
                   +'_px'+str(px0)+'_py'+str(py0)\
                   +'_Nslits'+str(Nslt)+'_dev'+str(Ndev)+'_r'+str(Nr)\
                   +'_Sch_v5.mp4', writer=writer)
    if central==1:
        anim9.save('Nx'+str(Nx)+'_Ny'+str(Ny)+'_Z'+str(Z)\
                   +str(sign)+'_central_pot'\
                   +'_Sch_v5.mp4', writer=writer)
    
V=np.real(V)
print(V)
plt.imshow(np.transpose(V))
plt.savefig(str(Nslt)+"V_slits_r.png")

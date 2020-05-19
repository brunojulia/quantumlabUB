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


dxvec=np.array([0.03])
dtvec=np.array([0.01])

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

def sigma2t(t,sigma0):
    sigma2t=sigma0+ (0.25/sigma0)*t**2
    return sigma2t

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
    energia=(1/2.)*(p0x**2+p0y**2 +1/(2.*s2))+(np.sqrt(2*np.pi))*s2**(3/2)
    return energia

def eteop(s2):
    V=(np.sqrt(2*np.pi))*s2**(3/2)
    return V

def eteoc(p0x,p0y,s2):
    energia=(1/2.)*(p0x**2+p0y**2 +1/(2.*s2))
    return energia

def xespt1(t):   
    x=(10./2)*t
    if t>=0.6:
        x=3-(10./2)*(t-0.6)
    return x

def xespt2(t):   
    x=(10.)*t
    if t>=0.3:
        x=3-(10.)*(t-0.3)
    if t>=0.9:
        x=-3+(10.)*(t-0.9)
    if t>=1.5:
        x=3-10*(t-1.5)
    return x

#%%
c=['g','b','r']
fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Energia estat fonamental(w=2)')
for i in range(len(dxvec)):
    #Calcul de la energia       
    for j in range(len(dtvec)):       
        tvec=np.load('tvecharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        Hc=np.load('Echarmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        Hp=np.load('Epharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        plt.plot(tvec,Hp,label='Ep(computació)')
        plt.plot(tvec[0:500],Hc[0:500],label='Ec(computació)')
        Et=Hc+Hp
        plt.plot(tvec,Et,label='Et(computació)')
        
#A=eteoc(5.,0.,0.25)


#plt.plot(25,label='Et(teorica)')

#energpvec=np.array([eteop(0.25) for i in range(len(tvec))])        
#plt.plot(tvec,energpvec,label='Ep(teorica)')


# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
plt.xlabel('t')
plt.ylabel('E')
plt.savefig('Eharm2')
#%%
c=['g','b','r']
fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Energia potencial paquet(p0=0)(Vharm(w=1))')
energpvec=np.array([eteop(0.25) for i in range(len(tvec))])        
plt.plot(tvec,energpvec,label='Ep(teorica)')

plt.plot(tvec,Hp,label='Ep(computació)')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
plt.xlabel('t')
plt.ylabel('Ep')
plt.savefig('energiapotencialharmdiscretitzat')


#%%

#Calcul de la norma:
fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('norma(dx,dt)')
for i in range(len(dxvec)):
    
    for j in range(len(dtvec)):       
        normavector=np.load('normatotharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        tvec=np.load('tvecharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        plt.plot(tvec[0:500],normavector[0:500],'.',label='dt={}/dx={}'.format(dtvec[j],dxvec[i]))
    
# Shrink current axis by 20%
box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.ylim(0.99,1.01)
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
plt.xlabel('t')
plt.ylabel('norma')
plt.savefig('normadiscretizatp05')
#%%
#Dispersió i valor sperat per x
fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Dispersiox(p0=5)'.format(dxvec[i]))

for i in range(len(dxvec)):
    #Calcul de la norma:
   
    for j in range(len(dtvec)):       
        normavector=np.load('dispersioxharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        tvec=np.load('tvecharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
#        plt.plot(tvec,normavector,'.',label='dt={}/dx={}'.format(dtvec[j],dxvec[i]))
        plt.plot(tvec,normavector,label='Valor computat')
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
sigmavec=sigma2t(tvec,0.25)
plt.ylim(0,1.75)
plt.plot(tvec,sigmavec,label='Valor teoric')
plt.xlabel('t')
plt.ylabel('sigmax**2')
plt.savefig('dispersioxpo5')
#%%
#Calcul de la xesp
fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Valor esperat x(dx,dt)')

for i in range(len(dxvec)):
    for j in range(len(dtvec)):       
        normavector=np.load('xespxharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        tvec=np.load('tvecharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        plt.plot(tvec,normavector,'.',label='dt={}/dx={}'.format(dtvec[j],dxvec[i]))

xespvect1=np.array([xespt1(tvec[i]) for i in range(len(tvec))]) 
xespvect2=np.array([xespt2(tvec[i]) for i in range(len(tvec))])
plt.plot(tvec,xespvect1,'.',label='x=(t*p0)/m')
#plt.plot(tvec,xespvect2,'.',label='x=(t*p0)/m')

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlim(0,1.5)
plt.ylim(-3,3)
plt.xlabel('t')
plt.ylabel('x')
plt.show()
plt.savefig('xespxp05dt{}'.format(i))

#%%
#dipsersio en y    
for i in range(len(dxvec)):
    #Calcul de la norma:
    fig=plt.figure(figsize=[10,8])
    ax=plt.subplot(111)
    plt.suptitle('Dispersio y(p0=5)'.format(dxvec[i]))
    for j in range(len(dtvec)):       
        normavector=np.load('dispersioyharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        tvec=np.load('tvecharmdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
#        plt.plot(tvec,normavector,'.',label='dt={}'.format(dtvec[j]))
        plt.plot(tvec,normavector,label='Valor computat')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.plot(tvec,sigmavec,label='Valor teoric')
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    sigmavec=sigma2t(tvec,0.25)
    
    plt.xlabel('t')
    plt.ylabel('sigmay**2')
    plt.show()
    plt.savefig('dispersioypo05')

#%%

for i in range(len(dxvec)):
    #Calcul de la norma:
    fig=plt.figure(figsize=[10,8])
    ax=plt.subplot(111)
    plt.suptitle('Valor esperat y(dx={},dt)'.format(dxvec[i]))
    for j in range(len(dtvec)):       
        normavector=np.load('yespydx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        tvec=np.load('tvecdx{}dt{}.npy'.format(dxvec[i],dtvec[j]))
        plt.plot(tvec,normavector,'.',label='dt={}'.format(dtvec[j]))
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.xlabel('t')
    plt.ylabel('y')
    plt.show()
    plt.savefig('yespydiscretizatdt{}'.format(i))


    

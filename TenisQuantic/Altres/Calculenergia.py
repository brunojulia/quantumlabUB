# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

import numpy as np
import matplotlib.pyplot as plt

def Ham(psi,hbar,m,dx,Vvec):
    #Funcion que calcula energia cinetica i potencial
    Ec=np.zeros((np.shape(psi)),dtype=complex)
    Ep=np.zeros((np.shape(psi)),dtype=complex)
    s=(1/2.)*(1./dx**2)
    
    
    for i in range(1,Nx):
        for j in range(1,Nx):
            Ec[i,j]=-s*(psi[i+1,j]+psi[i-1,j]+psi[i,j-1]
                +psi[i,j+1]-4.*psi[i,j])
            Ec[i,j]=np.real(Ec[i,j]*np.conj(psi[i,j]))
            Ep[i,j]=psi[i,j]*Vvec[i,j]
            Ep[i,j]=np.real(Ep[i,j]*np.conj(psi[i,j]))
            
    return Ec,Ep
            

def trapezis(xa,xb,ya,yb,dx,fun):
    #Funcion que hace una integral doble entre -L i L
    Nx=np.int((xb-xa)/dx)
    Ny=np.int((yb-ya)/dx)    
    funsum=0.
    for i in range(Nx+1):
#        funsum=funsum+(fun[i,0]+fun[i,-1])*(dx**2)/2
        for j in range(1,Ny):
            funsum=funsum+fun[i,j]*dx**2
    return funsum

    
def psi0f(x,y,s2,p0):
    #Genera un paquete de ondas situado en x=0 i y=0
    # de sigma**2=s2 i momento p0
    n=1./((2*np.pi*s2)**(1/2))
    a=n*np.exp(-((x)**2+(y)**2)/(4*s2))*(np.exp(x*p0*1j))
    return a

def norma(psi):
    #Calcula la norma de psi
    return np.array([[np.real((psi[i,j])*np.conj(psi[i,j])) for j in range(Nx+1)] 
             for i in range(Nx+1)])

#Anchura de la caja(acaba siendo 2*L)
L=3.
#Donde guardaremos los archivos
dxvec=np.zeros((5),dtype=float)
H=np.zeros((20,5),dtype=float)
normavec=np.zeros((5))

#Bucle para calcular las energias de diferentes paquetes en función de su discretizado

for j in range(0,20):
    for i in range(0,5):
        #Numero de pasos
        dx=np.float(0.15-0.01*3.*np.float(i))
        #Ancho del discreitzado
        Nx=np.int((2*L)/dx)
        #Guardamos este dato
        dxvec[i]=dx
        #Generamos un vectors psi0 con el discretizado dado.
        psi0=np.array([[psi0f(-L+np.float(i)*dx,-L+np.float(n)*dx,0.25,np.float(j)) for i in range(Nx+1)]
                      for n in range(Nx+1)],dtype=complex)
        #Matriz del potencial
        Vvec=np.zeros((np.shape(psi0)))
        #Calculamos energia cinética i potencial
        a,b=Ham(psi0,1.,1.,dx,Vvec)
        #Bucle que suma cada elemento de la matriz de e.cinetica
        for n in range(0,Nx+1):
            for m in range(0,Nx+1):
                H[j,i]=H[j,i]+a[n,m]
        #Multiplicamos esto ultimo por el paso al cuadrado ya que 
        #la energia se calcula mediante una integral
        H[j,i]=H[j,i]*dx**2
        #Aqui tambien podemos utilitzar trapezis(-L,L,-L,L,dx,a)
#        normas=norma(psi0)
#        normavec[i]=trapezis(-L,L,-L,L,dx,normas)
#%%
def eteo(p0x,p0y,s2):
    energia=(1/2.)*(p0x**2+p0y**2 +1/(2.*s2))
    return energia
#Representamos
fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Energia de paquet en funcio de dx i p0')   

cv=['b','r','m','k','g']
for j in range(1,5):
    plt.plot(dxvec,H[5*j-1,:],'.',color=cv[j],label='p0={}'.format(5*j-1))
    energvec=np.array([eteo(5*j-1,0,0.25) for i in range(0,5)])
    plt.plot(dxvec,energvec,color=cv[j],label='E_teorica(p0={})'.format(5*j-1))
    
plt.plot(dxvec,H[0,:],'.',color='y',label='p0={}'.format(0))
energvec=np.array([eteo(0,0,0.25) for i in range(0,5)])
plt.plot(dxvec,energvec,color='y',label='E_teorica(p0={})'.format(0))

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
plt.xlabel('dx')
plt.ylabel('E')
plt.savefig('Paquetenergiasmom')  



#plt.figure()
#plt.suptitle('Norma de paquet(p0=50,s**2=0.25,L=3) en funcio de dx')   
#plt.plot(dxvec,normavec)
#plt.xlabel('dx')
#plt.ylabel('norma')
#plt.ylim(0.9750,1.025)
#plt.savefig('Paquet0norma')    
    
     
    
    
    



# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 07:49:48 2018

@author: rafa

The program follows the nomenclature from:
    Notes on the solution to 1D Schrodinger equation
    for piecewise potentials - bjd
And tries to find its eigenenergies.
    
Mute index k is replaced by b 
(which stands for boundaries)
"""

import matplotlib.pyplot as plt
import numpy as np
    
"""
#------------------------------------------------------------

#generate a random potential

dx=1/np.float(1000)
x_vect=np.arange(0,1+dx,dx)

N=1 #number of potential columns -1
xb=np.zeros(N+1)
Vb=np.zeros(N+1)

for ib in range(N+1):
    xb[ib]=(((ib+1)/(N+1))
    +(1/(N+1))*(-(0.5-dx) + (1-2*dx)*np.random.random()))
    Vb[ib]=0.001*np.random.random()
xb[-1]=1 

#draw the random potential generated

V_vect=[]

ib=0
for x in x_vect:
    if x > xb[ib]:
        ib+=1
    V_vect.append(Vb[ib])
    
plt.plot(x_vect,V_vect,'r-')
#plt.axvline(x=0)
#plt.axvline(x=1)
plt.grid(True)
plt.title("Potencial")
plt.xlabel("x/L")
plt.ylabel("V")
"""

#------------------------------------------------------------
#Generate the easy potential
    
dx=1/np.float(100)
x_vect=np.arange(0,1+dx,dx)

N=1 #number of potential columns -1
xb=np.zeros(N+1)
Vb=np.zeros(N+1)

for ib in range(N+1):
    xb[ib]=((ib+1)/(N+1))
Vb=[0.001,100]
print(xb,Vb)

V_vect=[]

ib=0
for x in x_vect:
    if x > xb[ib]:
        ib+=1
    V_vect.append(Vb[ib])


#------------------------------------------------------------
#find the eigen-energies

phi=1j*np.zeros(shape=(N+1,2))

phi[0]=np.array((1,-1))

#phi[N]=[1,-np.exp(2j*kn)]


Eguay = np.pi*np.pi*0.5

dE=Eguay/np.float(10007)
EE=np.arange(0.5*Eguay,10*Eguay+dE,dE)

elf=[]

kb=1j*np.zeros(N+1)

for E in EE:
    
    for ib in range(N+1):
         kb[ib]=(np.sqrt(2*(E-Vb[ib])+0j))
        
    #define the effective matrix
    # n = “next“ = ib+1, l = “last“ = ib
            
    M_eff=np.eye(2)
    kn=kb[0] 
    
    for ib in range(N):
        kl=kn
        xn=xb[ib+1]
        kn=kb[ib+1]
        
        ex=np.exp(1j*kl*xn)
        M=np.array(((ex,ex**-1),(1j*kl*ex,-1j*kl*ex**-1)))
        
        ex=np.exp(1j*kn*xn)
        invM=np.array(((0.5*ex**-1,(-1j*0.5*(1/kn))*ex**-1),
                       (0.5*ex,(1j*0.5*(1/kn))*ex)))
        
        Mb=np.dot(invM,M)
        
        M_eff=np.dot(Mb,M_eff)        
        phi[ib+1]=np.dot(Mb,phi[ib])
    
    phi_test=np.dot(M_eff,phi[0])
    elf.append(np.absolute(np.exp(2j*kn)*phi_test[0]+phi_test[1]))

#plt.plot(EE/Eguay,elf,'r-')
#plt.grid(True)
#plt.title("Arrels d'energia")
#plt.xlabel(r"""$E(\~E)$""")
#plt.ylabel(r"""$f$""")
#plt.show()

#------------------------------------------------------------

E = 3.0498469501530496*Eguay

for ib in range(N+1):
    kb[ib]=(np.sqrt(2*(E-Vb[ib])+0j))

psi_vect=1j*np.zeros(shape=(len(x_vect)))#,N+1))


ib=0
for ix in range(len(x_vect)):
    if x_vect[ix]>xb[ib]:
        ib+=1
    expo=np.exp(1j*kb[ib]*x_vect[ix])
#    psi_vect[ix][ib]=phi[ib][0]*expo+phi[ib][1]*expo**-1
    psi_vect[ix]=np.absolute(phi[ib][0]*expo+phi[ib][1]*expo**-1)**2
#    print(ix,ib,kb[ib],phi[ib],psi_vect[ix])

print(np.absolute(np.exp(2j*kn)*phi[ib][0]+phi[ib][1]))      


plt.plot(x_vect,psi_vect,'g-')
#plt.plot(xx,sin2,'r-',label='$\sin(\pi x L)^2$')
plt.grid(True)
plt.legend()
plt.title("Polipot")
plt.xlabel("x/L")
plt.ylabel("$\Psi^2 /A$")
plt.show()



















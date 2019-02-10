# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 07:49:48 2018

@author: rafa

The program follows the nomenclature from:
    Notes on the solution to 1D Schrodinger equation
    for piecewise potentials - bjd
And finds its eigenenergies.
    
"""

import matplotlib.pyplot as plt
import numpy as np
        
#------------------------------------------------------------
#units
Eguay = np.pi*np.pi*0.5

#------------------------------------------------------------
#Generate the easy potential

#walls only
#Vk=[0,2]

#step
#Vk=[0,2]

#this is symetrical from 0 to 25
#Vk=[0,30,0]

#harmonic oscillator with max value = 25
Vk=[25,16,9,4,1,0,1,4,9,16,25]

Vk=np.dot(Vk,Eguay)

N=len(Vk) #number of potential columns

xk=np.zeros(N)
for k in range(N):
    xk[k]=(k+1)/N


#------------------------------------------------------------
#find the eigen-energies

phi=1j*np.zeros(shape=(N,2))
phi_bis=1j*np.zeros(shape=(N,2))

phi[0][0]=1
phi[0][1]=-1

phi_bis[0][0]=1
phi_bis[0][1]=-1

invM=1j*np.zeros(shape=(2,2))
M=1j*np.zeros(shape=(2,2))

dE=Eguay/np.float(1007)
EE=np.arange(0.9*Eguay,10*Eguay,dE)

elf=[]
relf=[]
ielf=[]
melf=[]

elf_min=np.infty
E_min=-123

kk=1j*np.zeros(N)

for E in EE:
        
    for k in range(N):
        kk[k]=np.sqrt((2+0j)*(E-Vk[k]))
    
    M_eff=np.eye(2)+1j*np.zeros(shape=(2,2))
        
    for k in range(N-1): 
        #k_python = k_notes -1
        #k_notes = 1 : N-1
        #k_python = 0 : N-2
            
        #make invM21 and M11 phi_2
                
        ex1=np.exp(1j*kk[k]*xk[k])
        ex_1=ex1**-1
        ex2=np.exp(1j*kk[k+1]*xk[k])
        ex_2=ex2**-1
        
        M[0][0]=ex1
        M[0][1]=ex_1
        M[1][0]=1j*kk[k]*ex1
        M[1][1]=-1j*kk[k]*ex_1
        
        invM[0][0]=0.5*ex_2
        invM[0][1]=-0.5j*(1/kk[k+1])*ex_2
        invM[1][0]=0.5*ex2
        invM[1][1]=0.5j*(1/kk[k+1])*ex2
        
        M_eff=np.dot(np.dot(invM,M),M_eff)
    
    phi[-1]=np.dot(M_eff,phi[0])
   
    elfE=np.log(phi[-1][1]+0j)-np.log(phi[-1][0]+0j)-2j*kk[-1]+1j*np.pi
    
    elfE=np.exp(elfE)-1
    
#    elf.append(np.absolute(elfE))
    
    relf.append(np.real(elfE))
    
    ielf.append(np.imag(elfE))
    
    melf.append(np.real(elfE)+np.imag(elfE))
    
    if len(melf)>3:
        if melf[-1]*melf[-3]<0:
            if relf[-1]*relf[-3]<0 or ielf[-1]*ielf[-3]<0:
                E_min=E
                break
    
#    if elf_min>np.absolute(elf[-1]):
#        elf_min=np.absolute(elf[-1])
#        E_min=E

print("polipot")
print("V =",Vk[-1]/Eguay,"Eguays")
print("E =",E_min/Eguay,"Eguays")

#------------------------------------------------------------

E = E_min
    
dx=1/np.float(1000)
x_vect=np.arange(0,1+dx,dx)

for k in range(N):
    kk[k]=np.sqrt((2+0j)*(E-Vk[k]))
        
for k in range(N-1): 
    #k_python = k_notes -1
    #k_notes = 1 : N-1
    #k_python = 0 : N-2
                
    #make invM21 and M11 phi_2
                    
    ex1=np.exp(1j*kk[k]*xk[k])
    ex_1=ex1**-1
    ex2=np.exp(1j*kk[k+1]*xk[k])
    ex_2=ex2**-1
            
    M[0][0]=ex1
    M[0][1]=ex_1
    M[1][0]=1j*kk[k]*ex1
    M[1][1]=-1j*kk[k]*ex_1
            
    invM[0][0]=0.5*ex_2
    invM[0][1]=-0.5j*(1/kk[k+1])*ex_2
    invM[1][0]=0.5*ex2
    invM[1][1]=0.5j*(1/kk[k+1])*ex2
            
    phi[k+1]=np.dot(np.dot(invM,M),phi[k])

psi_vect=np.zeros(shape=(len(x_vect)))
V_vect=np.zeros(shape=(len(x_vect)))

ixk=[0]
ik=0


for ix in range(len(x_vect)):
    if x_vect[ix]>xk[ik]:
        ik+=1
        ixk.append(ix-1)
        ixk.append(ix)
        
    V_vect[ix]=Vk[ik]*(4/np.max(Vk))

    ex=np.exp(1j*kk[ik]*x_vect[ix])
        
    psi_vect[ix]=np.absolute(
            phi[ik][0]*ex+phi[ik][1]*ex**-1)**2
            
            
            
def simpson(h,vect): #Simpson for discrete vectors

    #add the extrems
    add=(vect[0]+vect[len(vect)-1])

    #add each parity its factor
    for i in np.arange(2,len(vect)-1,2):
        add+=2*vect[i]

    for i in np.arange(1,len(vect)-1,2):
        add+=4*vect[i]

    #add the global factor
    add*=(h/np.float(3))

    return add
            
psi_vect=np.dot(simpson(dx,psi_vect)**-1,psi_vect)

        
ixk.append(ix-1)

plt.plot(x_vect,V_vect,'k--')

color=['b-', 'g-', 'r-', 'c-', 'm-']

for k in range(N):
    plt.plot(x_vect[ixk[2*k]:ixk[2*k+1]],
             psi_vect[ixk[2*k]:ixk[2*k+1]],
    color[k%5])

plt.grid(True)
plt.legend()
plt.title((E/Eguay, '$\tilde{E}$'))
plt.xlabel("x/L")
plt.ylabel("$\Psi^2 /A$")
plt.show()

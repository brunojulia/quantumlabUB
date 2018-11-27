# -*- coding: utf-8 -*-
"""
Rafa da Silva
Created Oct 18 2018

Last check Oct 19 2018 

Given the Schrödinger equation and a simple squared potential
side L=1, compute the energy levels and draw its wavefunction.
"""

import matplotlib.pyplot as plt
import numpy as np
    

#------------------------------------------------------------------

#Choose your potential
# V <= 10^4 to avoid overflow

V = 72
xs = 0.43

#Ground state of an infinite square potential with side L=1
Eguay = np.pi*np.pi*0.5

#The analitical finder of the energy levels from root.py

def fun(E,V,x):
    
    alpha = np.sqrt(2*E+0j)
    beta = np.sqrt(2*(E-V)+0j)
    
    eA = np.exp(2j*alpha*x)
    eB = np.exp(2j*beta*(x-1))
    
    f = ((eA-1)/(alpha*(eA+1)))/((eB-1)/(beta*(eB+1)))
    
    return np.absolute(f)-1

#-----------------------------------------------------------------------------
#draw potential with the wave function of the first energy level



a = 0
b = 100*Eguay
N = 1000001
deltaE = (b-a)/np.float(N)
EE=np.arange(a,b+deltaE,deltaE)

f0 = fun(EE[0],V,xs)
f1 = fun(EE[1],V,xs)

for j in range(len(EE)-1):
    if (f1*f0) < 0 :
        E=0.5*(EE[j-1]+EE[j])
        break
    f0 = fun(EE[j],V,xs)
    f1 = fun(EE[j+1],V,xs)

#E=EE[0]
print(E/Eguay, "Eguays")

def phi_fun_L(E,x):
    
    alpha = np.sqrt(2*E+0j)
    
    return np.exp(1j*alpha*x)-np.exp(-1j*alpha*x)

def phi_fun_R(E,V,x):
    
    beta = np.sqrt(2*(E-V)+0j)

    return np.exp(1j*beta)*(np.exp(1j*beta*(x-1))-np.exp(-1j*beta*(x-1)))

def simpson(a,b,h,vect): #Simpson for discrete vectors

    #add the extrems
    add=(vect[a]+vect[b])

    #add each parity its factor
    for i in np.arange(a+2,b,2):
        add+=2*vect[i]

    for i in np.arange(a+1,b,2):
        add+=4*vect[i]

    #add the global factor
    add*=(h/np.float(3))

    return add

def c2a2(E,V,x):
    
    alpha = np.sqrt(2*E+0j)
    beta = np.sqrt(2*(E-V)+0j)
    
    eA = np.exp(2j*alpha*x)
    eB = np.exp(2j*beta*(x-1))
    
    f = (alpha*(eA-eA**-1))/(beta*np.exp(2j*beta)*(eB-eB**-1))
    
    return np.absolute(f)

a=0
b=1
N=100007
deltax=(b-a)/np.float(N)
xx=np.arange(a+deltax,b,deltax)

phi_val_L=1j*np.zeros([len(xx)])
phi_val_R=1j*np.zeros([len(xx)])

i_L=int((len(xx)-len(xx)%(1/xs))*xs)
i_R=i_L+2

for i in range(0,i_L+1):
    phi_val_L[i]=phi_fun_L(E,xx[i])
for i in range(i_R,len(xx)):
    phi_val_R[i]=phi_fun_R(E,V,xx[i])

phi2_L=np.absolute(np.multiply(np.conjugate(phi_val_L),phi_val_L))
phi2_R=np.absolute(np.multiply(np.conjugate(phi_val_R),phi_val_R))

A2=1/(simpson(0,i_L,deltax,phi2_L)
+np.absolute(c2a2(E,V,xs))*simpson(i_R,len(xx)-1,deltax,phi2_R))

C2=np.absolute(c2a2(E,V,xs))*A2
C2_fake=(phi2_L[i_L]/phi2_R[i_R])*A2

phi2_L=np.multiply(phi2_L,A2)
phi2_R=np.multiply(phi2_R,C2)

suma=0
for i in range(0,i_L+1):
    suma+=phi2_L[i]*deltax
for i in range(i_R,len(xx)):
    suma+=phi2_R[i]*deltax
    
print(suma)
    
#-----------------------------------------------------------------------------
#What the whole thing looks like

def pot(V,x,xs):
    if x<=xs and x>0:
        return 0
    elif x<1 and x>xs:
        return V
    else: 
        return 1000000000

pot_vect =[]
sin2 = []
sin22 = []
for x in xx:
    sin2.append(2*np.sin(x*np.pi)**2)
    sin22.append((1/xs)*2*np.sin(x*np.pi*(1/xs))**2)
    pot_vect.append(pot(V,x,xs))
    
plt.plot(xx,np.multiply(pot_vect,2*(1/xs)*(1/V)),
         'r--',label=("$V$",V,"$x_S$",xs))
plt.plot(xx[:i_L],phi2_L[:i_L],'g-',label='$\Psi^2_L$')
plt.plot(xx[i_R:],phi2_R[i_R:],'b-',label='$\Psi^2_R$')
plt.plot(xx,sin2,'c--',
         label='$\Psi^2(V=0) = 2 sin^2(\pi x / L)/ L$')
plt.plot(xx[:i_L],sin22[:i_L],'m--',
         label='$\Psi^2(V=\infty) = 2 sin^2(\pi x / x_S)/ x_S$')
plt.grid(True)
plt.legend()
plt.title('Probability')
plt.xlabel("x/L")
plt.ylabel("$\Psi^2$")
plt.show()











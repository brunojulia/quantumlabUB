#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:43:26 2022

@author: annamoresoserra
"""

# Codi per comprovar el correcte funcionament de CN2D.
# Volem veure que l'energia correspongui al valor teòric.
import numpy as np
import matplotlib.pyplot as plt

# Funció per normalitzar a cada temps fent servir el mètode de Simpson(? millor?)
# per calcular les integrals.


xmi=0.
xma=1.
ymi=0.
yma=1.
	
tb=0.5
ta=0.
deltax=0.03	
deltay=deltax
deltat=0.01
Nx=int((1.)/deltax)
Ny=Nx
Nt=int((tb-ta)/deltat)



def trapezis(xa,xb,ya,yb,dx,fun):
  
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*(dx**2)/2
        for j in range(1,Ny):
        	funsum=funsum+fun[i,j]*dx**2
    
    return funsum
	
# Normalització per a cada temps
def normatotal(norma,dx):
	normalitza=np.zeros( (len(norma[1,1,:])))
	for j in range(len(norma[1,1,:])):
		normalitza[j]= trapezis(0,xma,0,yma,dx,norma[:,:,j])
	return normalitza
    
    
Vvec=np.load('Vvecharmdx{}dt{}.npy'.format(deltax,deltat))
normes=np.load('normaharmdx{}dt{}.npy'.format(deltax,deltat))
psivec=np.load('psiharmdx{}dt{}.npy'.format(deltax,deltat))	
normavect=normatotal(normes,deltax)
print(normavect)
np.save('normatotharmdx{}dt{}.npy'.format(deltax,deltat),normavect)
          

# Càlcul de l'energia per un temps determinat
#def Energia(psi,dx,dy,m,hbar,V):
#	Ec=np.zeros((np.shape(psi)),dtype=complex)
#	Ep=np.zeros((np.shape(psi)))
#	Nx=int(len(psi[0,:]))-1
#	Ny=int(len(psi[:,0]))-1
#	for i in range (1,Nx):
#		for j in range (1,Nx):
#			Ec[i,j]=-((hbar**2)/(2*m))*(((psi[i+1,j]-psi[i,j]+psi[i-1,j]+psi[i,j+1])/(dx**2))+
#				((psi[i,j+1]-psi[i,j]+psi[i,j-1])/(dy**2)))
		
	
	
# Càlcul de l'energia total obtinguda




fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Norma partícula en caixa')


normavector=np.load('normatotharmdx{}dt{}.npy'.format(deltax,deltat))
tvec=np.load('tvecharmdx{}dt{}.npy'.format(deltax,deltat))
plt.plot(tvec[0:500],normavector[0:500],'.-',label='dt={}/dx={}'.format(deltat,deltax))
		
		
# Shrink current axis by 20%
box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
plt.ylim(0.99,1.01)
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
plt.xlabel('t')
plt.ylabel('norma')
plt.savefig('normadiscretizatp05')

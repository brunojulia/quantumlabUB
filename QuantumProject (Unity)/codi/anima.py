#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 01:44:35 2022

@author: annamoresoserra
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


L=3.
m=1.
hbar=1.
tb=2.
ta=0.
w=2.
deltax=0.040	
deltay=deltax
deltat=0.01
Nx=int((2.*L)/deltax)
Ny=Nx
Nt=int((tb-ta)/deltat)
 
normas=np.load('normaoh2vgauss13dx{}dt{}.npy'.format(deltax,deltat))

#Animació
fig, ax = plt.subplots()
ax=plt.subplot(111)
plt.suptitle('Evolució densitat de probabilitat OH n=2 amb barrera gaussiana')


ax.set_xlabel('x')
ax.set_ylabel('y')

im=ax.imshow(normas[:,:,0],extent=(-L,L,-L,L))
#imvec=ax.contour(Vvec[:,:,0],level=[-.1,1],colors='gray',linestyles='-',
#                 extent=(-L,L,-L,L))
cn=plt.colorbar(im,label='P')
# First set up the figure, the axis, and the plot element we want to animate
#line, = ax.plot([], [], lw=2)
# initialization function: plot the background of each frame
def init():
    im.set_array(normas[:,:,0])
    return im,cn

# animation function.  This is called sequentially
def animate(i): 
    # exponential decay of the values    
    im.set_array(normas[:,:,i])
#   imvec=ax.contour(Vvec[:,:,i],level=[-.1,.1],colors='gray',linestyles='-',
#                 extent=(-L,L,-L,L))
    return im,cn

anim=animation.FuncAnimation(fig, animate, frames=range(1,Nt-1),
                    interval=0.001 ,init_func=init)

#He de repetir la següent
# Gaussiana de px=-8 py=0 desplaçada -1 en x amb una barrera -1.4<=x<=-1.) and ((-3.<=y<=-0.5) or (0.5<=y<=3.))


#
anim.save('13animoh2vgaussdx0.04dt0.01.gif')




plt.imshow(normas[:,:,25])

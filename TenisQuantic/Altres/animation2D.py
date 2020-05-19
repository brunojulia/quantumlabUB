# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

dxvec=np.array([0.03])
dtvec=np.array([0.01])

for j in range(len(dtvec)):
    normas=np.load('normaharmdx0.03dt{}.npy'.format(dtvec[j]))
    dades=np.load('dvecharmdx0.03dt{}.npy'.format(dtvec[j]))
    
    L=3.
    Nt=np.int(1/dades[1])
    #Animació
    fig, ax = plt.subplots()
    #ax.set_title('Paquet gaussia[p0=5,sigma**2=0.25]//tb=1.8//dx=0.03//dt={}'.format(dtvec[j]))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Estat fonamental per w=2')
    im=ax.imshow(normas[:,:,0],origin={'lower'},extent=(-L,L,-L,L))
    #imvec=ax.contour(Vvec[:,:,0],level=[-.1,1],colors='gray',linestyles='-',
    #                 extent=(-L,L,-L,L))
    cn=plt.colorbar(im,label='p(x,y)')
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
    
    anim=matplotlib.animation.FuncAnimation(fig, animate, frames=range(1,Nt-1),
                        interval=0.0000000000001 ,init_func=init)
    anim.save('Paquetharm2dxdt{}.gif'.format(j))









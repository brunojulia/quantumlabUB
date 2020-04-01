# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation


normas=np.load('normasprov.npy')
dades=np.load('dadesprov.npy')
Vvec=np.load('Vvec.npy')
L=dades[0]
Nt=dades[2]
#Animació
fig, ax = plt.subplots()
ax.set_title('Potencial infinito variable con t[p0=50,sigma**2=0.25]')
ax.set_xlabel('x')
ax.set_ylabel('y')

im=ax.imshow(normas[:,:,0],origin={'lower'},extent=(-L,L,-L,L))
imvec=ax.contour(Vvec[:,:,0],level=[-.1,1],colors='gray',linestyles='-',
                 extent=(-L,L,-L,L))
cn=plt.colorbar(im,label='p(x,y)')
# First set up the figure, the axis, and the plot element we want to animate
#line, = ax.plot([], [], lw=2)
# initialization function: plot the background of each frame
def init():
    im.set_array(normas[:,:,0])
    return im,cn,imvec

# animation function.  This is called sequentially
def animate(i): 
    # exponential decay of the values    
    im.set_array(normas[:,:,i])
    imvec=ax.contour(Vvec[:,:,i],level=[-.1,.1],colors='gray',linestyles='-',
                 extent=(-L,L,-L,L))
    return im,imvec,cn

anim=matplotlib.animation.FuncAnimation(fig, animate, frames=range(1,Nt-1),
                    interval=0.00000000001 ,init_func=init)

anim.save('Potencialvariable.gif')







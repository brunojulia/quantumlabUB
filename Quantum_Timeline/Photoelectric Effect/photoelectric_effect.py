# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 16:52:23 2021

@author: llucv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


N_blur=256
vals = np.ones((N_blur, 4,N_blur*5))

for i in range(N_blur):
    
    vals[:N_blur, 0, i] = np.linspace(1, 1, N_blur)
    vals[:N_blur, 1, i] = np.linspace(1, i/N_blur, N_blur)
    vals[:N_blur, 2, i] = np.linspace(1, 0, N_blur)
    
    vals[:N_blur,0,i+N_blur]=\
                            np.linspace(1, (N_blur-i)/N_blur, N_blur)
    vals[:N_blur,1,i+N_blur]=np.linspace(1,1,N_blur)
    vals[:N_blur,2,i+N_blur]=np.linspace(1,0,N_blur)
    
    vals[:N_blur, 0, i+2*N_blur] = np.linspace(1, 0, N_blur)
    vals[:N_blur, 1, i+2*N_blur] = np.linspace(1, 1, N_blur)
    vals[:N_blur, 2, i+2*N_blur] = \
                            np.linspace(1, i/N_blur, N_blur)
    
    vals[:N_blur, 0, i+3*N_blur] = np.linspace(1, 0, N_blur)
    vals[:N_blur, 1, i+3*N_blur] = \
                        np.linspace(1, (N_blur-i)/N_blur, N_blur)
    vals[:N_blur, 2, i+3*N_blur] = np.linspace(1, 1, N_blur)
    
    vals[:N_blur, 0, i+4*N_blur] = \
                        np.linspace(1, (i)/N_blur, N_blur)
    vals[:N_blur, 1, i+4*N_blur] = np.linspace(1, 0, N_blur)
    vals[:N_blur, 2, i+4*N_blur] = np.linspace(1, 1, N_blur)
    
def cmp(i):
    cmp=ListedColormap(vals[:,:,i])
    return cmp

c=((1,2,3),(3,4,5),(0,0,2))
plt.imshow(c,cmap=cmp(10))
plt.show()

def pe_I(Vcol,light_inty,light_freq):
    pass

    
    
    

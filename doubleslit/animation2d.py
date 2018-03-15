"""
This script creates a matplotlib animation and a gif of the results of the script cn2d.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

psit = np.load("psit2d.npy")
t = np.loadtxt("times2d.dat")
x = np.loadtxt("x2d.dat")
y = np.loadtxt("y2d.dat")

frames = psit.shape[0]
fig = plt.figure()
ax = plt.axes()
im = ax.imshow(np.absolute(psit[0])**2)

def init():
    im.set_array(np.absolute(psit[0])**2)
    return im

def animate(i):
    p = np.absolute(psit[i])**2
    im.set_array(p)
    ax.set_title('{:6.4f}'.format(t[i]))
    return p

anim = animation.FuncAnimation(fig, animate, init_func=init, interval=20, frames=frames)
anim.save("psit2d.gif", dpi = 80, writer='imagemagick')
plt.show()

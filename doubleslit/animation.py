"""
This script creates a matplotlib animation and a gif of the results of the script cn1d.py
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

psit = np.loadtxt("psit.dat").view(complex)
t = np.loadtxt("times.dat")
V = np.loadtxt("V.dat")
x = np.loadtxt("x.dat")


frames = psit.shape[0]

fig = plt.figure()
ax = plt.axes(xlim=(x[0], x[-1]), ylim=(0, 1))
psiline, = ax.plot([], [], lw=1)
psiline2, = ax.plot([], [], "--", lw=1)

potline, = ax.plot([], [], lw=1, c = "r")
txt = ax.text(0.5,0.5,"")


def psi_teo(x, t):
    s = 0.5
    return (2/np.pi)**0.25*np.sqrt(s/(2*s**2+1j*t))*np.exp(-x**2/(4*s**2+2j*t))

def init():
    psiline.set_data([], [])
    psiline2.set_data([], [])
    potline.set_data([], [])

    return psiline,psiline2,potline

def animate(i):
    psiline.set_data(x, np.absolute(psit[i])**2)
    psiline2.set_data(x, np.absolute(psi_teo(x, t[i]))**2)
    potline.set_data(x, V)

    txt.set_text("t= " + str(t[i]))
    return psiline,psiline2,potline,txt

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=frames, interval=1, blit=True)

anim.save('psit.gif', dpi=80, writer='imagemagick')
plt.show()

"Time evolution of the solutions given for the bright and dark solitons, in solitonic units"

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation


def bright(z,t,v,n):
    "solution for a bright soliton whith Vext=0"
    imag= 0.0 + 1j
    arg1=(z*v/np.sqrt(2))
    arg2=((0.5 -v**2/4)*t)
    psi=np.sqrt(n)*(1/np.cosh((z+10)/np.sqrt(2) -v*t))*np.exp(arg1*imag)*np.exp(arg2*imag)
    return psi

def grey(z,t,v,n):
    "solution for a dark soliton whith Vext=0"
    imag=0.0 +1j
    psi=np.sqrt(n)*(imag*v + np.sqrt(1-v**2)*np.tanh((z+10)/np.sqrt(2)-v*t)*np.sqrt(1-v**2))
    return psi

"time evolution animation "

#we create a figure window, create a single axis in the figure and then
#create a line object which will be modified in the animation.
#we simply plot an empty line, data will be added later
fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-6, 6))
line, = ax.plot([], [], lw=1)

#function which makes the animation happen. this function will be called to create the
#base frame upon which the animation takes place
def init():
    line.set_data([], [])
    #its important to return the line object, this tells the aimator which objects on the plot
    #to update after each frame
    return line, 


n=3 #density
vel=0.5 #velocity
steps=150

def animate_bright(i):
    z = np.arange(-10,10,0.01)
    psi = np.real((bright(z,i,vel,n)*np.conjugate(bright(z,i,vel,n))))
    line.set_data(z, psi)
    return line,

def animate_grey(i):
    z = np.arange(-10,10,0.01)
    psi = np.real((grey(z,i,vel,n)*np.conjugate(grey(z,i,vel,n))))
    line.set_data(z, psi)
    return line,

"""
#Slider
xvel= plt.axes([0.25, 0.03,0.50,0.02])
svel = Slider(xvel, 'Vel', 0, 1, valinit=vel)

def update(val):
    #vel current value of the slider
    vel = svel.val

svel.on_changed(update)
"""

#create animation start buttons
def button_bright(self):
    a = FuncAnimation(fig, animate_bright, init_func=init, frames=steps,interval=40, repeat=False)
    fig.canvas.draw()

def button_grey(self):
    anim = FuncAnimation(fig, animate_grey, init_func=init, frames=steps,interval=40, repeat=False)
    fig.canvas.draw()
#animation button
axbright = fig.add_axes([0.01, 0.6, 0.075, 0.10])
bbright = Button(axbright, 'Bright')
bbright.on_clicked(button_bright)
#animation button
axgrey = fig.add_axes([0.01, 0.4,0.075, 0.10])
bgrey = Button(axgrey, 'Grey')
bgrey.on_clicked(button_grey)


plt.show()

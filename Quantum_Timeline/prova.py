# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:44:24 2020

@author: llucv
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
print("hola")
def sumatori(x,y):
    s=x+y
    return s
sumatori(23,24)
print(sumatori(2, 8))
def fibonacci(n):
    a_n=[0,1]
    value=1
    for i in range(1,n+1):
        value=a_n[0]+a_n[1]
        a_n[0]=a_n[1]
        a_n[1]=value
    return value
print (fibonacci(1))




fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ax.set_xlim(0, 2*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.sin(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128),
                    init_func=init, blit=True)
plt.show()





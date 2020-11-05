# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:50:23 2020

@author: llucv
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as ani

e=np.e
pi=np.pi

xx=np.array([0,1,2,3])
yy=np.array([0,1,2,3])
zz=np.zeros((4,4,2))
C=np.zeros((4,4))

print(xx)

for i in range(4):
    for j in range(4):
        zz[i,j,1]=3*i+2*j
        C[i,j]=zz[i,j,1]

plt.imshow(C,vmin=0,origin='lower',extent=(0,5,0,5))
plt.savefig('prova.png')
        
xmax=2
ymax=2
h=0.1
p=0.05
c=1

x=np.linspace(0,xmax,101)
y=np.linspace(0,ymax,101)
u=np.zeros((101,101,3))
g=np.zeros((101,101))
Ut=np.zeros((101,101))
Uxt=np.zeros((101))

"El que s'intenta aquí és modelar les primers intants de l'ona plana\
    però l'ona resultant no actua com una plana."

def u0(t):
    val=np.sin(5*t)
    return val

def animate(frame):
    t=p*frame
    for j in range(1,100):
        u[0,j,0]=u0(c*t)
        u[0,j,1]=u0(c*(t+p))
        for i in range(1,100):
            A=u[i+1,j,1]+u[i,j+1,1]+u[i-1,j,1]+u[i,j-1,1]-4*u[i,j,1]
            B=2*u[i,j,1]-u[i,j,0]
            u[i,j,2]=((c*p/h)**2)*A+B
            u[i,j,1]=u[i,j,2]
            u[i,j,0]=u[i,j,1]
            Ut[i,j]=u[i,j,2]
            Uxt[i]=Ut[i,50]
    plt.pcolormesh(x,y,Ut,vmax=1,vmin=-1)
    
fig = plt.figure()
ax1 = plt.subplot()

Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim = ani.FuncAnimation(fig, animate, 
                               frames = 1000, 
                               blit = False, interval=20)

guarda1=1
if guarda1==1:
    anim.save('ona_explicit_plana_wrong.mp4', writer=writer)

a=np.zeros((101,101,3))
g=np.zeros((101,101))
s=np.zeros((101,101))
at=np.zeros((101,101))    
    
def point_source(t):
    val=50
    return val

"falta u(0) i u(p)"

def update(frame):
    t=p*frame
    for i in range(49,52):
        for j in range(49,52):
            s[i,j]=point_source(t)
    for j in range(1,100):
        for i in range(1,100):
            A=a[i+1,j,1]+a[i,j+1,1]+a[i-1,j,1]+a[i,j-1,1]-4*a[i,j,1]\
                +s[i,j]*h**2
            B=2*a[i,j,1]-a[i,j,0]
            a[i,j,2]=((c*p/h)**2)*A+B
            at[i,j]=a[i,j,1]
            a[i,j,1]=a[i,j,2]
            a[i,j,0]=a[i,j,1]

    plt.imshow(at,vmax=1,vmin=-1,origin='lower',extent=(0,2,0,2))
    
fig2 = plt.figure()
ax1 = plt.subplot()

Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim2 = ani.FuncAnimation(fig2, update, 
                               frames = 1000, 
                               blit = False, interval=20)

guarda2=0
if guarda2==1:
    anim2.save('ona_explicit_esferica.mp4', writer=writer)
    

    



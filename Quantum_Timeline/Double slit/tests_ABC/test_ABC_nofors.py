# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:53:49 2020

@author: llucv
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

e=np.e
pi=np.pi


a=np.zeros((101,31,301))
u=np.zeros((101,31))
ut=np.zeros((101,31))
s=np.zeros((101,31,301))
sgm=np.zeros((101,31,301))
PBC=np.zeros((32))

c=1.4
dt=0.05
dl=0.1
w=5
amp=5
sgm_max=0.1
m=3
gruix_paret=15
rao=(c*dt/dl)**2


def p_s(t,amp,w):
    val=amp*np.sin(w*t)
    return val

for i in range(1,31):
    PBC[i]=(i-1)

PBC[0]=(30)
PBC[31]=(0)

for i in range(100-gruix_paret,100):
    for j in range(0,31):
        sgm[i,j,:]=sgm_max*((i-99+gruix_paret)/gruix_paret)**m


start=time.time()
for k in range(2,301):
    t=(k-2)*dt
    if t<=(2*pi/w):
        s[1,:,k]=p_s(t,amp,w)
    else:
        s=np.zeros((101,31,301))
        
    a[1:-1,1:-1,k]=(rao*(a[2:,1:-1,k-1]+a[0:-2,1:-1,k-1]\
                    +a[1:-1,2:,k-1]+a[1:-1,0:-2,k-1]\
                    -4*a[1:-1,1:-1,k-1])+s[1:-1,1:-1,k]\
                +2*a[1:-1,1:-1,k-1]-a[1:-1,1:-1,k-2]\
                +sgm[1:-1,1:-1,k-1]*a[1:-1,1:-1,k-2]/(2*dt))\
                /(1+sgm[1:-1,1:-1,k-1]/(2*dt))
                
    a[1:-1,0,k]=(rao*(a[2:,0,k-1]+a[0:-2,0,k-1]\
                    +a[1:-1,1,k-1]+a[1:-1,30,k-1]\
                    -4*a[1:-1,0,k-1])+s[1:-1,0,k]\
                +2*a[1:-1,0,k-1]-a[1:-1,0,k-2]\
                +sgm[1:-1,0,k-1]*a[1:-1,0,k-2]/(2*dt))\
                /(1+sgm[1:-1,0,k-1]/(2*dt))
                
    a[1:-1,30,k]=(rao*(a[2:,30,k-1]+a[0:-2,30,k-1]\
                    +a[1:-1,0,k-1]+a[1:-1,29,k-1]\
                    -4*a[1:-1,30,k-1])+s[1:-1,30,k]\
                +2*a[1:-1,30,k-1]-a[1:-1,30,k-2]\
                +sgm[1:-1,30,k-1]*a[1:-1,30,k-2]/(2*dt))\
                /(1+sgm[1:-1,30,k-1]/(2*dt))
elapsed_time=(time.time()-start)
print(elapsed_time)

a=np.zeros((101,31,301))
u=np.zeros((101,31))
ut=np.zeros((101,31))
s=np.zeros((101,31))
sgm=np.zeros((101,31))
for i in range(100-gruix_paret,100):
    for j in range(0,31):
        sgm[i,j]=sgm_max*((i-99+gruix_paret)/gruix_paret)**m

start=time.time()
for k in range(1,301):
    for j in range(0,31):
        t=(k-1)*dt
        if t<=(2*pi/w):
            s[1,j]=p_s(t,amp,w)
        else:
            s=np.zeros((101,31))
        for i in range(1,100):
            jp=int(PBC[j+1])
            A=a[i+1,jp,k-1]+a[i,jp+1,k-1]+a[i-1,jp,k-1]+a[i,jp-1,k-1]\
              -4*a[i,jp,k-1]+s[i,jp]
            B=2*a[i,jp,k-1]-a[i,jp,k-2]
            C=sgm[i,jp]/(2*dt)
            a[i,j,k]=(rao*A+B+C*a[i,jp,k-2])/(1+C)
elapsed_time=(time.time()-start)
print(elapsed_time)


do_anim=1
if do_anim==1:
    def update01(frame):
            k=frame*5
            u=a[:,:,k]
            ut=u.transpose()
            plt.imshow(ut
                       ,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,3))
     
    fig01 = plt.figure()
    ax1 = plt.subplot()
        
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        
    anim01 = ani.FuncAnimation(fig01, update01, 
                                       frames = 60, 
                                       blit = False, interval=200)
                
anim01.save('front_pla_normal_ABC_v2.mp4', writer=writer)



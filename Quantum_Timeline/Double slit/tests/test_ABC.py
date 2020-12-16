# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 17:56:33 2020

@author: llucv
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

e=np.e
pi=np.pi

do1=0
if do1==1:
    a=np.zeros((101,31,301))
    u=np.zeros((101,31))
    ut=np.zeros((101,31))
    s=np.zeros((101,31))
    sgm=np.zeros((101,31))
    
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
    
    for i in range(100-gruix_paret,100):
        for j in range(0,31):
            sgm[i,j]=sgm_max*((i-99+gruix_paret)/gruix_paret)**m
    
    for k in range(1,301):
        for j in range(0,31):
            t=(k-1)*dt
            if t<=(2*pi/w):
                s[1,j]=p_s(t,amp,w)
            else:
                s=np.zeros((101,31))
            for i in range(1,100):
                if j==0:
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,30,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    C=sgm[i,j]/(2*dt)
                    a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                elif j==30:
                    A=a[i+1,j,k-1]+a[i,0,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    C=sgm[i,j]/(2*dt)
                    a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                else:
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    C=sgm[i,j]/(2*dt)
                    a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
    
    do_anim=0
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
            
        anim01.save('front_pla_normal_ABC.mp4', writer=writer)
        
"""
funció que calcula el coeficient de reflexió per una ona com la que es 
modela en el programa principal
"""

def p_s(t,amp,w):
    val=amp*np.sin(w*t)
    return val

def reflection_coefficient(sgm_max,gruix_paret,m):
    a=np.zeros((101,31,301))
    s=np.zeros((101,31,301))
    sgm=np.zeros((101,31,301))
    
    c=1.4
    dt=0.05
    dl=0.1
    w=5
    amp=5
    rao=(c*dt/dl)**2
    
    for i in range(100-gruix_paret,100):
        for j in range(0,31):
            sgm[i,j,:]=sgm_max*((i-99+gruix_paret)/gruix_paret)**m
    
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
    
    amp_incident=np.max(a[30,15,0:150])
    amp_reflectida=np.max(a[30,15,151:300])
    
    ref_coef=amp_reflectida/amp_incident
    
    return(ref_coef)

r_c=np.zeros((100,10))

file1=open("coeficient_de_reflexio_v2.txt","a")
L = ["sgm max  ","gruix paret  ","m  ","coeficient d'absorció","\n"]
file1.writelines(L)

for i in range(100):
    for j in range(10):
        sgm_max=0.02+0.0005*i
        m=4+0.03*j
        gruix_paret=20
        r_c[i,j]=reflection_coefficient(sgm_max,gruix_paret,m)
        L=[str(sgm_max)+" - ",str(gruix_paret)+" - ",
           str(m)+" - ",str(r_c[i,j])+" - ","\n"]
        file1.writelines(L)

print(np.min(r_c))
print(np.where(r_c==np.min(r_c)))

file1.close()
            
        
            
    
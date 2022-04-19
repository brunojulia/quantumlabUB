#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:58:27 2022

@author: annamoresoserra
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn




L=3.
m=1.
hbar=1.
tb=2.
ta=0.
w=2.
deltax=0.5
deltay=deltax
deltat=0.01
Nx=int((2.*L)/deltax)
Ny=Nx
Nt=int((tb-ta)/deltat)



# Gràifc del temps de compilació segons dx
#dx=np.array([0.01,0.012])
#dx=np.append(dx,[0.015+i*0.005 for i in range (0,12)])

#t1=np.zeros((14))
#t2=np.zeros((14))
#t3=np.zeros((14))


#for i in range (np.size(dx)):
#    dades1=np.load('dadeshvh{:.3f}dx0.01dt.npy'.format(dx[i]))
#    t1[i]=dades1[3]
#    dades2=np.load('dadeshvh{:.3f}dx0.02dt.npy'.format(dx[i]))
#    t2[i]=dades2[3]
#    dades3=np.load('dadeshvh{:.3f}dx0.03dt.npy'.format(dx[i]))
#    t3[i]=dades3[3]


#fig=plt.figure(figsize=[10,6])
#ax=plt.subplot(111)
#plt.title('Computing time')


#plt.ylabel("t",fontsize=12)
#plt.xlabel("dx",fontsize=11)
#plt.minorticks_on()
#plt.plot(dx,t1,".-",color="#003f5c",label="dt=0.01")
#plt.plot(dx,t2,".-",color="#58508d",label="dt=0.02")
#plt.plot(dx,t3,".-",color="#bc5090",label="dt=0.03")
#plt.legend(loc=0,fontsize=11)
#plt.savefig("computing_time.png",dpi=300)





        
    

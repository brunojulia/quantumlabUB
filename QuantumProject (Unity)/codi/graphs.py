#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:22:34 2022

@author: annamoresoserra
"""
import matplotlib.pyplot as plt
import numpy as np

# Conservation of the norm
def norm_graph(dx, dt, V_type, wave_type):
    fig=plt.figure(figsize=[10,6])
    plt.title('Norm conservation of a {} wave packet in a {} potential'.format((wave_type, V_type)), fontsize = 13)
    normavector=np.load('normatot_dx{}dt{}.npy'.format(dx,dt))
    tvec=np.load('t_dx{}dt{}.npy'.format(dx,dt))
    plt.plot(tvec, normavector, '.', color="#003f5c")
    plt.xticks(fontsize = 13)
    plt.xlabel('t', fontsize = 13)
    #plt.ylabel('($1-|\psi^{*}\psi|$)·$10^9$',fontsize=13)
    plt.ylabel('$|\psi^{*}\psi|$', fontsize = 13)
    plt.legend(fontsize = 13)
    plt.show()
'''
    #plt.savefig("norm_conservation_gauss.png",dpi=300)
#plt.plot(tvec,(1-normavector)*10**9,'.',color="#003f5c",label='$p_{0x}$=0; $p_{0y}$=8')	
#plt.ticklabel_format(useOffset=False, style='plain')
#mit=0.
#for i in range (len(normavector)):
#    mit=mit+normavector[i]
    
#mit=mit/(len(normavector))
#sd=0.
#for i in range (len(normavector)):
#    sd=sd+(normavector[i]-mit)**2

#sd=np.sqrt(sd/len(normavector))

#print(mit)
#print(sd)
#y_ticks = [3.53785,3.53780, 3.53775, 3.53770]
#y_tick_labels = ['3.53785', '3.53780', '3.53775', '3.53770']
#plt.yticks(y_ticks,fontsize=13)
plt.xticks(fontsize=13)
plt.xlabel('t',fontsize=13)
#plt.ylabel('($1-|\psi^{*}\psi|$)·$10^9$',fontsize=13)
plt.ylabel('$|\psi^{*}\psi|$',fontsize=13)
plt.legend(fontsize=13)
plt.show()
#plt.savefig("norm_conservation_gauss.png",dpi=300)
'''


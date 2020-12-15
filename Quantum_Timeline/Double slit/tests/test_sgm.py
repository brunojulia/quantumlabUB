# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:48:24 2020

@author: llucv
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

e=np.e
pi=np.pi

#dades del recinte i visualització
Nx=301
Ny=301
Nt=5
h_display=100
w_display=200

#dades parets: detector i escletxes
sgm=np.zeros((Nx,Ny))
sgm_max=0.03
m=4.1

    #parets del detecor
sgm_det=np.zeros((Nx,Ny))
w_det=101

    #sgm_wall dependra del temps, per afegir i treure escletxes
sgm_wall=np.zeros((Nx,Ny))
w_wall=20
x_wall=105

#sigma a les parets del detector
w_det=int(Ny/3)+1
for k in range(w_det):
    sgm_det[Nx-1-k,0+k:Ny-1-k]=sgm_max*((w_det-k)/w_det)**m
    sgm_det[x_wall:Nx-k,k]=sgm_max*((w_det-k)/w_det)**m
    sgm_det[x_wall:Nx-k,Ny-1-k]=sgm_max*((w_det-k)/w_det)**m

plt.imshow(sgm_det.transpose(),vmin=0,origin='lower',extent=(0,30,0,30))
plt.savefig('sigma_detector.png')


for k in range(Nt):
#dades que poden canviar en el temps (el jugador)  
    Nsplt=2
    w_splt=6
    
    splt_i=np.zeros((Nsplt+2),dtype=int)
    splt_f=np.zeros((Nsplt+2),dtype=int)
    splt_n=np.linspace(1,Nsplt,Nsplt,dtype=int)
    wall_presence=np.zeros((Ny),dtype=int)
    
    #posicio del final i l'inici de cada escletxa
    splt_i[1:Nsplt+1]=int(Ny/2)-int(h_display/2)-int(w_splt/2)\
                +splt_n[:]*int(h_display/(1+Nsplt))
    splt_f[1:Nsplt+1]=int(Ny/2)-int(h_display/2)+int(w_splt/2)\
                +splt_n[:]*int(h_display/(1+Nsplt))
    splt_f[0]=0
    splt_f[Nsplt+1]=Ny
    splt_i[Nsplt+1]=Ny
    
    #les escletxes van de splt_i a splt_f-1, en aquests punts no hi ha paret,
    # a slpt_f ja hi ha paret
    for n in range(1,Nsplt+2):
        wall_presence[splt_f[n-1]:splt_i[n]]=1
        wall_presence[splt_i[n]:splt_f[n]]=0
    
    # matriu que, amb el gruix de la paret com a nombre de files, ens diu si 
    # hi ha paret o escletxes a cada una de les y(representades en les columnes)
    wall_presence=np.tile(np.array([wall_presence],dtype=int),(w_wall,1))

    #matriu que diu com de "dins" som a la paret
    wall_n=np.linspace(1,w_wall,w_wall)
    wall_ny=np.tile(np.array([wall_n],dtype=int).transpose(),(1,Ny))

    #valors de coeficient d'absorció a les parets
    sgm_wall[x_wall-w_wall:x_wall,:]=wall_presence[:,:]\
                *sgm_max*((wall_ny[:,:])/w_wall)**m
    
    sgm=sgm_wall+sgm_det

    



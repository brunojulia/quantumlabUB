# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 17:03:42 2020

@author: llucv

Aquest programa serà una versió, ja, del final per aquest experiment.
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
Nt=1001
h_display=100
w_display=200

#dades parets: detector i escletxes
sgm=np.zeros((Nx,Ny))
sgm_max=0.06
m=4.1

    #parets del detecor
sgm_det=np.zeros((Nx,Ny))
w_det=101

    #sgm_wall dependra del temps, per afegir i treure escletxes
sgm_wall=np.zeros((Nx,Ny))
w_wall=25
x_wall=105

#sigma a les parets del detector, ho defineixo ja perquè no canviarà
w_det=int(Ny/3)+1
for k in range(w_det):
    sgm_det[Nx-1-k,0+k:Ny-1-k]=sgm_max*((w_det-k)/w_det)**m
    sgm_det[x_wall:Nx-k,k]=sgm_max*((w_det-k)/w_det)**m
    sgm_det[x_wall:Nx-k,Ny-1-k]=sgm_max*((w_det-k)/w_det)**m
    
#llista amb l'amplitud de l'ona en funció del temps i el punt
a=np.zeros((Nx,Ny,Nt))

#llista amb els valors de la font (source)
s=np.zeros((Nx,Ny))

#dades de l'ona
w=5
c=1.4
amp=5

#dades de la discretització (intervals espacials i temporals)
dl=0.1
dt=0.05
rao=(c*dt/dl)**2

#font d'ones sinussoïdal (point_source)
def p_s(t,amp,w):
    val=amp*np.sin(w*t)
    return val

start=time.time()

for k in range(2,Nt):
    t=(k-2)*dt
    s[1,:]=p_s(t,amp,w)
    
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
    
    #resolució de l'equació d'ones a cada temps a l'interior del recinte
    a[1:-1,1:-1,k]=(rao*(a[2:,1:-1,k-1]+a[0:-2,1:-1,k-1]\
                    +a[1:-1,2:,k-1]+a[1:-1,0:-2,k-1]\
                    -4*a[1:-1,1:-1,k-1])+s[1:-1,1:-1]\
                +2*a[1:-1,1:-1,k-1]-a[1:-1,1:-1,k-2]\
                +sgm[1:-1,1:-1]*a[1:-1,1:-1,k-2]/(2*dt))\
                /(1+sgm[1:-1,1:-1]/(2*dt))
    
    #condicions periòdiques de contorn a les parets superior i inferior
    a[1:x_wall,0,k]=(rao*(a[2:x_wall+1,0,k-1]+a[0:x_wall-1,0,k-1]\
                    +a[1:x_wall,1,k-1]+a[1:x_wall,Ny-1,k-1]\
                    -4*a[1:x_wall,0,k-1])+s[1:x_wall,0]\
                +2*a[1:x_wall,0,k-1]-a[1:x_wall,0,k-2]\
                +sgm[1:x_wall,0]*a[1:x_wall,0,k-2]/(2*dt))\
                /(1+sgm[1:x_wall,0]/(2*dt))
                
    a[1:x_wall,Ny-1,k]=(rao*(a[2:x_wall+1,Ny-1,k-1]+a[0:x_wall-1,Ny-1,k-1]\
                    +a[1:x_wall,0,k-1]+a[1:x_wall,Ny-2,k-1]\
                    -4*a[1:x_wall,Ny-1,k-1])+s[1:x_wall,Ny-1]\
                +2*a[1:x_wall,Ny-1,k-1]-a[1:x_wall,Ny-1,k-2]\
                +sgm[1:x_wall,Ny-1]*a[1:x_wall,Ny-1,k-2]/(2*dt))\
                /(1+sgm[1:x_wall,Ny-1]/(2*dt))

elapsed_time=(time.time()-start)
print(elapsed_time)

#imatge sigmes
sgma=sgm.transpose()
plt.imshow(sgma[int((Ny-h_display)/2):int((Ny+h_display)/2),\
                           0:w_display]\
           ,vmin=0,origin='lower',\
            extent=(0,int(w_display*0.1),0,int(h_display*0.1)))
plt.savefig(str(Nsplt)+'splits_sigma.png')

#animació
start=time.time()

def update(frame):
    k=frame*5
    at=a[:,:,k]
    plt.imshow(at.transpose()[int((Ny-h_display)/2):int((Ny+h_display)/2),\
                              0:w_display]
               ,vmax=10,vmin=-10,origin='lower',\
                extent=(0,int(w_display*0.1),0,int(h_display*0.1)))

fig = plt.figure()
ax1 = plt.subplot()

Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim = ani.FuncAnimation(fig, update, 
                               frames = int((Nt-1)/5), 
                               blit = False, interval=100)

anim.save(str(Nsplt)+'splits_plane_wave.mp4', writer=writer)

elapsed_time=(time.time()-start)
print(elapsed_time)










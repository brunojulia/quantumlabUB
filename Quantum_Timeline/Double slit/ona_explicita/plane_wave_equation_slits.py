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

#integral per trapezis, on h és el pas, i f, una llista amb els valors de 
#la funció a inetgrar, és una llista 3D d'una funció 2D amb dependència temps
#i com que estem integrant en el temps ens retorna una llista 2D
def trapezis(h,f):
    val=(np.sum(f[:,:,0:-1],axis=2)+np.sum(f[:,:,1:],axis=2))*h/2
    return val

#dades del recinte i visualització
Nx=301
Ny=301
Nt=1001
h_display=100
w_display=200

#dades parets: detector i escletxes
sgm=np.zeros((Nx,Ny))
sgm_max=0.02615
m=1.54

    #parets del detecor
sgm_det=np.zeros((Nx,Ny))
w_det=51

    #sgm_wall dependra del temps, per afegir i treure escletxes
sgm_wall=np.zeros((Nx,Ny))
w_wall=10
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
w=2*pi
c=1.4
amp=2.5
tau=2*pi/w

#font d'ones sinussoïdal (point_source)
def p_s(t,amp,w):
    val=amp*np.sin(w*t)
    return val

#dades de la discretització (intervals espacials i temporals)
dl=0.1
dt=0.05
rao=(c*dt/dl)**2

#nombre de passos temporals en un període
Ntau=int(tau/dt+1)

#funció que calcula l'amplitud al quadrat de l'ona
def amplitud2(psi):
    val=(abs(psi))**2
    return val

#llista que tindrà els valors de l'amplitud al quadrat en un període
a2=np.zeros((Nx,Ny,Ntau))

#llista amb els valors d'intensitat de l'ona
inty=np.zeros((Nx,Ny,Nt))

start=time.time()

for k in range(2,Nt):
    t=(k-2)*dt
    s[1,:]=p_s(t,amp,w)
    
    #dades que poden canviar en el temps (el jugador)  
    Nslt=2
    w_slt=8
    
    slt_i=np.zeros((Nslt+2),dtype=int)
    slt_f=np.zeros((Nslt+2),dtype=int)
    slt_n=np.linspace(1,Nslt,Nslt,dtype=int)
    wall_presence=np.zeros((Ny),dtype=int)
    
    #posicio del final i l'inici de cada escletxa
    slt_i[1:Nslt+1]=int(Ny/2)-int(h_display/2)-int(w_slt/2)\
                +slt_n[:]*int(h_display/(1+Nslt))
    slt_f[1:Nslt+1]=int(Ny/2)-int(h_display/2)+int(w_slt/2)\
                +slt_n[:]*int(h_display/(1+Nslt))
    slt_f[0]=0
    slt_f[Nslt+1]=Ny
    slt_i[Nslt+1]=Ny
    
    #les escletxes van de splt_i a splt_f-1, en aquests punts no hi ha paret,
    # a slpt_f ja hi ha paret
    for n in range(1,Nslt+2):
        wall_presence[slt_f[n-1]:slt_i[n]]=1
        wall_presence[slt_i[n]:slt_f[n]]=0
    
    # matriu que, amb el gruix de la paret com a nombre de files, ens diu si 
    # hi ha paret o escletxes a cada una de les y(representades en les columnes)
    wall_presence=np.tile(np.array([wall_presence],dtype=int),(w_wall,1))

    #matriu que diu com de "dins" som a la paret
    wall_n=np.linspace(1,w_wall,w_wall)
    wall_ny=np.tile(np.array([wall_n],dtype=int).transpose(),(1,Ny))

    #valors de coeficient d'absorció a les parets
    sgm_wall[x_wall-w_wall:x_wall,:]=wall_presence[:,:]\
                *sgm_max*((wall_ny[:,:])/w_wall)**m
    
    sgm_wall[x_wall,:]=wall_presence[0,:]*1000
    
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
                
    #calculs de la intensitat
    a2[:,:,:-1]=a2[:,:,1:]
    a2[:,:,Ntau-1]=a[:,:,k]*a[:,:,k]
    
    inty[:,:,k]=trapezis(dt,a2)/tau
    

elapsed_time=(time.time()-start)
print(elapsed_time)

#imatge sigmes
sgma=sgm.transpose()
im2=plt.imshow(sgma[int((Ny-h_display)/2):int((Ny+h_display)/2),\
                           0:w_display]\
           ,vmax=0.03,origin='lower',\
            extent=(0,int(w_display*0.1),0,int(h_display*0.1)))
cbar=plt.colorbar(im2)
cbar.set_label("Coeficient d'absorció")
plt.savefig(str(Nslt)+'slits_sigma_display.png')


im3=plt.imshow(sgma\
           ,vmax=0.03,origin='lower',\
            extent=(0,int(Nx*0.1),0,int(Ny*0.1)))

plt.savefig(str(Nslt)+'slits_sigma_recint.png')

#animació
start=time.time()

def update1(frame):
    k=frame*5
    at=a[:,:,k]
    plt.imshow(at.transpose()[int((Ny-h_display)/2):int((Ny+h_display)/2),\
                              0:w_display]
               ,vmax=4,vmin=-4,origin='lower',\
                extent=(0,int(w_display*dl),0,int(h_display*dl)))

fig1,ax1 = plt.subplots()

Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim1 = ani.FuncAnimation(fig1, update1, 
                               frames = int((Nt-1)/5), 
                               blit = False, interval=100)

anim1.save(str(w_slt)+'width_'+str(Nslt)+'slits_plane_wave.mp4', writer=writer)

elapsed_time=(time.time()-start)
print(elapsed_time)

start=time.time()

def update2(frame):
    k=frame*5
    it=inty[:,:,k]
    plt.imshow(it.transpose()[int((Ny-h_display)/2):int((Ny+h_display)/2),\
                              0:w_display]
               ,vmax=5,origin='lower',\
                extent=(0,int(w_display*dl),0,int(h_display*dl)),cmap="gray")

fig2,ax1 = plt.subplots()

Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

anim2 = ani.FuncAnimation(fig2, update2, 
                               frames = int((Nt-1)/5), 
                               blit = False, interval=100)

anim2.save(str(w_slt)+'width_'+str(Nslt)+'slits_intensity.mp4', writer=writer)

elapsed_time=(time.time()-start)
print(elapsed_time)

plt.figure(5)
plt.plot(inty[200,50:250,900])
plt.savefig(str(w_slt)+'width'+str(Nslt)+'slits_intensity_detector.png')












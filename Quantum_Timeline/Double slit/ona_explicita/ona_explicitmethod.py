# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:50:23 2020

@author: llucv
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import time

e=np.e
pi=np.pi


"""
el següent és una prova

"""
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
        
xmax=10
ymax=10
h=0.1
p=0.05
c=1

u=np.zeros((101,101,3))
g=np.zeros((101,101))
Ut=np.zeros((101,101))
Uxt=np.zeros((101))

"""
El que s'intenta aquí és modelar les primers intants de l'ona plana
però l'ona resultant no actua com una plana, ja que té massa poca amplada
en la direcció perpendicular a la de direcció
"""


a=np.zeros((101,101,3))
g=np.zeros((101,101))
s=np.zeros((101,101))
at=np.zeros((101,101))    
    
def point_source(t):
    val=50*np.sin(t)
    return val

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

"""
d'ara en endavant l'amplitud serà una llista amb una dimensió més (temps)
i es representaran uns temps concrets d'aquesta llista
El primer cas amb aquest metode és un pols d'ona que es propaga radialment 
(amb simetria esfèrica)
"""
    
a=np.zeros((101,101,1001))
g=np.zeros((101,101))
s=np.zeros((101,101))
at=np.zeros((101,101))


xo=5
yo=5
rao=(c*p/h)**2    

do3=0
if do3==1:
    for i in range(101):
        for j in range(101):
            x=i*h
            y=j*h
            r=((x-xo)**2+(y-yo)**2)**(1/2)
            if r < 0.5:
             a[i,j,0]=10*np.cos(pi*r)
             A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]
             a[i,j,1]=a[i,j,0]+rao*A/2
    
    for k in range(2,1001):
        for j in range(1,99):
            for i in range(1,99):
                A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]-4*a[i,j,k-1]
                B=2*a[i,j,k-1]-a[i,j,k-2]
                a[i,j,k]=rao*A+B
                
    def update2(frame):
        k=frame*10
        for i in range(101):
           for j in range(101):
               at[i,j]=a[i,j,k]
        plt.imshow(at,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,10))
    
    fig3 = plt.figure()
    ax1 = plt.subplot()
    
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    anim3 = ani.FuncAnimation(fig3, update2, 
                                   frames = 100, 
                                   blit = False, interval=200)
    
    guarda3=0
    if guarda3==1:
        anim3.save('ona_explicit_esferica_v2.mp4', writer=writer)


"""
Pel següent cas s'ha fet que els límits de la caixa on es propaga l'ona es 
trobin lluny de la figura representada per així evitar reflexions amb els 
extrems (el calcul triga massa i s'ha tornat al cas inicial)
"""
    
a=np.zeros((101,101,1001))
g=np.zeros((101,101))
s=np.zeros((101,101))
at=np.zeros((101,101))

c=1
p=0.05
h=0.1
xo=50
yo=50
rao=(c*p/h)**2

def p_s(t):
    val=10*np.sin(c*t*3)
    return val

do4=0
if do4==1:
    for i in range(1,99):
        for j in range(1,99):
            s[50,50]=p_s(0)
            A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]+s[i,j]
            a[i,j,1]=rao*A+a[i,j,0]
            
    for k in range(2,1001):
        for j in range(1,99):
            for i in range(1,99):
                t=(k-1)*p
                s[50,50]=p_s(t)
                A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]-4*a[i,j,k-1]\
                    +s[i,j]
                B=2*a[i,j,k-1]-a[i,j,k-2]
                a[i,j,k]=rao*A+B
            
    def update4(frame):
        k=frame*5
        for i in range(101):
           for j in range(101):
               at[i,j]=a[i,j,k]
        plt.imshow(at,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,10))

    fig4 = plt.figure()
    ax1 = plt.subplot()

    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    anim4 = ani.FuncAnimation(fig4, update4, 
                                   frames = 100, 
                                   blit = False, interval=200)

    guarda4=0
    if guarda4==1:
        anim4.save('ona_explicit_esferica_v3.mp4', writer=writer)
    
"""
El següent cas és utilitzar el cas anterior per fer una ona plana, amb algunes
modificacions. L'ona plana es modelarà com un conjunt de fonts puntuals que 
oscil·len 

"""
a=np.zeros((401,101,1001))
g=np.zeros((401,101))
s=np.zeros((401,101))

c=1
p=0.05
h=0.1
rao=(c*p/h)**2

def p_s(t):
    val=5*np.sin(c*t*4)
    return val

do5=0
if do5==1:
    for i in range(1,399):
        s[i,1]=p_s(0)
        for j in  range(1,99):
             A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]+s[i,j]
             a[i,j,1]=rao*A+a[i,j,0]
    
    for k in range(2,1001):
        for i in range(1,399):
            t=(k-1)*p
            s[i,1]=p_s(t)
            for j in range(1,99):
                A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                  -4*a[i,j,k-1]+s[i,j]
                B=2*a[i,j,k-1]-a[i,j,k-2]
                a[i,j,k]=rao*A+B
    
    def update5(frame):
        k=frame*5
        plt.imshow(a[150:250,3:100,k]
                   ,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,10))
    
    fig5 = plt.figure()
    ax1 = plt.subplot()
    
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    anim5 = ani.FuncAnimation(fig5, update5, 
                                   frames = 200, 
                                   blit = False, interval=200)
    
    guarda5=1
    if guarda5==1:
        anim5.save('ona_explicit_plana_v2.mp4', writer=writer)


"""
Afegim escletxes (1) a la ona plana, es modelen com zones de la caixa on 
l'amplitud de l'ona és 0.
"""
a=np.zeros((401,101,1001))
g=np.zeros((401,101))
s=np.zeros((401,101))

c=1
p=0.05
h=0.1
rao=(c*p/h)**2

def p_s(t):
    val=5*np.sin(c*t*4)
    return val

do6=0
if do6==1:
    for i in range(1,399):
        s[i,1]=p_s(0)   
        for j in  range(1,99):
             if j==50:
                 if i in range(195,206):
                     A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]\
                       +s[i,j]
                     a[i,j,1]=rao*A+a[i,j,0]
                 else:
                     a[i,j,1]=0
             else:
                 A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]\
                   +s[i,j]
                 a[i,j,1]=rao*A+a[i,j,0]
    
    for k in range(2,1001):
        for i in range(1,399):
            t=(k-1)*p
            s[i,1]=p_s(t)
            for j in range(1,99):
                if j==50:                    
                    if i in range(195,206):
                        A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                          -4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        a[i,j,k]=rao*A+B
                    else:
                        a[i,j,k]=0
                else:
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    a[i,j,k]=rao*A+B
    
    def update6(frame):
        k=frame*5
        plt.imshow(a[150:250,3:100,k]
                   ,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,10))
    
    fig6 = plt.figure()
    ax1 = plt.subplot()
    
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    anim6 = ani.FuncAnimation(fig6, update6, 
                                   frames = 200, 
                                   blit = False, interval=200)
    
    guarda6=0
    if guarda6==1:
        anim6.save('ona_explicit_1esceltxa.mp4', writer=writer)

"""
Afegim escletxes (2) a la ona plana, es modelen com zones de la caixa on 
l'amplitud de l'ona és 0.
"""
a=np.zeros((401,101,1001))
g=np.zeros((401,101))
s=np.zeros((401,101))

c=1
p=0.05
h=0.1
rao=(c*p/h)**2

def p_s(t):
    val=5*np.sin(c*t*4)
    return val

do7=0
if do7==1:
    for i in range(1,399):
        s[i,1]=p_s(0)   
        for j in  range(1,99):
             if j==50:
                 if i in range(170,181) or i in range(220,231):
                     A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]\
                       +s[i,j]
                     a[i,j,1]=rao*A+a[i,j,0]
                 else:
                     a[i,j,1]=0
             else:
                 A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]\
                   +s[i,j]
                 a[i,j,1]=rao*A+a[i,j,0]
    
    for k in range(2,1001):
        for i in range(1,399):
            t=(k-1)*p
            s[i,1]=p_s(t)
            for j in range(1,99):
                if j==50:                    
                    if i in range(170,181) or i in range(220,231):
                        A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                          -4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        a[i,j,k]=rao*A+B
                    else:
                        a[i,j,k]=0
                else:
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    a[i,j,k]=rao*A+B
    
    def update7(frame):
        k=frame*5
        plt.imshow(a[150:250,3:100,k]
                   ,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,10))
    
    fig7 = plt.figure()
    ax1 = plt.subplot()
    
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    anim7 = ani.FuncAnimation(fig7, update7, 
                                   frames = 20, 
                                   blit = False, interval=200)
    
    guarda7=1
    if guarda7==1:
        anim7.save('ona_explicit_2esceltxes.mp4', writer=writer)

"""
L'ultim cas d'aquest programa: s'ha afegit al cas de la font puntual periòdica 
unes parets amb gruix i coeficient d'absorció que incrementa amb l'endinsament
a la paret per evitar reflexions
"""
    
a=np.zeros((101,101,1001))
g=np.zeros((101,101))
s=np.zeros((101,101))
sgm=np.zeros((101,101))
at=np.zeros((101,101))

c=1.4
w=6
p=0.05
h=0.1
xo=50
yo=50
rao=(c*p/h)**2

xo=50
yo=50

def p_s(t):
    val=10*np.sin(w*t)
    return val

do8=1
if do8==1:
        
    for k in range(11):
        for i in range(0+k,101-k):
            sgm[i,k]=0.1*((10-k)/10)**3
            sgm[i,100-k]=0.1*((10-k)/10)**3
            sgm[k,i]=0.1*((10-k)/10)**3
            sgm[100-k,i]=0.1*((10-k)/10)**3


    print(sgm[0,50])
    print(sgm[100,50])
    plt.imshow(sgm,vmin=0,origin='lower',extent=(0,10,0,10))
    plt.savefig('sigma.png')
    
    for i in range(101):
        for j in range(101):
            x=i*h
            y=j*h
            r=((x-xo)**2+(y-yo)**2)**(1/2)
            if r < 0.5:
                print(r)
                a[i,j,0]=10*np.cos(pi*r)
                A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]
                a[i,j,1]=a[i,j,0]+rao*A/2
        print(r)
             
    plt.imshow(a[:,:,0],vmin=0,origin='lower',extent=(0,10,0,10))
    plt.savefig('sigma.png')
    
    for k in range(2,501):
        for j in range(1,100):
            for i in range(1,100):
                A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                  -4*a[i,j,k-1]+s[i,j]
                B=2*a[i,j,k-1]-a[i,j,k-2]
                C=sgm[i,j]/(2*p)
                a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)

    def update8(frame):
        k=frame*5
        for i in range(101):
           for j in range(101):
               at[i,j]=a[i,j,k]
        plt.imshow(at,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,10))

    fig8 = plt.figure()
    ax1 = plt.subplot()

    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    anim8 = ani.FuncAnimation(fig8, update8, 
                                   frames = 200, 
                                   blit = False, interval=200)

    guarda8=1
    if guarda8==1:
        anim8.save('ona_exp_esf_abscoef.mp4', writer=writer)

"""
L'ultim cas d'aquest programa: utilitzant el cas d'un sol front d'ona esfèric,
posem unes noves condicions de contorn als extrems (Engquist) per veure si 
s'absorveix l'ona amb aquestes.
"""
a=np.zeros((101,101,1001))
g=np.zeros((101,101))
s=np.zeros((101,101))
at=np.zeros((101,101))


xo=5
yo=5
rao=(c*p/h)**2    

do9=0
if do9==1:
    for i in range(101):
        for j in range(101):
            x=i*h
            y=j*h
            r=((x-xo)**2+(y-yo)**2)**(1/2)
            if r < 0.5:
             a[i,j,0]=10*np.cos(pi*r)
             A=a[i+1,j,0]+a[i,j+1,0]+a[i-1,j,0]+a[i,j-1,0]-4*a[i,j,0]
             a[i,j,1]=a[i,j,0]+rao*A/2
    
    for k in range(2,1001):
        for j in range(0,101):
            for i in range(0,101):
                if i==0 or i==100 or j==0 or j==100:
                    if i==0:
                        a[i,j,k]=(p/h)*(a[i+1,j,k-1]-a[i,j,k-1])+a[i,j,k]
                    if i==100:
                        a[i,j,k]=(p/h)*(a[i,j,k-1]-a[i-1,j,k-1])+a[i,j,k]
                    if j==0:
                        a[i,j,k]=(p/h)*(a[i,j+1,k-1]-a[i,j,k-1])+a[i,j,k]
                    if j==100:
                        a[i,j,k]=(p/h)*(a[i,j,k-1]-a[i,j-1,k-1])+a[i,j,k]
                else:
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    a[i,j,k]=rao*A+B
                      
    def update9(frame):
        k=frame*10
        for i in range(101):
           for j in range(101):
               at[i,j]=a[i,j,k]
        plt.imshow(at,vmax=10,vmin=-10,origin='lower',extent=(0,10,0,10))
    
    fig9 = plt.figure()
    ax1 = plt.subplot()
    
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    anim9 = ani.FuncAnimation(fig9, update9, 
                                   frames = 100, 
                                   blit = False, interval=200)
    
    guarda9=0
    if guarda9==1:
        anim9.save('ona_engquist_bc.mp4', writer=writer)
        
"""
Afegim escletxes i definim noves condicions de contorn
"""
a=np.zeros((301,301,1001))
u=np.zeros((301,301))
ut=np.zeros((301,301))
g=np.zeros((301,301))
s=np.zeros((301,301))
sgm=np.zeros((301,301))

c=1.4
p=0.05
h=0.1
w=5
amp=5
sgm_max=0.1
rao=(c*p/h)**2

gruix=10
splt1_i=160
splt1_f=splt1_i+gruix
splt2_i=130
splt2_f=splt2_i+gruix



def p_s(t):
    val=amp*np.sin(w*t)
    return val

do10=0
if do10==1:
    
    #parets del detector amb coeficients d'absorció
    
    for k in range(101):
        for j in range(0+k,301-k):
            sgm[300-k,j]=sgm_max*((101-k)/101)**3
        for i in range(100,301-k):
            sgm[i,k]=sgm_max*((101-k)/101)**3
            sgm[i,300-k]=sgm_max*((101-k)/101)**3

    #paret amb escletxes programat amb coeficients d'absorció 
    
    for i in range(94,105):
        for j in range(0,301):
            if j in range(splt1_i,splt1_f+1) or j in range(splt2_i,splt2_f+1):
                sgm[i,j]=0
            else:
                sgm[i,j]=sgm_max*((i-93)/11)**3
    
    plt.imshow(sgm.transpose(),vmin=0,origin='lower',extent=(0,10,0,10))
    plt.savefig('sigma2.png')
    
    plt.imshow(sgm[280:300,280:300])


    start=time.time()

    for k in range(1,1001):
        for j in range(0,301):
            t=(k-1)*p
            s[1,j]=p_s(t)
            for i in range(1,105):
                
                # paret superior i inferior
                if j==0:
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,300,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    C=sgm[i,j]/(2*p)
                    a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                if j==300:
                    A=a[i+1,j,k-1]+a[i,0,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    C=sgm[i,j]/(2*p)
                    a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                
                # escletxtes
                if i in range(94,105):
                    #contacte amb la paret
                    if j==splt1_f:
                        A=a[i+1,j,k-1]+a[i,splt1_i,k-1]+a[i-1,j,k-1]\
                            +a[i,j-1,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    if j==splt1_i:
                        A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]\
                            +a[i,splt1_f,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    if j==splt2_f:
                        A=a[i+1,j,k-1]+a[i,splt2_i,k-1]+a[i-1,j,k-1]\
                            +a[i,j-1,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    if j==splt2_i:
                        A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]\
                            +a[i,splt2_f,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    
                    #paret en si
                    if j==splt1_f+1:
                        A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]\
                            +a[i,splt1_i-1,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    if j==splt1_i-1:
                        A=a[i+1,j,k-1]+a[i,splt1_f+1,k-1]+a[i-1,j,k-1]\
                            +a[i,j-1,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    if j==splt2_f+1:
                        A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]\
                            +a[i,splt2_i-1,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    if j==splt2_i-1:
                        A=a[i+1,j,k-1]+a[i,splt2_f+1,k-1]+a[i-1,j,k-1]\
                            +a[i,j-1,k-1]-4*a[i,j,k-1]+s[i,j]
                        B=2*a[i,j,k-1]-a[i,j,k-2]
                        C=sgm[i,j]/(2*p)
                        a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                
                #interior del recinte
                if j in range(1,300):
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    C=sgm[i,j]/(2*p)
                    a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
            
            #zona del detector
            for i in range(105,300):
                if j==0 or j==300:
                    a[i,j,k]=0
                else:
                    A=a[i+1,j,k-1]+a[i,j+1,k-1]+a[i-1,j,k-1]+a[i,j-1,k-1]\
                      -4*a[i,j,k-1]+s[i,j]
                    B=2*a[i,j,k-1]-a[i,j,k-2]
                    C=sgm[i,j]/(2*p)
                    a[i,j,k]=(rao*A+B+C*a[i,j,k-2])/(1+C)
                    
    elapsed_time=(time.time()-start)
    print(elapsed_time)
            
    def update10(frame):
        k=frame*5
        u=a[:,:,k]
        plt.imshow(u.transpose()[0:201,100:201]
                   ,vmax=10,vmin=-10,origin='lower',extent=(0,20,0,10))
    
    fig10 = plt.figure()
    ax1 = plt.subplot()
    
    Writer = ani.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    
    anim10 = ani.FuncAnimation(fig10, update10, 
                                   frames = int((1001-1)/5), 
                                   blit = False, interval=100)
    
    guarda10=1
    if guarda10==1:
        anim10.save('ona_plana_2esceltxes.mp4', writer=writer)
        




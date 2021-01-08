# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 11:17:37 2020

@author: llucv
"""

import numpy as np

e=np.e
pi=np.pi
        
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
    w=6.5
    amp=5
    rao=(c*dt/dl)**2
    
    for i in range(100-gruix_paret,100):
        for j in range(0,31):
            sgm[i,j,:]=sgm_max*np.exp(-m*((i-99+gruix_paret)/gruix_paret))
    
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
sgm_max=0.012
gruix_paret=10
m=1
print(reflection_coefficient(sgm_max,gruix_paret,m))

r_c=np.zeros((30,30))

file1=open("coeficient_de_reflexio_v2.txt","a")
L = ["sgm max  ","gruix paret  ","m  ","coeficient d'absorció","\n"]
file1.writelines(L)

for i in range(30):
    for j in range(30):
        sgm_max=0.005+0.0005*i
        m=0.05+0.005*j
        gruix_paret=10
        r_c[i,j]=reflection_coefficient(sgm_max,gruix_paret,m)
        L=[str(sgm_max)+" - ",str(gruix_paret)+" - ",
           str(m)+" - ",str(r_c[i,j])+" - ","\n"]
        file1.writelines(L)

print(np.min(r_c))
print(np.where(r_c==np.min(r_c)))


file1.close()
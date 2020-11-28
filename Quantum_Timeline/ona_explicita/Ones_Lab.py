import numpy as np
import matplotlib
import matplotlib.pyplot as plt
e=np.e
pi=np.pi

print(e)
print(pi)
print(e**(1j*pi))

def ona_plana(x,y,t,c,x0):
    if x>x0:
        valor=0
    elif x>c*t:
        valor=0
    else:
        valor=np.cos(x-c*t)
    return valor

def ona_esf(x,y,x0,y0,t,c):
    r=((x-x0)**2+(y-y0)**2)**(1/2)
    if x<x0:
        valor=0
    elif r>c*t:
        valor=0
    else:
        valor=np.cos(r-c*t+x0)
    return valor

def intensitat(psi):
    val=(abs(psi))**2
    return val

Lab=np.zeros((101,201))
Int=np.zeros((101,201))
x=np.zeros((201))
y=np.zeros((101))

xmax=50*pi
ymax=25*pi
pas=ymax/100
print(xmax,ymax,pas)

for i in range(201):
    x[i]=pas*i
    if i<101:
        y[i]=pas*i
        
print(x[0],x[200])
print(y[0],y[100])

""" 
Primer cas: 
fixem el temps per representar una situació instantània
"""
t=300
x0=40
y1=55
y2=15
c=1

print(ona_esf(45,5,x0,y1,t,c))

for i in range(201):
    for j in range(101):
        Lab[j,i]=ona_plana(x[i],y[j],t,c,x0)\
                 +ona_esf(x[i],y[j],x0,y1,t,c)\
                     +ona_esf(x[i],y[j],x0,y2,t,c)
        Int[j,i]=intensitat(Lab[j,i])
        


plt.pcolormesh(x,y,Lab,vmin=Lab.min(),vmax=Lab.max())

print(Lab.min(),Lab.max())



    
        



    


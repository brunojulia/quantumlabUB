

import numpy as np

L = 1
dx = 0.001
nx = int(L/dx)

T = 10
dt = 0.1
nt = int(T/dt) 

phi = np.ones([nx,nx])
m = 1
trajectory = np.ones([nt+1,4])


def RK4(t,dt,r):
    k1 = f(t,r)
    k2 = f(t+dt/2, r+k1*dt/2)
    k3 = f(t+dt/2, r+k2*dt/2)
    k4 = f(t+dt, r+k3*dt)
    
    rlater = r + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    return rlater

def f(t,r):
    partialx = (phi[int(r[1]/nx),int(r[0]/nx)+1]-phi[int(r[1]/nx),int(r[0]/nx)])/dx
    partialy = (phi[int(r[0]/nx)+1,int(r[0]/nx)]-phi[int(r[0]/nx),int(r[0]/nx)])/dx
    
    f = np.zeros([4])
    f[0] = r[2]
    f[1] = r[3]
    f[2] = partialx/m
    f[3] = partialy/m
    return f

def RK4s(y):
    for i in range(0,nt):
        y[i+1,:] = RK4(i*dt,dt,y[i,:])
        
    return y

trajectory[0,:] = np.array([0,0,10,10])
trajectory = RK4s(trajectory) 
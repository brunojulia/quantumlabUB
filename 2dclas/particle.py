

import numpy as np

L = 1
dx = 0.001
nx = int(L/dx)
phi = np.ones([nx,nx])

T = 10
dt = 0.1
nt = int(T/dt) 

class Particle():
    
    def __init__(self,mass,trajectory,charge,h):
        self.mass = mass
        self.charge = charge
        self.trajectory = trajectory
        self.h = h
    
    def RightHand(self,r):
        partialx = (phi[int(r[1]/nx),int(r[0]/nx)+1]-phi[int(r[1]/nx),int(r[0]/nx)])/dx
        partialy = (phi[int(r[0]/nx)+1,int(r[0]/nx)]-phi[int(r[0]/nx),int(r[0]/nx)])/dx
    
        f = np.zeros([4])
        f[0] = r[2]
        f[1] = r[3]
        f[2] = partialx/self.mass
        f[3] = partialy/self.mass
        return f
    
    def RK4(self,t,dt,r):
        k1 = self.RightHand(t,r)
        k2 = self.RightHand(t+dt/2, r+k1*dt/2)
        k3 = self.RightHand(t+dt/2, r+k2*dt/2)
        k4 = self.RightHand(t+dt, r+k3*dt)
    
        rlater = r + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return rlater
    
    def ComputeTrajectory(self,r0):
        for i in range(0,nt):
            self.trajectory[i+1,:] = self.RK4(i*dt,dt,self.trajectory[i,:])
        
        return self.trajectory
    

p = Particle(1,1,np.ones([nt+1,4]),0.1)
p.ComputeTrajectory(np.array([0,0,1,1]))    

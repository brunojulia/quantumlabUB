

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as pltanim
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from potentials import *

class Phi():
    
    def __init__(self):
        self.functions = np.array([])
        self.dfunctionsx = np.array([])
        self.dfunctionsy = np.array([])
    def add_function(self,fun,dfunx,dfuny,param):
        self.functions = np.append(self.functions,(fun,param))
        self.dfunctionsx = np.append(self.dfunctionsx,(dfunx,param))
        self.dfunctionsy = np.append(self.dfunctionsy,(dfuny,param))
        return
    def val(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.functions.shape[0],2):
            value = value + self.functions[i](r,self.functions[i+1])
        return value
    def dvalx(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.dfunctionsx.shape[0],2):
            value = value + self.dfunctionsx[i](r,self.dfunctionsx[i+1])
        return value
    def dvaly(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.dfunctionsy.shape[0],2):
            value = value + self.dfunctionsy[i](r,self.dfunctionsy[i+1])
        return value

    def clear(self):
        self.functions = np.array([])
        self.dfunctionsx = np.array([])
        self.dfunctionsy = np.array([])
        


class Particle():
    
    
    def __init__(self,mass,charge,dt):
        self.mass = mass
        self.charge = charge
        self.trajectory = np.zeros([1,4])
        self.dt = dt
        self.steps = np.array([0])
        self.h = 1
        self.trax = 0
        self.tray = 0
        self.travx = 0
        self.travy = 0
        self.pot = 0
    
    def RightHand(self,r):

        f = np.zeros([4])
        f[0] = r[2]
        f[1] = r[3]
        f[2] = -self.pot.dvalx(r[0],r[1])/self.mass
        f[3] = -self.pot.dvaly(r[0],r[1])/self.mass
        return f
    
    
    def RK4(self,t,dt,r):
        k1 = self.RightHand(r)
        k2 = self.RightHand(r+k1*self.dt/2)
        k3 = self.RightHand(r+k2*self.dt/2)
        k4 = self.RightHand(r+k3*self.dt)
    
        rlater = r + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return rlater
    
    def RKF(self,r):
        eps = 10**-2

        hnew = self.dt
        safety = 0
        while(hnew<self.h):
            safety += 1
            if(safety>100):
                break
            
            self.h = hnew
            k0 = self.RightHand(r)
            k1 = self.RightHand(r + self.h/4.*k0)
            k2 = self.RightHand(r + (3.*self.h/32.)*k0 + (9.*self.h/32.)*k1)
            k3 = self.RightHand(r + (1932.*self.h/2197.)*k0 - (7200.*self.h/2197.)*k1 + (7296.*self.h/2197.)*k2)
            k4 = self.RightHand(r + (439.*self.h/216.)*k0 - 8.*self.h*k1 + (3680.*self.h/513.)*k2 - (845.*self.h/4104.)*k3)
            k5 = self.RightHand(r - (8.*self.h/27.)*k0 + 2.*self.h*k1 - (3544.*self.h/2565.)*k2 + (1859.*self.h/4104.)*k3 - (11.*self.h/40.)*k4)
#            rnexthat = r + (16.*self.h/135.)*k0 + (6656.*self.h/12825.)*k2 + (28561.*self.h/56430.)*k3 - (9.*self.h/50.)*k4 + (2.*self.h/55.)*k5
            delta = self.h*((1./360.)*k0 - (128./4275.)*k2 - (2197./75240.)*k3 + (1./50.)*k4 + (2./55.)*k5)
            try:
                hnew = 0.9*self.h*(np.abs(self.h)*eps/np.sqrt(np.sum(delta**2)))**(1./4.)
            except RuntimeWarning:
                hnew = self.dt
            except FloatingPointError:
                hnew = self.dt
            
            if(hnew>self.dt):
                hnew = self.dt
        self.h = hnew
        hfinal = hnew
        rlater = r + (16.*self.h/135.)*k0 + (6656.*self.h/12825.)*k2 + (28561.*self.h/56430.)*k3 - (9.*self.h/50.)*k4 + (2.*self.h/55.)*k5
        return rlater,hfinal
    
    
    
    
    def ComputeTrajectory(self,r0):
        self.trajectory[0,:] = r0
        self.steps = np.array([0])
        for i in range(0,nt):
            try:
                tranext = self.RK4(i*self.dt,self.dt,self.trajectory[i,:])
                if(np.abs(tranext[0]) >= L/2 or np.abs(tranext[1]) >= L/2):
                    break
                else:
                    self.trajectory = np.append(self.trajectory,tranext.reshape(1,4),axis=0)
            except IndexError:
                break
            
            
    def ComputeTrajectoryF(self,r0,T,pot):
        self.pot = pot
        self.trajectory = np.zeros([1,4])
        self.trajectory[0,:] = r0
        
        self.steps = np.array([0])
        self.trax = 0
        self.tray = 0
        self.travx = 0
        self.travy = 0
        
        L = 200
        dt = 0.1
        i = 0
        while(self.steps.sum() < T):
            self.h = 1
            try:
                tranext , newstep = self.RKF(self.trajectory[i,:])
                self.trajectory = np.append(self.trajectory,tranext.reshape(1,4),axis=0)
                self.steps = np.append(self.steps,newstep)
                i += 1
            except IndexError:
                break
        
        self.trax = interp1d(self.steps.cumsum(),self.trajectory[:,0],kind='quadratic')
        self.tray = interp1d(self.steps.cumsum(),self.trajectory[:,1],kind='quadratic')
        self.travx = interp1d(self.steps.cumsum(),self.trajectory[:,2],kind='quadratic')
        self.travy = interp1d(self.steps.cumsum(),self.trajectory[:,3],kind='quadratic')
        
    def KEnergy(self):
        KEnergy = np.zeros([self.trajectory.shape[0]])
        for i in range(0,self.trajectory.shape[0]):
            KEnergy[i] = self.mass/2. * (self.trajectory[i,2]**2+self.trajectory[i,3]**2)
        return KEnergy
    
    def PEnergy(self):
        PEnergy = np.zeros(self.trajectory.shape[0])
        for k in range(0,self.trajectory.shape[0]):
            PEnergy[k] = self.pot.val(self.trajectory[k,0],self.trajectory[k,1])
        return PEnergy 
            
    def Energy(self):
        E = np.zeros([self.trajectory.shape[0]])
        for i in range(0,self.trajectory.shape[0]):
            E[i] = self.KEnergy()[i] + self.PEnergy()[i]
        return E

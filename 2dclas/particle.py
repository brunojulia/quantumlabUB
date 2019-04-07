

import numpy as np
import matplotlib.pyplot as plt






L = 200


T = 10
dt = 0.01
nt = int(T/dt) 


class Phi():
    
    def __init__(self):
        self.functions = np.array([])
        self.dfunctionsx = np.array([])
        self.dfunctionsy = np.array([])
    def add_function(self,fun,dfunx,dfuny,param):
        self.functions = np.append(self.functions,(fun,param))
#        self.functions = np.append(self.functions,param)
        self.dfunctionsx = np.append(self.dfunctionsx,(dfunx,param))
#        self.derivatives = np.append(self.derivatives,param)
        self.dfunctionsy = np.append(self.dfunctionsy,(dfuny,param))
        return
    def val(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.functions.shape[0],2):
            value = value + self.functions[0](r,self.functions[i+1])
        return value
    def dvalx(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.dfunctionsx.shape[0],2):
            value = value + self.dfunctionsx[0](r,self.dfunctionsx[i+1])
        return value
    def dvaly(self,x,y):
        value = 0
        r = (x,y)
        for i in range(0,self.dfunctionsy.shape[0],2):
            value = value + self.dfunctionsy[0](r,self.dfunctionsy[i+1])
        return value

    def clear(self):
        self.functions = np.array([])
        self.dfunctionsx = np.array([])
        self.dfunctionsy = np.array([])
        


class Particle():
    
    def __init__(self,mass,charge,tra,dt):
        self.mass = mass
        self.charge = charge
        self.trajectory = tra
        self.dt = dt
        self.steps = np.array([0])
        self.h = 1
    
    def RightHand(self,r):

        f = np.zeros([4])
        f[0] = r[2]
        f[1] = r[3]
        f[2] = -pot.dvalx(r[0],r[1])/self.mass
        f[3] = -pot.dvaly(r[0],r[1])/self.mass
        return f
    
    def RK4(self,t,dt,r):
        k1 = self.RightHand(r)
        k2 = self.RightHand(r+k1*self.dt/2)
        k3 = self.RightHand(r+k2*self.dt/2)
        k4 = self.RightHand(r+k3*self.dt)
    
        rlater = r + self.dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        return rlater
    
    def RKF(self,r):
        eps = 0.000001
        hnew = 0.1
        safety = 0
        while(hnew<self.h):
            safety += 1
            if(safety>10):
                break
            
            self.h = hnew
            k0 = self.RightHand(r)
            k1 = self.RightHand(r + self.h/4.*k0)
            k2 = self.RightHand(r + (3.*self.h/32.)*k0 + (9.*self.h/32.)*k1)
            k3 = self.RightHand(r + (1932.*self.h/2197.)*k0 - (7200.*self.h/2197.)*k1 + (7296.*self.h/2197.)*k2)
            k4 = self.RightHand(r + (439.*self.h/216.)*k0 - 8.*self.h*k1 + (3680.*self.h/513.)*k2 - (845.*self.h/4104.)*k3)
            k5 = self.RightHand(r - (8.*self.h/27.)*k0 + 2.*self.h*k1 - (3544.*self.h/2565.)*k2 + (1859.*self.h/4104.)*k3 - (11.*self.h/40.)*k4)
#            rnext = r + h*(25*k0/256 + 1408*k2/2565 + 2197*k3*4104 - k4/5)
            rnexthat = r + (16.*self.h/135.)*k0 + (6656.*self.h/12825.)*k2 + (28561.*self.h/56430.)*k3 - (9.*self.h/50.)*k4 + (2.*self.h/55.)*k5
            delta = self.h*((1./360.)*k0 - (128./4275.)*k2 - (2197./75240.)*k3 + (1./50.)*k4 + (2./55.)*k5)
#            '''
            try:
                hnew = 0.9*self.h*(np.abs(self.h)*eps/np.sqrt(np.sum(delta**2)))**(1./4.)
            except RuntimeWarning:
                hnew = 0.1
            
            if(hnew>0.1):
                hnew = 0.1
        self.h = hnew
        hfinal = hnew
        rlater = rnexthat
        return rlater,hfinal
    
    
    
    
    def ComputeTrajectory(self,r0):
        self.trajectory[0,:] = r0
        self.steps = np.array([0])
        for i in range(0,nt):
            try:
                tranext = self.RK4(i*self.dt,self.dt,self.trajectory[i,:])
                if(tranext[0] >= L/2 or tranext[1] >= L/2):
                    break
                else:
                    self.trajectory = np.append(self.trajectory,tranext.reshape(1,4),axis=0)
            except IndexError:
                break
            
    def ComputeTrajectoryF(self,r0):
        self.trajectory[0,:] = r0
        i = 0
        while(self.steps.sum() < T):
            self.h = 1
            try:
                tranext , newstep = self.RKF(self.trajectory[i,:])
                if(tranext[0] >= L/2 or tranext[1] >= L/2):
                    break
                else:
                    self.trajectory = np.append(self.trajectory,tranext.reshape(1,4),axis=0)
                    self.steps = np.append(self.steps,newstep)
                    i += 1
            except IndexError:
                break
                
    def KEnergy(self):
        KEnergy = np.zeros([self.trajectory.shape[0]])
        for i in range(0,self.trajectory.shape[0]):
            KEnergy[i] = self.mass/2. * (self.trajectory[i,2]**2+self.trajectory[i,3]**2)
        return KEnergy
    
    def PEnergy(self):
        PEnergy = np.zeros(self.trajectory.shape[0])
        for k in range(0,self.trajectory.shape[0]):
            PEnergy[k] = pot.val(self.trajectory[k,0],self.trajectory[k,1])
        return PEnergy 
            
    def Energy(self):
        E = np.zeros([self.trajectory.shape[0]])
        for i in range(0,self.trajectory.shape[0]):
            E[i] = self.KEnergy()[i] + self.PEnergy()[i]
        return E
    
pot = Phi()

def linear(r,param):
    f = param[0]*r[0] + param[1]*r[1]
    return f

def dlinearx(r,param):
    f = param[0]
    return f

def dlineary(r,param):
    f = param[1]
    return f

#############################
    
def osc(r,param):
    x0 = param[0]
    y0 = param[1]
    k = param[2]
    rlim = param[3]
    rval = np.sqrt((r[0]-x0)**2+(r[1]-y0)**2)
    
    if(rval < rlim):
        f = k*rval**2
    else:
        vlim = k*rlim**2
#        f = 0.
        f = -2*k*((rlim-x0)*np.exp(-np.abs(r[0]-x0)) + (rlim-y0)*np.exp(-np.abs(r[1]-y0)))
    return f

def doscx(r,param):
    x0 = param[0]
    y0 = param[1]
    k = param[2]
    rlim = param[3]
    
#    if(np.sqrt((r[0]-x0)**2+(r[1]-y0)**2) < rlim):
    if(np.abs(r[0]-x0) < rlim):
        f = k*(2*(r[0]-x0))
    else:
#        f = 0.
        f = k*2*(rlim-x0)*np.exp(-(np.abs(r[0]-x0)))
    return f

def doscy(r,param):
    x0 = param[0]
    y0 = param[1]
    k = param[2]
    rlim = param[3]
    
#    if(np.sqrt((r[0]-x0)**2+(r[1]-y0)**2) < rlim):
    if(np.abs(r[1]-y0) < rlim):
        f = k*(2*(r[1]-y0))
    else:
#        f = 0.
        f = k*2*(rlim-y0)*np.exp(-(np.abs(r[1]-y0)))
    return f

         

#Oscilador Harmonico
pot.add_function(osc,doscx,doscy,[0,0,1,10])
#pot.add_function(osc,doscx,doscy,[25,0,1,10])

dx = 1
nx = int(L/dx)
xx,yy, = np.meshgrid(np.linspace(-L/2,L/2,nx,endpoint=True),np.linspace(-L/2,L/2,nx,endpoint=True))
im = np.zeros((nx,nx))
for i in range(0,nx):
    for j in range(0,nx):
        im[i,j] = pot.val(xx[i,j],yy[i,j])

p = Particle(1,1,np.ones([1,4]),dt)
p.ComputeTrajectoryF(np.array([0,0,10,0]))    
a = p.trajectory
b = p.steps
t = p.steps.cumsum()
#t = dt*np.ones(a.shape[0]).cumsum()

fun = np.zeros(nx)
xs = np.zeros(nx)
for i in range(0,nx):
    x = i*dx - L/2
    y=0
    xs[i] = x 
    fun[i] = pot.dvalx(x,y)
plt.figure()
plt.plot(xs,fun)

#'''
plt.figure()
plt.subplot(2,2,1)
plt.contourf(xx,yy,im,cmap="plasma")
plt.colorbar()
plt.axis("square")
plt.xlabel('x([L])')
plt.ylabel('y([L])')
plt.subplot(2,2,2)
plt.plot(t,p.KEnergy(),"r-",t,p.PEnergy(),"b-",t,p.Energy(),"g-")
plt.legend(('EC','EP','EM'),loc='best')
plt.xlabel('t([T])')
plt.ylabel('Energy([M]*[L]^2*[T]^-2)')
plt.subplot(2,2,3)
plt.plot(t,a[:,0])
plt.xlabel('t([T])')
plt.ylabel('x([L])')
plt.subplot(2,2,4)
plt.plot(a[:,0],a[:,1])
plt.xlabel('x([L])')
plt.ylabel('y([L])')

plt.tight_layout()
plt.show()
#'''




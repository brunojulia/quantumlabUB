

import numpy as np
import matplotlib.pyplot as plt






L = 100


T = 10
dt = 0.01
nt = int(T/dt) 

#Funcion para calcular los tiempos para unos pasos de tiempo arbitrarios
def times(h):
    t = np.zeros([h.shape[0]])
    for i in range(0,h.shape[0]):
        t[i] = i*h[i]
    return t

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
    
    def ComputeTrajectory(self,r0):
        self.trajectory[0,:] = r0
        for i in range(0,nt):
            try:
                tranext = self.RK4(i*self.dt,self.dt,self.trajectory[i,:])
                if(tranext[0] >= L/2 or tranext[1] >= L/2):
                    break
                else:
                    self.trajectory = np.append(self.trajectory,tranext.reshape(1,4),axis=0)
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
    f = param[0]*(r[0]**2+r[1]**2)
    return f

def doscx(r,param):
    f = param[0]*(2*r[0])
    return f

def doscy(r,param):
    f = param[0]*(2*r[1])
    return f

         

#Oscilador Harmonico
pot.add_function(osc,doscx,doscy,[1])

dx = 1
nx = int(L/dx)
im = np.zeros((nx,nx))
for i in range(0,nx):
    for j in range(0,nx):
        x = i*dx - L/2
        y = j*dx - L/2
        im[i,j] = pot.val(x,y)
im = im.transpose()


p = Particle(1,1,np.ones([1,4]),dt)
p.ComputeTrajectory(np.array([25,0,0,0]))    
a = p.trajectory
t = times(dt*np.ones([a.shape[0]]))



plt.figure()
plt.subplot(2,2,1)
plt.imshow(im,cmap="plasma")
plt.axis("off")
plt.subplot(2,2,2)
plt.plot(t,p.KEnergy(),"r-",t,p.PEnergy(),"b-",t,p.Energy(),"g-")
plt.legend(('EC','EP','EM'),loc='best')
plt.subplot(2,2,3)
plt.plot(t,a[:,0])
plt.xlabel('t')
plt.ylabel('x')
plt.subplot(2,2,4)
plt.plot(a[:,0],a[:,1])
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()


#Lineal
pot.clear()
pot.add_function(linear,dlinearx,dlineary,[1,0])

dx = 1
nx = int(L/dx)
im = np.zeros((nx,nx))
for i in range(0,nx):
    for j in range(0,nx):
        x = i*dx - L/2
        y = j*dx - L/2
        im[i,j] = pot.val(x,y)
im = im.transpose()
        
        
p = Particle(1,1,np.ones([1,4]),dt)
p.ComputeTrajectory(np.array([0,0,0,0]))    
a = p.trajectory
t = times(dt*np.ones([a.shape[0]]))



plt.figure()
plt.subplot(2,2,1)
plt.imshow(im,cmap="plasma")
plt.axis("off")
plt.subplot(2,2,2)
plt.plot(t,p.KEnergy(),"r-",t,p.PEnergy(),"b-",t,p.Energy(),"g-")
plt.legend(('EC','EP','EM'),loc='best')
plt.subplot(2,2,3)
plt.plot(t,a[:,0])
plt.xlabel('t')
plt.ylabel('x')
plt.subplot(2,2,4)
plt.plot(a[:,0],a[:,1])
plt.xlabel('x')
plt.ylabel('y')

plt.tight_layout()
plt.show()
#'''



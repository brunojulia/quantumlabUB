

import numpy as np
import matplotlib.pyplot as plt

L = 100
dx = 0.01
nx = int(L/dx)

def osc(j,i):
    x = i*dx - L/2
    y = j*dx - L/2
    f = (x**2+y**2)*1
    return f

def linear(j,i):
    x = i*dx - L/2
    y = j*dx - L/2
    f = x
    return f

#phi = np.fromfunction(lambda y, x: 0.1*(x-nx/2),(nx,nx))
#phi = np.ones([nx,nx])

#Funciones para calcular a que punto de la malla corresponde un x,y
def xp(r):
    if(r>=0):
        x = round(r/dx) + round(nx/2)
    if(r<0):    
        x = abs(abs(round(r/dx)) - round(nx/2))
    return int(x)

def yp(r):
    if(r>=0):
        y = round(r/dx) + round(nx/2)
    if(r<0):    
        y = abs(abs(round(r/dx)) - round(nx/2))
    return int(y)

T = 10
dt = 0.01
nt = int(T/dt) 

#Funcion para calcular los tiempos para unos pasos de tiempo arbitrarios
def times(h):
    t = np.zeros([h.shape[0]])
    for i in range(0,h.shape[0]):
        t[i] = i*h[i]
    return t


class Particle():
    
    def __init__(self,mass,charge,tra,dt):
        self.mass = mass
        self.charge = charge
        self.trajectory = tra
        self.dt = dt
    
    def RightHand(self,r):
        partialx = (phi[yp(r[1]),xp(r[0])+1]-phi[yp(r[1]),xp(r[0])])/dx
        partialy = (phi[yp(r[1])+1,xp(r[0])]-phi[yp(r[1]),xp(r[0])])/dx
    
        f = np.zeros([4])
        f[0] = r[2]
        f[1] = r[3]
        f[2] = -partialx/self.mass
        f[3] = -partialy/self.mass
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
            '''
            if(abs(tranext[0]) >= (L/4) or abs(tranext[1]) >= (L/4)):
#                print('fin')
                break
            else:
#                print('estoy aqui')
                self.trajectory = np.append(self.trajectory,tranext.reshape(1,4),axis=0)
                '''
                
                
    def KEnergy(self):
        KEnergy = np.zeros([self.trajectory.shape[0]])
        for i in range(0,self.trajectory.shape[0]):
            KEnergy[i] = self.mass/2. * (self.trajectory[i,2]**2+self.trajectory[i,3]**2)
        return KEnergy
    
    def PEnergy(self):
        PEnergy = np.zeros(self.trajectory.shape[0])
        for k in range(0,self.trajectory.shape[0]):
            PEnergy[k] = phi[yp(self.trajectory[k,1]),xp(self.trajectory[k,0])]
#            i = yp(self.trajectory[k,1])
#            j = xp(self.trajectory[k,0])
#            PEnergy[k] = phi[i-1:i+2,j-1:j+2].sum()/9
        return PEnergy 
            
    def Energy(self):
        E = np.zeros([self.trajectory.shape[0]])
        for i in range(0,self.trajectory.shape[0]):
            E[i] = self.KEnergy()[i] + self.PEnergy()[i]
        return E
            
#Oscilador Harmonico 'preciso'       
phi = np.fromfunction(osc,(nx,nx))   

p = Particle(1,1,np.ones([1,4]),dt)
p.ComputeTrajectory(np.array([25,0,0,0]))    
a = p.trajectory
t = times(dt*np.ones([a.shape[0]]))


plt.figure()
plt.subplot(2,2,1)
plt.imshow(phi,cmap="plasma")
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


#Oscilador Harmonico
dx = 0.1
nx = int(L/dx)    
phi = np.fromfunction(osc,(nx,nx))   

p = Particle(1,1,np.ones([1,4]),dt)
p.ComputeTrajectory(np.array([25,0,0,0]))    
a = p.trajectory
t = times(dt*np.ones([a.shape[0]]))


plt.figure()
plt.subplot(2,2,1)
plt.imshow(phi,cmap="plasma")
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


#Lineal
dx = 0.01
nx = int(L/dx)    
phi = np.fromfunction(linear,(nx,nx))   

p = Particle(1,1,np.ones([1,4]),dt)
p.ComputeTrajectory(np.array([0,0,1,1]))    
a = p.trajectory
t = times(dt*np.ones([a.shape[0]]))


plt.figure()
plt.subplot(2,2,1)
plt.imshow(phi,cmap="plasma")
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




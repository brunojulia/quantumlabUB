
import numpy as np

def LJ(r):
    V = 1
    sig = 5
    
    x = np.where(r < 0.01, 0, 1)*(sig/r)
    
    value = 4*V*((x)**12-(x)**6)
    return value

def dLJx(x,y,param):
    V = param[0]
    sig = param[1]
    
    d = x**2+y**2
    mask = np.where(d < 0.000001,1,0)
    d = d + mask
    x = x + mask
    
    value = 24*(sig**6)*V*x*((d**3 - 2*(sig**6))/(d**7))
    value = value*(1-mask)
    return value

def dLJy(x,y,param):
    V = param[0]
    sig = param[1]
    
    d = x**2+y**2
    mask = np.where(d < 0.000001,1,0)
    d = d + mask
    y = y + mask
    
    value = 24*(sig**6)*V*y*((d**3 - 2*(sig**6))/(d**7))
    value = value*(1-mask)
#    print(value)
    return value



class particle:
    
    def __init__(self,m,q,r0,v0,D):
        self.m = m
        self.q = q
        self.r0 = r0
        self.v0 = v0
        self.r = r0
        self.v = v0


class PhySystem:
    
    def __init__(self,particles,param):
        self.particles = particles
        self.param = param
#        self.m = np.vectorize(lambda i: i.m)(particles)
#        self.q = np.vectorize(lambda i: i.q)(particles)
    def RK4(self,t,dt):
        
        k1 = self.f(t,np.zeros([self.particles.size,4]))
        k2 = self.f(t+dt/2,k1*dt/2)
        k3 = self.f(t+dt/2,k2*dt/2)
        k4 = self.f(t+dt,k3*dt)
    
        for j in range(0,self.particles.size):
            self.particles[j].r = self.particles[j].r + dt/6 * (k1[j,:2] + 2*k2[j,:2] + 2*k3[j,:2] + k4[j,:2])
            self.particles[j].v = self.particles[j].v + dt/6 * (k1[j,2:] + 2*k2[j,2:] + 2*k3[j,2:] + k4[j,2:])
        return
        
    def f(self,t,delta):
        N = self.particles.size
        X = np.vectorize(lambda i: i.r[0])(self.particles) + delta[:,0]
        Y = np.vectorize(lambda i: i.r[1])(self.particles) + delta[:,1]
        MX, MXT = np.meshgrid(X,X)
        MY, MYT = np.meshgrid(Y,Y)
        
        dx = MXT - MX
        dy = MYT - MY
#        print(dx,dy)
        dUx = 0.
        dUy = 0.
        
        f = np.zeros([N,4])
        for j in range(0,N):
            dUx = np.sum(dLJx(dx[j,:],dy[j,:],self.param))
            dUy = np.sum(dLJy(dx[j,:],dy[j,:],self.param))
#            print(dUx,dUy)
            f[j,:] = np.array([self.particles[j].v[0] + delta[j,2],self.particles[j].v[1] + delta[j,3],-(1/self.particles[j].m)*dUx,-(1/self.particles[j].m)*dUy])
        return f
    
    def solve(self,T,dt):
        t = 0.
        N = int(T/dt)
        
        X = np.vectorize(lambda i: i.r[0])(self.particles)
        Y = np.vectorize(lambda i: i.r[1])(self.particles)
        
        for i in range(0,N):
            self.RK4(t,dt)
            t = t + dt
            X = np.vstack((X,np.vectorize(lambda i: i.r[0])(self.particles)))
            Y = np.vstack((Y,np.vectorize(lambda i: i.r[1])(self.particles)))
        return X,Y

#a = particle(1,1,np.array([-3,0]),np.array([0,0]),2)
#b = particle(1,1,np.array([3,0]),np.array([0,0]),2)
#c = particle(1,1,np.array([0,1]),np.array([0,1]),2)
#d = particle(1,1,np.array([3,0]),np.array([1,1]),2)
#e = particle(1,1,np.array([7,4]),np.array([-1,-1]),2)
#
#s = PhySystem(np.array([a,b,c,d,e]))
#s = PhySystem(np.array([a,b]))
#print(s.particles[0].r)
#print(s.particles[1].r)
#X,Y = s.solve(10,0.1)


#a = np.array([a,b,c,d,e])
#msf = lambda i: i.m
#msfv = np.vectorize(msf)
#
#ms = np.vectorize(lambda i: i.m)(a) 
#R = np.vectorize(lambda i: i.r[0])(a)
#MR, MRT = np.meshgrid(R,R)
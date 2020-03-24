import numpy as np


#Lennard jones potentials, see documentation for the expressions.
def dLJverlet(x,r2,param):
    """The derivative has the same form for x and y so only one is needed,
    this only changes when calling the interaction on the algotyhm,
    for all isotrope interactions this should still hold."""
    V = param[0]
    sig = param[1]
    L = param[2]
    rc = 3.


    value = ((48.*x)/(r2))*(1./(r2**6) - 0.5/(r2**3))

    return value

def LJverlet(r2,param):
    V = param[0]
    sig = param[1]
    L = param[2]
    rc = 3.


    #The extra term comes from the truncation of the potential energy
    #See section 3.2.2 of reference [2] of the doc
    value = 4*(1./(r2**6) - 1./(r2**3)) - 4*(1./(rc**12) - 1./(rc**6))

    return value

def walls(r,param):
    """For saving on lines I have designed the walls function and the derivative in such a way
    that the same line can be used for the right-left and the top-down walls.
    This works thanks to the np.sign.
    The height/width of the wall is scaled to the size of the box so if L is modified
    you don't need to modify this. The parameter a is also escaled to the unit of lenght
    (radius of the particles)"""
    V = param[0]
    sig = param[1]
    L = param[2]

    a = 1/sig


    x0 = L/2.
    y0 = 0.
    V0 = 10000*V
    Rx = 0.01*L
    Ry = 0.6*L

    x = r[0] - x0*np.sign(r[0])
    y = r[1] - y0*np.sign(r[1])
    px = np.sqrt(x**2)
    py = np.sqrt(y**2)

    f1 = V0*(1/(1 + np.exp((px-Rx)/a)))*(1/(1 + np.exp((py-Ry)/a)))

    x0 = 0.
    y0 = L/2.
    V0 = 10000*V
    Rx = 0.6*L
    Ry = 0.01*L

    x = r[0] - x0*np.sign(r[0])
    y = r[1] - y0*np.sign(r[1])
    px = np.sqrt(x**2)
    py = np.sqrt(y**2)

    f2 = V0*(1/(1 + np.exp((px-Rx)/a)))*(1/(1 + np.exp((py-Ry)/a)))

    value = f1+f2
    return value

def dwalls(r,param):
    """See walls function for more information, this is just the derivative."""
    V = param[0]
    sig = param[1]
    L = param[2]

    a = 1/sig

    x0 = L/2.
    y0 = 0.
    V0 = 10000*V
    Rx = 0.01*L
    Ry = 0.6*L

    x = r[0] - x0*np.sign(r[0])
    y = r[1] - y0*np.sign(r[1])


    px = np.sqrt(x**2)
    py = np.sqrt(y**2)
    try:
        f1 = -V0*((np.sign(x)*np.exp((px-Rx)/a))/(a*(np.exp((px-Rx)/a)+1)**2))*(1/(1 + np.exp((py-Ry)/a)))

        x0 = 0.
        y0 = L/2.
        V0 = 10000*V
        Rx = 0.6*L
        Ry = 0.01*L

        x = r[0] - x0*np.sign(r[0])
        y = r[1] - y0*np.sign(r[1])
        px = np.sqrt(x**2)
        py = np.sqrt(y**2)

        f2 = -V0*((np.sign(x)*np.exp((Rx+px)/a))/(a*(np.exp(Rx/a)+np.exp(px/a))**2))*(1/(1 + np.exp((py-Ry)/a)))
    except RuntimeWarning:
        f1 = 0.
        f2 = 0.
    except FloatingPointError:
        f1 = 0.
        f2 = 0.
    f = f1 + f2
    return f



class particle:
    """particle(m,q,r0,v0,D) class stores the intrinsic properties of a particle (mass, charge)
    and its initial and current position (r0,v0) as well as the dimension of the space (D).
    The dimension of the space is not used but it could be useful for some applications.
    r0 and v0 can be numpy.arrays or lists"""

    def __init__(self,m,q,r0,v0,D):
        self.m = m
        self.q = q
        self.r0 = r0
        self.v0 = v0
        self.r = r0
        self.v = v0
    def reset(self):
        self.r = self.r0
        self.v = self.v0

class PhySystem:
    """PhySystem(particles,param) class stores all the particles and contains functions for
    computing the trajectory of the system. Particles is a list or numpy.array full of
    particle objects. param is a numpy array or list with any parameters of the system.
    PhySystem has the verlet algorythm incorporated and param are the units of the system
    (potential depth for the LJ for energy,particle radius for lenght and size of the box, see
    intsim.py and documentation for more details on reduced units)."""
    def __init__(self,particles,param):
        self.particles = particles
        self.param = param
        #Usage of the vectorize function is very useful throughout this class,
        #in this case is not necessary since all particles have the same mass
        #but it is useful for other aplications (Gravitational problems, for example)
        self.m = np.vectorize(lambda i: i.m)(particles)
#        self.q = np.vectorize(lambda i: i.q)(particles)
        self.U = np.array([])

    def verlet(self,t,dt,r0,r1):
        """verlet(t,dt,r0,r1) performs one step of the verlet algorythm at time t
        with a step of dt with the previous position r0 and the current position r1, returns
        the next position r2.
        All of the r have shape (2,N) where N is the number of particles. The first
        index acceses either the x or y coordinate and the second the particle. The function
        returns the coordinates by separate."""
        r2 = np.zeros([2,self.particles.size])
        r2 = (2*r1 - r0 + np.transpose(self.fv(r1[0,:],r1[1,:])) * (dt**2))
        #The transpose is necessary because I messed up the shapes when I did the fv function.


        return r2[0,:],r2[1,:]

    def fv(self,X,Y):
        """fv(X,Y) represents the forces that act on all the particles at a particular time.
        It computes the matrix of forces using the positions given with X and Y which are
        the arrays of size N containing all the positions (coordinates X and Y).
        The resulting matrix, f is of shape (N,2) (it should be (2,N), see the verlet function)."""

        L = self.param[2]

        rc = 3.

        N = self.particles.size
        #For computing all the distances I use a trick with the meshgrid function,
        #see the documentation on how this works if you dont see it.
        MX, MXT = np.meshgrid(X,X)
        MY, MYT = np.meshgrid(Y,Y)

        dx = MXT - MX
        dx = dx

        dy = MYT - MY
        dy = dy

        r2 = np.square(dx)+np.square(dy)

        dUx = 0.
        dUy = 0.
        #p = 1 #Pel test de col·lisions
        utot = np.array([])
        f = np.zeros([N,2])
        for j in range(0,N):

            #Actualment, el cutoff de 3 unitats de llargada no fa massa en quan al rendiment del programa, s'ha de mirar bé si és el millor mètode

            #In the force computation we include the LJ and the walls. I truncate the interaction at 3 units of lenght,
            #I also avoid distances close to 0 (which only should affect the diagonal in the matrix of distances)
            #All these conditions are included using the numpy.where function.
            #If you want to include more forces you only need to add terms to these lines.
            dUx = np.sum(np.where(np.logical_and(r2[j,:] < (rc**2), r2[j,:] > 10**(-2)), dLJverlet(dx[j,:],r2[j,:],self.param),0.)) - dwalls([X[j],Y[j]],self.param)
            dUy = np.sum(np.where(np.logical_and(r2[j,:] < (rc**2), r2[j,:] > 10**(-2)), dLJverlet(dy[j,:],r2[j,:],self.param),0.)) - dwalls([Y[j],X[j]],self.param)
            u =  np.sum(np.where(np.logical_and(r2[j,:] < (rc**2), r2[j,:] > 10**(-2)), LJverlet(r2[j,:],self.param),0.)) + np.where((X[j]**2+Y[j]**2) > (0.8*L)**2,walls([X[j],Y[j]],self.param),0.)
#            if(np.array_equal(np.where(np.logical_and(r2[j,:] < (rc**2), r2[j,:] > 10**(-2)),"hey",None),[None,None,None,None,None,None,None,None])):
#                pass
#            else:
#                print("Toc",p)
#                p += 1
            f[j,:] = np.array([dUx,dUy])
            utot = np.append(utot,u)
        self.U = np.append(self.U,np.sum(utot))
        return f

    def solveverlet(self,T,dt):
        """solververlet(T,dt) solves the equation of movement from t=0 to t=T
        at a step of dt. It also computes the potential and kinetic energy as well
        as the temperature of the system both at each instant and acumulated
        every delta (see below)."""
        t = 0.
        self.n = int(T/dt)

        progress = t/T*100

        np.vectorize(lambda i: i.reset())(self.particles)#This line resets the particles to their initial position

        #X,Y,VX,VY has the trajectories of the particles with two indexes that
        #access time and particles, respectively
        self.X = np.vectorize(lambda i: i.r[0])(self.particles)
        self.Y = np.vectorize(lambda i: i.r[1])(self.particles)
        self.VX = np.vectorize(lambda i: i.v[0])(self.particles)
        self.VY = np.vectorize(lambda i: i.v[1])(self.particles)


        #Generation of the precious position (backwards euler step)
        X1 = self.X
        Y1 = self.Y
        X0 = X1 - self.VX*dt
        Y0 = Y1 - self.VY*dt

        for i in range(0,self.n):
            #Call verlet to compute the next position
            X2,Y2 = self.verlet(t,dt,np.array([X0,Y0]),np.array([X1,Y1]))
            t = t + dt

            #Add the new positions to X,Y,VX,VY
            self.X = np.vstack((self.X,X2))
            self.Y = np.vstack((self.Y,Y2))
            self.VX = np.vstack((self.VX,(X2-X0)/(2*dt)))
            self.VY = np.vstack((self.VY,(Y2-Y0)/(2*dt)))

            #Redefine and repeat
            X0,Y0 = X1,Y1
            X1,Y1 = X2,Y2

            #Update and show progress through console
            progress = t/T*100
            if(i%1000 == 0):
                print(int(progress),'% done')

        #Once the computation has ended, I compute the kinetic energy,
        #the magnitude of the velocity V and the temperature
        #(see doc for temperature definition)
        self.KE()
        self.V = np.sqrt((self.VX**2 + self.VY**2))
        self.T = (np.sum(self.V**2,axis=1)/(self.particles.size*2 - 2))

        #Generation of the MB functions, you can modify the definition by
        #changing the linspace points
        vs,a = np.meshgrid(np.linspace(0,self.V.max(),100),self.T)
        a,ts = np.meshgrid(np.linspace(0,self.V.max(),100),self.T)
        self.MB = (vs/(ts)*np.exp(-vs**2/(2*ts)))

        #Here I generate the accumulated V,T and MB using lists
        #The reason I use lists is because if you append
        #two numpy arrays to an empty numpy array
        #they merge instead of remaining separate
        #You could technically use splicing to save on memory
        #but sacrificing cpu
        self.Vacu = []
        self.Tacu = []
        self.MBacu = []
        self.Vacu.append(self.V[int(self.n/2),:])
        self.Tacu.append(np.sum(self.V[int(self.n/2),:]**2)/(self.particles.size*2 - 2))

        vs = np.linspace(0,self.V.max(),100)
        self.MBacu.append((vs/(self.Tacu[0])*np.exp(-vs**2/(2*self.Tacu[0]))))

        #This delta controls the time interval for accumulation, right now its every 5 units
        delta = 5./dt

        #This 40 that appers in these lines is the time from which I start accumulating
        #to ensure the system has reached equilibrium.
        for i in range(1,int((self.n-(40./dt))/delta)):
            self.Vacu.append(np.hstack((self.Vacu[i-1],self.V[int(40./dt)+int(i*delta),:])))
            self.Tacu.append(np.sum(self.Vacu[i]**2)/(self.Vacu[i].size*2 - 2))
            self.MBacu.append((vs/(self.Tacu[i])*np.exp(-vs**2/(2*self.Tacu[i]))))
        return


    def KE(self):
        #Function for computing the kinetic energy, it also computes the mean kinetic energy.p
        Ki = self.m*(self.VX**2 + self.VY**2)/2.
        self.K = np.sum(Ki,axis=1)[1:]
        self.Kmean = (np.sum(Ki,axis=1)/self.particles.size)[1:]
        return
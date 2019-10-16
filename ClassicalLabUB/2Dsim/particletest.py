

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as pltanim
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import interp1d
from particle import *
from potentials import *



L = 200


T = 30
dt = 0.1
nt = int(T/dt) 

    
pot = Phi()



pot.add_function(woodsaxon,dwoodsaxonx,dwoodsaxony,[35,0,100,10,70,0.3])
#pot.add_function(woodsaxon,dwoodsaxonx,dwoodsaxony,[0,75,-500,10,0.3])
#pot.add_function(gauss,dgaussx,dgaussy,[50,0,-500,5])
#pot.add_function(gauss,dgaussx,dgaussy,[50,-5,-100,20])
#pot.add_function(gauss,dgaussx,dgaussy,[-60,75,100,20])
#pot.add_function(gauss,dgaussx,dgaussy,[-75,3,100,5])
#pot.add_function(rect,drectx,drecty,[25,0,100,20,20])



dx = 1
nx = int(L/dx)
xx,yy, = np.meshgrid(np.linspace(-L/2,L/2,nx,endpoint=True),np.linspace(-L/2,L/2,nx,endpoint=True))
im = np.zeros((nx,nx))
for i in range(0,nx):
    for j in range(0,nx):
        im[i,j] = pot.val(xx[i,j],yy[i,j])

p = Particle(1,1,dt)
p.ComputeTrajectoryF(np.array([0,0,10,0]),T,pot)    
a = p.trajectory
b = p.steps
t = p.steps.cumsum()
#t = dt*np.ones(a.shape[0]).cumsum()

'''
fun = np.zeros(nx)
xs = np.zeros(nx)
for i in range(0,nx):
    x = i*dx - L/2
    y = 0
    xs[i] = x 
    fun[i] = pot.val(x,y)
plt.figure()
plt.plot(xs,fun)
'''

#'''
plt.figure()
plt.subplot(3,3,1)
plt.contourf(xx,yy,im,cmap="gray")
#plt.contour(xx,yy,im)
plt.colorbar()
plt.axis("square")
plt.xlabel('x([L])')
plt.ylabel('y([L])')

plt.subplot(3,3,2)
plt.plot(t,p.KEnergy(),"r-",t,p.PEnergy(),"b-",t,p.Energy(),"g-")
plt.legend(('EC','EP','EM'),loc='best')
plt.xlabel('t([T])')
plt.ylabel('Energy([M]*[L]^2*[T]^-2)')

plt.subplot(3,3,3)
plt.plot(a[:,0],a[:,1])
plt.xlabel('x([L])')
plt.ylabel('y([L])')
plt.axis((-100,100,-100,100),'square')

plt.subplot(3,2,3)
plt.plot(t,a[:,0])
plt.xlabel('t([T])')
plt.ylabel('x([L])')

plt.subplot(3,2,4)
plt.plot(t,a[:,1])
plt.xlabel('t([T])')
plt.ylabel('y([L])')

plt.subplot(3,2,5)
plt.plot(a[:,0],a[:,2])
plt.xlabel('x([L])')
plt.ylabel('x([L][T]^-1)')
plt.axis((-100,100,-100,100),'square')

plt.subplot(3,2,6)
plt.plot(a[:,1],a[:,3])
plt.xlabel('y([L])')
plt.ylabel('vy([L][T]^-1)')
plt.axis((-100,100,-100,100),'square')

#plt.tight_layout()
plt.show()


#'''
#trax = np.poly1d(np.polyfit(t,a[:,0],2))
#trax = Polynomial.fit(t,a[:,0],1)
times = interp1d(t,t)
trax = interp1d(t,a[:,0],kind='quadratic')
tray = interp1d(t,a[:,1],kind='quadratic')
'''

def background():
    plt.contourf(xx,yy,im,cmap="inferno")
    plt.colorbar()
    return image,



def animation(frame):
    inst = frame*(1./25.)
    image.set_data(trax(inst),tray(inst))
    return image,


fps = 25
freqms = (1./25.)*1000
totalframes = int(fps*t.max()) + 1



Writer = pltanim.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=3600)


anim = plt.figure()
image, = plt.plot([],[],'w.',markersize=10)
plt.axis((-100,100,-100,100),'square')
video = FuncAnimation(anim,animation,init_func=background,frames=totalframes,interval=freqms,blit='True')

video.save('video.mp4',writer=writer)
'''


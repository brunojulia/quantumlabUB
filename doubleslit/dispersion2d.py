import numpy as np
import matplotlib.pyplot as plt

def trapz2d(z,x=None,y=None,dx=1.,dy=1.):
    ''' Integrates a regularly spaced 2D grid using the composite trapezium rule.
    IN:
       z : 2D array
       x : (optional) grid values for x (1D array)
       y : (optional) grid values for y (1D array)
       dx: if x is not supplied, set it to the x grid interval
       dy: if y is not supplied, set it to the x grid interval
    '''
    import numpy as N

    sum = N.sum
    if x != None:
        dx = (x[-1]-x[0])/(N.shape(x)[0]-1)
    if y != None:
        dy = (y[-1]-y[0])/(N.shape(y)[0]-1)

    s1 = z[0,0] + z[-1,0] + z[0,-1] + z[-1,-1]
    s2 = sum(z[1:-1,0]) + sum(z[1:-1,-1]) + sum(z[0,1:-1]) + sum(z[-1,1:-1])
    s3 = sum(z[1:-1,1:-1])

    return 0.25*dx*dy*(s1 + 2*s2 + 4*s3)

def sigma(t, s = 0.5):
    return np.sqrt(s**2 + t**2/(4*s**2))

psit = np.load("psit2d.npy")
t = np.loadtxt("times2d.dat")
x = np.loadtxt("x2d.dat")
y = np.loadtxt("y2d.dat")

dx = y[1]-y[0]
p = np.absolute(psit)**2

r = np.sqrt(x**2+y**2)
mean_r = np.array([trapz2d(x*p[i], dx = dx, dy=dx) for i in range(p.shape[0])])
mean_r2 = np.array([trapz2d(x**2*p[i], dx = dx, dy=dx) for i in range(p.shape[0])])
print(np.min(sigma(t)), np.min(mean_r2-(mean_r)**2))
plt.plot(t, mean_r2-(mean_r)**2)
plt.plot(t, sigma(t))
plt.show()

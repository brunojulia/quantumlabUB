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

def psi0(x, y):
    """
    Wave function at t = 0
    """
    x0 = 0
    y0 = 0
    s = 0.5
    p0 = 0.0/Lx
    r2 = (x-x0)**2 + (y-y0)**2
    return np.exp(-1j*p0*x)*np.exp(-r2/(4*s**2))/(2*s**2*np.pi)**(.5)

Lx = 17
Ly = 17
dx = 400
x, y = np.meshgrid(np.arange(-Lx, Lx, dx), np.arange(-Ly, Ly, dx))

psi = psi0(x, y)

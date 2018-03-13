"""
Solution to the time dependent Schrodinger equation in 2D.
Of a free particle

The convention for the diagonals is the following:
    a = inferior diagonal, with a[0] = 0
    b = central diagonal
    c = superior diagonal, with c[-1] = 0

"""

import numpy as np
import matplotlib.pyplot as plt

#Constants
hbar = 1
m = 1

#Grid properties
Lx = 17.0
Ny = 100
Nx = 200
dx = 2*Lx/Nx
Ly = Ny*dx/2



x, y = np.meshgrid(np.arange(-Lx, Lx, dx), np.arange(-Ly, Ly, dx))

print(x.shape, y.shape)
#Time parameters
dt = 0.01
tmax = 2

r = 1j*dt*hbar/(4*m*(dx)**2)

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
    x0 = -12
    y0 = 0
    s = np.sqrt(2)
    p0 = 200.0/Lx

    psi0 = np.exp(-((x-x0)**2+(y-y0)**2)/(2*s**2))/np.sqrt(np.pi*s**2)

    """psi2 = np.absolute(psi0)**2
    print(trapz2d(psi2,dx=dx,dy=dx))
    plt.imshow(psi2, cmap = "gray")
    plt.colorbar()
    plt.show()"""

    return psi0

def diagonalsA(size):
    """
    Size is the size of the matrix (when computing Ax -> Nx and Ay -> Ny)
    """
    b = np.full(size, 1+2*r)
    a = np.full(size, -r)
    a[0] = 0
    c = np.full(size, -r)
    c[-1] = 0

    return a, b, c

def compute_bx(row, psi):
    """
    Row is the row index.
    Returns the vector bx, dimension Nx (independent terms when solving rows)
    """

    bx = np.zeros(Nx, dtype = np.complex)
    if row != 0:
        bx += r*psi[row-1,:]

    if row != Ny-1:
        bx += r*psi[row+1,:]

    return bx + (1-2*r)*psi[row,:]

def compute_by(col, psit):
    """
    Col is the column index
    Return the vector by, dimension Ny (independent terms when solving columns)
    """

    by = np.zeros(Ny, dtype = np.complex)
    if col != 0:
        by += r*(psit[:,col-1])
    if col != Nx-1:
        by += r*(psit[:,col+1])

    return by + (1-2*r)*psit[:,col]

#Aquest l'he trobat per internet
## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    '''
    nf = len(a)     # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))     # copy the array
    for it in range(1, nf):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1]
        dc[it] = dc[it] - mc*dc[it-1]

    xc = ac
    xc[-1] = dc[-1]/bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il]-cc[il]*xc[il+1])/bc[il]

    del bc, cc, dc  # delete variables from memory

    return xc

def clean_file(filename = "psi2d.dat"):
    with open(filename, "w") as outFile:
        outFile.write("")

def save_psi(t, psi, filename = "psi2d.dat"):
    psi2 = np.square(np.absolute(psi))
    with open(filename, "a") as outFile:
        for i in range(Nx):
            for j in range(Ny):
                outFile.write(str(i) + "," + str(j) + "," + str(psi2[j,i]) + "\n")
        outFile.write("$"+str(t) + "\n")


if __name__ == '__main__':
    clean_file()

    psi = psi0(x, y).astype(np.complex)

    Axa, Axb, Axc = diagonalsA(Nx)
    Aya, Ayb, Ayc = diagonalsA(Ny)


    t = 0

    while t < tmax:
        print(t, trapz2d(np.absolute(psi)**2,dx=dx,dy=dx))
        save_psi(t, psi)

        psit = np.zeros(psi.shape, dtype = np.complex)
        #rows
        for j in range(Ny):
            bx = compute_bx(j, psi)
            psit[j,:] = TDMAsolver(Axa, Axb, Axc, bx)

        #columns
        for i in range(Nx):
            by = compute_by(i, psit)
            psi[:,i] = TDMAsolver(Aya, Ayb, Ayc, by)

        t += dt

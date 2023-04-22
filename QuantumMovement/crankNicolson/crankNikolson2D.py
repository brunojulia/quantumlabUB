
import types
import numpy as np
# import scipy.fft ??? Fast Fourier Transform, recordar
import numba
from numba import jit

"""
Resolució de l'equació de Schrodinger dependent del temps no relativista en 2D:
- Joan Ainaud Fondevila

Funcionament general:
La funció d'ona psi(x, y, t) va descrita per arrays numpy  x -> x_i,  y -> y_j
La evolució va a un pas: S'obté l'array que descriu psi(x, y, t+dt)
Mètode Crank Nicolson ADI 2 mig passos dt/2, alternant direcció de derivades


Tot fet en Python, optimitzat amb @jit i numpy per anar més ràpid a operacions, sobretot bucles

Es podria mirar de paral·lelitzar (ex. energia):

Numba global interpreter lock: nogil=True
Parallelize
Numba: Parallel = True


Si resulta molt important aconseguir el màxim de rendiment es pot fer servir Fortran/C
https://numpy.org/doc/stable/user/c-info.python-as-glue.html

Fortran to Python: https://numpy.org/doc/stable/f2py/
"""

hred = 1  # Unitats naturals, hred es refereix a h reduïda: h/2pi
M = 1


# It's hard to test when is a method better than another
# Sometimes abs2 is faster: Mod[:][:] = abs2(psi)
# Sometimes, when changing psi before, or with very big matrices, abs2Matrix is faster (jit functions have overhead).
@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


@jit(nopython=True)
def abs2Matrix(psi, result):
    for i in range(len(psi)):
        for j in range(len(psi[0])):
            result[i][j] = psi[i][j].real ** 2 + psi[i][j].imag ** 2

@jit(nopython=True)
def abs2MatrixMultiplied(X, Y, func, psi, result, t=0., extra_param=None):
    for i in range(len(psi)):
        for j in range(len(psi[0])):
            result[i][j] = (psi[i][j].real ** 2 + psi[i][j].imag ** 2) * func(X[i], Y[j], t=t, extra_param=extra_param)


@jit(nopython=True)
def set2DMatrix(X, Y, func, psi, t=0., extra_param=None):
    for i in range(len(X)):
        for j in range(len(Y)):
            psi[i][j] = func(X[i],Y[j],t, extra_param)


def simpson_complex_list(dx, psi, rec=0, pos=()):
    """
    Calculates integral of n dimensional function saved as numpy array psi.
    The values must be evenly spaced and within a rectangular grid
    :param dx: tuple of all the dx in each direction. (Evenly spaced grid)
    :param psi: Function, as an array, to be integrated
    :param rec: Integration is done recursively, rec indicates recursion step
    :param pos: pos indicates which part of the subarray we are integrating in the recursive step
    :return: Result of integral, Complex type
    """
    size = len(psi[pos])    # Ideally size should be odd. If not, we need to add last term with trapezoid
    if size == 2:
        print("No es pot fer simpson amb només 2 punts, alguna cosa ha anat malament")
        return None
    if rec < psi.ndim - 1:
        integral = simpson_complex_list(dx, psi, rec+1, pos+(0,))
        # We integrate over this dimension, with size
        for it in range(1, size-3, 2):
            integral += 4*simpson_complex_list(dx, psi, rec+1, pos+(it  ,))
            integral += 2*simpson_complex_list(dx, psi, rec+1, pos+(it+1,))
        # After the loop, if size is odd we only miss: [size-2] [size-1]. otherwise [size-3] [size-2] [size-1]
        if size % 2 == 1:
            integral += 4*simpson_complex_list(dx, psi, rec+1, pos+(size-2,))
            integral +=   simpson_complex_list(dx, psi, rec+1, pos+(size-1,))
        else:
            integral += 4 * simpson_complex_list(dx, psi, rec + 1, pos + (size - 3,))
            integral += 2.5*simpson_complex_list(dx, psi, rec + 1, pos + (size - 2,))
            integral += 1.5*simpson_complex_list(dx, psi, rec + 1, pos + (size - 1,))
    else:
        # Same as before, but we can integrate now
        integral = psi[pos][0]
        for it in range(1, size-3, 2):
            integral += 4 * psi[pos][it]
            integral += 2 * psi[pos][it+1]
        if size % 2 == 1:
            integral += 4 * psi[pos][size-2]
            integral += psi[pos][size-1]
        else:
            integral += 4 * psi[pos][size-3]
            integral += 2.5 * psi[pos][size-2]
            integral += 1.5 * psi[pos][size-1]

    integral = integral*dx[rec]/3
    return integral


""" SPEED TEST. Each method is faster than the last
N = 1000001
list1 = np.linspace(-100,100,N)
dx = 200/N

t0 = time.time()
integral = list1[0]
for i in range(1,N-1-(N % 2 == 0)):
    if i%2==0: integral += 2*list1[i]
    else: integral += 4*list1[i]
integral += list1[N-1-(N%2==0)]
integral = integral*dx/3
if(N%2==0): integral += (list1[N-1]+list1[N-2])*dx/2
print("Integral: ", integral, "   ha trigat: ", time.time()-t0, " (s)")

t0 = time.time()
integral = list1[0]
for i in range(1,(N-1)//2):
    integral += 4*list1[2*i-1]
    integral += 2*list1[2*i]
if N % 2 == 1:
    integral += 4 * list1[N-2]
    integral += list1[N-1]
else:
    integral += 4 * list1[N - 3]
    integral += 2.5 * list1[N - 2]
    integral += 1.5 * list1[N - 1]
integral = integral * dx / 3
print("Integral: ", integral, "   ha trigat: ", time.time()-t0, " (s)")

t0 = time.time()
integral = list1[0]
for i in range(1,N-3,2):
    integral += 4*list1[i]
    integral += 2*list1[i+1]
if N % 2 == 1:
    integral += 4 * list1[N-2]
    integral += list1[N-1]
else:
    integral += 4 * list1[N - 3]
    integral += 2.5 * list1[N - 2]
    integral += 1.5 * list1[N - 1]
integral = integral * dx / 3
print("Integral: ", integral, "   ha trigat: ", time.time()-t0, " (s)")
"""


@jit(nopython=True)
def simpson_complex_list1D(dx, psi) -> np.complex128:
    integral = psi[0]
    for it in range(1, len(psi) - 3, 2):
        integral += 4 * psi[it]
        integral += 2 * psi[it + 1]
    if len(psi) % 2 == 1:
        integral += 4 * psi[len(psi) - 2]
        integral += psi[len(psi) - 1]
    else:
        integral += 4 * psi[len(psi) - 3]
        integral += 2.5 * psi[len(psi) - 2]
        integral += 1.5 * psi[len(psi) - 1]
    integral = integral * dx / 3.
    return integral


@jit(nopython=True)
def simpson_complex_list2D(dx, dy, psi) -> np.complex128:
    integral = simpson_complex_list1D(dy, psi[0])
    for it in range(1, len(psi) - 3, 2):
        integral += 4 * simpson_complex_list1D(dy, psi[it])
        integral += 2 * simpson_complex_list1D(dy, psi[it + 1])
    if len(psi) % 2 == 1:
        integral += 4 * simpson_complex_list1D(dy, psi[len(psi) - 2])
        integral += simpson_complex_list1D(dy, psi[len(psi) - 1])
    else:
        integral += 4 * simpson_complex_list1D(dy, psi[len(psi) - 3])
        integral += 2.5 * simpson_complex_list1D(dy, psi[len(psi) - 2])
        integral += 1.5 * simpson_complex_list1D(dy, psi[len(psi) - 1])

    integral = integral * dx / 3
    return integral


def applyOperatorPoint(psi, operator, pos, X, dx=1, t=0):
    if operator.isinstance(types.FunctionType):
        operator = operator(psi.ndim, X, dx, t)

    result = operator[0] * psi[pos]
    atBoundary = False
    for i in range(len(pos)):
        if pos[i] == psi.shape[i] - 1 or pos[i] == 0: atBoundary = True
    if not atBoundary:
        for it2 in range(1, len(operator)):
            pos_minus = tuple(pos[i] - (i == it2) for i in range(len(pos)))
            pos_plus = tuple(pos[i] + (i == it2) for i in range(len(pos)))
            result += operator[it2][0] * psi[pos_minus] + operator[it2][1] * psi[pos_plus]

    return result


def applyOperator(x0, xf, psi, result, operator, t=0., doConjugate=False):
    """
    Sets result to copy of psi with the operator applied to it
    :param operator: 1D operators have shape [(), [(), ()]]. 2D operators [(), [(), ()], [(), ()]] etc.
    Applied to a point, the first element of the operator shows dependance on self. First list shows dependence on
    neighbours in x direction, second list in y direction, etc.
    Example in 2D:  (i: x,  j: y)
                                        [ xi-j-  xi-j   xi-j+ ]
    OPERATOR [m,[a-, a+],[b-,b+]] AT:   [ xi j-  xi j   xi j+ ]  ==>  m * xij + (a- * xi-j) + (a+ * xi+j)
                                        [ xi+j-  xi+j   xi+j+ ]               + (b- * xij-) + (b+ * xij+)
    Operator can be given either as the np.ndarray or as a function that generates the np.ndarray
    Note. in order to be given as a ndarray a rectangle shape must be taken, giving an extra garbage value:
    OPERATOR [[m, 0],[a-, a+],[b-,b+]].  Necessary to work with numba : @jit
    :param t: Optional, in case function depends on time
    :param doConjugate: Optional. If doConjugate = True, then every element is also multiplied by conjugate of psi,
    useful for calculating expected value later
    In general operators can't be applied at the bounds
    """
    if np.shape(result) != np.shape(psi): raise ValueError("Copy and original nparray must have the same shape")
    if type(operator) is list:
        if len(operator) != psi.ndim+1: raise ValueError("Operator doesn't match dimensions of matrix")
        for pos in np.ndindex(psi.shape):
            result[pos] = operator[0][0]*psi[pos]
            atBoundary = False
            for i in range(len(pos)):
                if pos[i] == psi.shape[i] - 1 or pos[i] == 0: atBoundary = True
            if not atBoundary:
                for it2 in range(1, len(operator)):  # it2 indicates dimension. 1: x, 2: y, 3: z, etc
                    pos_minus = tuple([pos[i] - (i == it2 - 1) for i in range(len(pos))])
                    pos_plus = tuple([pos[i] + (i == it2 - 1) for i in range(len(pos))])
                    result[pos] += operator[it2][0] * psi[pos_minus] + operator[it2][1] * psi[pos_plus]
            if doConjugate: result[pos] = result[pos] * psi[pos].conjugate()

    elif type(operator) is types.FunctionType:
        dx = np.divide((np.array(xf) - np.array(x0)), tuple([shape_i - 1 for shape_i in psi.shape]))
        for pos in np.ndindex(psi.shape):
            X = np.array(x0) + np.multiply(dx, pos)
            op = operator(psi.ndim, X, dx, t)
            result[pos] = op[0] * psi[pos]
            atBoundary = False
            for i in range(len(pos)):
                if pos[i] == psi.shape[i] - 1 or pos[i] == 0: atBoundary = True
            if not atBoundary:
                for it2 in range(1, len(op)):
                    pos_minus = tuple([pos[i] - (i == it2 - 1) for i in range(len(pos))])
                    pos_plus = tuple([pos[i] + (i == it2 - 1) for i in range(len(pos))])
                    result[pos] += op[it2][0] * psi[pos_minus] + op[it2][1] * psi[pos_plus]
            if doConjugate: result[pos] = result[pos] * psi[pos].conjugate()

    else: raise ValueError("Invalid type for operator")
"""# Code to show applying operator : partial derivative x
operator = [[-1./dx[0]], [0., +1./dx[0]] , [0.,0.]]
print(operator)

np.conjugate(psi, out=psiCopy)
np.multiply(psi, psiCopy, out=psiCopy)
resTemp = np.real(psiCopy)
res = np.copy(resTemp)
mathPhysics.applyOperator(X0, Xf, resTemp, res, operator)

X = np.linspace(X0[0], Xf[0], Nx)
Y = np.linspace(X0[1], Xf[1], Ny)
plt.gca().set_aspect("equal")
plt.pcolor(X,Y,res.T)
#plt.plot(X, res)
plt.show()
"""



#@jit(nopython=True)
def applyOperator2D(X, Y, psi, result, operator, t=0., extra_param=None, doConjugate = False):
    if len(X) != len(psi) or len(Y) != len(psi[0]): raise ValueError("X and Y coordinates don't match shape of Psi")
    if not(callable(operator)):
        applyOperator2DOp(X, Y, psi, result, operator, doConjugate=doConjugate)
    else:
        applyOperator2DFunc(X, Y, psi, result, operator, t=t, extra_param=extra_param, doConjugate=doConjugate)


@jit(nopython=True)
def applyOperator2DOp(X, Y, psi, result, operator, doConjugate = False):
    #if len(X) != len(psi) or len(Y) != len(psi[0]): raise ValueError("X and Y coordinates don't match shape of Psi")
    if len(operator) != psi.ndim+1: raise ValueError("Operator doesn't match dimensions of matrix")
    if not doConjugate:
        for i in range(1, len(psi)-1):
            for j in range(1, len(psi[0]-1)):
                result[i,j] = operator[0][0]*psi[i,j] + operator[1][0] * psi[i-1,j] + operator[1][1] * psi[i+1,j] + \
                                                        operator[2][0] * psi[i,j-1] + operator[2][1] * psi[i,j+1]
        result[0, :] = psi[0, :]*operator[0][0]
        result[len(psi)-1, :] = psi[len(psi)-1, :] * operator[0][0]
        result[:, 0] = psi[:, 0] * operator[0][0]
        result[:, len(psi[0])-1] = psi[:, len(psi[0])-1] * operator[0][0]
    else:
        for i in range(1, len(psi)-1):
            for j in range(1, len(psi[0]-1)):
                result[i,j] = operator[0][0]*psi[i,j] + operator[1][0] * psi[i-1,j] + operator[1][1] * psi[i+1,j] + \
                                                        operator[2][0] * psi[i,j-1] + operator[2][1] * psi[i,j+1]
                result[i,j] = result[i,j] * np.conj(psi[i,j])
        result[0, :] = abs2(psi[0, :])*operator[0][0]
        result[len(psi)-1, :] = abs2(psi[len(psi)-1, :]) * operator[0][0]
        result[:, 0] = abs2(psi[:, 0]) * operator[0][0]
        result[:, len(psi[0])-1] = abs2(psi[:, len(psi[0])-1]) * operator[0][0]


@jit(nopython=True)
def applyOperator2DFunc(X, Y, psi, result, operator, t=0., extra_param=None, doConjugate=False):
    if len(X) != len(psi) or len(Y) != len(psi[0]): raise ValueError("X and Y coordinates don't match shape of Psi")
    dx = (X[-1] - X[0]) / (len(psi) - 1)
    dy = (Y[-1] - Y[0]) / (len(psi[0]) - 1)
    op = np.array([[1., 0.], [0., 0.], [0., 0.]])
    operator(op, (X[0], Y[0]), (dx, dy), t, extra_param=extra_param, onlyUpdate=False)
    if not doConjugate:
        for i in range(1, len(psi) - 1):
            for j in range(1, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0][0] * psi[i, j] + op[1][0] * psi[i - 1, j] + op[1][1] * psi[i + 1, j] + \
                               op[2][0] * psi[i, j - 1] + op[2][1] * psi[i, j + 1]
        for i in range(len(psi)):
            for j in (0, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0][0] * psi[i, j]
        for j in range(1, len(psi[0]) - 1):
            for i in (0, len(psi) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0][0] * psi[i, j]
        #if doConjugate: result[...] = result[...] * np.conj(psi[...])
    else:
        for i in range(1, len(psi) - 1):
            for j in range(1, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0][0] * psi[i, j] + op[1][0] * psi[i - 1, j] + op[1][1] * psi[i + 1, j] + \
                               op[2][0] * psi[i, j - 1] + op[2][1] * psi[i, j + 1]
                result[i, j] = result[i, j] * np.conj(psi[i, j])
        for i in range(len(psi)):
            for j in (0, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0][0] * abs2(psi[i, j])
        for j in range(1, len(psi[0]) - 1):
            for i in (0, len(psi) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0][0] * abs2(psi[i, j])


#@jit(nopython=True)
def expectedValueOperator2D(X, Y, psi, result, operator = None, t = 0., extra_param=None):
    if np.shape(result) != np.shape(psi): raise ValueError("Psi and result must have the same shape")
    if operator is None:            # Calculate total probability, should be 1
        np.conjugate(psi, out=result)
        np.multiply(psi, result, out=result)
        return simpson_complex_list2D((X[-1]-X[0])/(len(psi)-1), (Y[-1]-Y[0])/(len(psi[0])-1), result)    #dx, dy
    applyOperator2D(X, Y, psi, result, operator, t=t, extra_param=extra_param, doConjugate=True)
    return simpson_complex_list2D((X[-1]-X[0])/(len(psi)-1), (Y[-1]-Y[0])/(len(psi[0])-1), result)        #dx, dy


def expectedValueOperator(x0, xf, psi, result, operator=None, t = 0.):
    """
    Calculates expected value of operator: ∫...∫(psi)* · Operator[psi] d^n x
    If no operator is given it calculates total probability (should be 1).
    The area to integrate over is rectangular, but it can be restricted by
    defining the operator at each point (IE only non 0 around a circle).
    :param t: time
    :param x0: lower integral bounds, list/tuple
    :param xf: upper integral bounds, list/tuple. Reminder, single element tuple: (x,) not (x)
    :param psi: Function, as an array, over which to calculate expected value
    :param result: Array with same shape as psi. Intermediate copies are stored here
    :param operator: Operator. IE: Identity (prob. density function), Hamiltonian, etc.
    :return: Complex, expected value of operator. Most operators should have real expected value,
    such as observables. It is useful then to see if the complex part is almost 0 to check it makes sense
    """
    if np.shape(result) != np.shape(psi): raise ValueError("Copy and original nparray must have the same shape")
    if operator is None:            # Calculate total probability, should be 1
        np.conjugate(psi, out=result)
        np.multiply(psi, result, out=result)
        return simpson_complex_list(x0,xf,result)
    applyOperator(x0, xf, psi, result, operator, t=t, doConjugate=True)
    return simpson_complex_list(x0, xf, result)


"""
Python problems. Not being able to optimize the general case makes it not worthwile. The Kinetic Energy
for a 2D (201x201 points) gaussian packet is calculated using first the general method and then the 2D one.
Specific 2D case is an order of magnitude faster
Process took:  1.353860855102539  (s)
(25.162173517551764+0.023004578141175114j)
Process took:  0.18692708015441895  (s)
(25.162173517551764+0.023004578141174608j)
"""

#@jit(nopython=True) Can't. fft2 not supported with numba
def fourierTransform2D(X, Y, psi, result):
    result[:][:] = np.fft.fft2(psi[:][:], norm="ortho")  # Older 1.10.0- numpy versions may not allow ortho
                                                            # normalization. Equal to dividing by sqrt(lenX * lenY)
    result[:][:] = np.fft.fftshift(result[:][:])            # Center freq (by default 0 freq is on bottom left)
    #Px = np.fft.fftfreq(len(X))
    #Px = np.fft.fftfreq(len(Y))


def kineticEnergyN(n, X, dx, t=0, dir=-1):
    # -h^2/2m ∇^2
    # if dir != -1,  dir indicates only one direction dir=1: x, dir=2: y, etc
    global hred, M
    if dir == -1:
        K = [+hred*hred/(M) * sum([1/delta**2 for delta in dx])]
        K += [2*[-hred*hred/(2*M)/delta**2] for delta in dx]
    else:
        K = [+hred*hred/(M)/(dx[dir])**2]
        K += [[0, 0]]*(dir-1)
        K += [2*[-hred*hred/(2*M)/dx[dir-1]**2]]
        K += [[0, 0]]*(n-dir)
    return K


@jit
def kineticEnergy(op, X, dx, t=0, extra_param=None, dir=-1, onlyUpdate=True):
    # -h^2/2m ∇^2
    # if dir != -1,  dir indicates only one direction dir=1: x, dir=2: y, etc
    global hred, M
    if onlyUpdate: return # The operator doesn't need to be updated, as dx and dy constant -> operator constant
    if dir == -1:
        laplac = 0.
        for delta in dx: laplac += 1./delta**2
        op[0][0] = hred*hred/M * laplac
        for it in range(1,len(op)):
            op[it][0] = -hred*hred/(2*M) / dx[it-1]**2
            op[it][1] = op[it][0]
    else:
        op[0][0] = +hred*hred/(M)/(dx[dir])**2
        for it in range(1, len(op)):
            op[it][0] = -hred * hred / (2 * M) / dx[it] ** 2
            op[it][1] = op[it][0]
        op[dir][0] = -hred * hred / (2 * M) / dx[dir] ** 2
        op[dir][1] = op[dir][0]


#@jit
def potentialEnergyGenerator(potential):
    def potentialEnergy(op, X, dx, t=0, dir=-1, onlyUpdate=True):
        # V(x,y,t,) Psi
        # if dir != -1,  dir indicates only one direction dir=1: x, dir=2: y, etc. Doesn't matter in this case
        global hred, M
        if onlyUpdate:
            op[0][0] = potential(*X, t)
        else:
            op[0][0] = potential(*X, t)
            for i in range(1, len(op)):
                op[i][0] = 0.
                op[i][1] = 0.
    return potentialEnergy


@jit(nopython=True)
def tridiag(gamma, trid, bVec, xVec):
    """
    Solves tridiagonal system. Tridiag is a matrix of the form (A- = trid[0][:], A = trid[1][:], A+ = trid[2][:]
    Thomas algorithm
    matrix:     A-(0) |A (0) A+(0) ...   0   |
                      |A-(1) A (1) ...   0   |    * PSI    =   bVec
                      | ...   ...  ...  ...  |
                      |  0     0   ... A (n) |
    :param gamma: numpy vector of size n+1. Must be already created to avoid allocating memory every time tridiag is called
    :param trid: tridiagonal matrix
    :param bVec: Right side vector
    :param xVec: Unknown vector to find
    :return:
    """
    #if trid[1][0] == 0.: raise ValueError("matriu tridiagonal invàlida pel mètode tridiagonal, 0 a la diagonal")
    beta = trid[1][0]
    xVec[0]=bVec[0]/beta
    for j in range(1,len(bVec)):
        gamma[j]=trid[2][j-1]/beta
        beta=trid[1][j]-trid[0][j]*gamma[j]
        #if beta == 0: raise ValueError("Matriu singular")     # It is better to check == 0 with tolerance
        xVec[j]=(bVec[j]-trid[0][j]*xVec[j-1])/beta

    for j in range(len(bVec)-1):
        xVec[len(bVec)-2-j] = xVec[len(bVec)-2-j] - gamma[len(bVec)-j-1]*xVec[len(bVec)-j-1]


@jit(nopython=True)
def crankNikolson2DSchrodingerStep(X, Y, t, dt, potential, psi, psiTemp, extra_param=None):
    """ Solves System:  i h_red d/dt psi = [-h_red^2 /2 (d^2/dx^2 + d^2/dy^2)  + V ] psi
    Step 0: Psi(t)  ->   Step 1: psiTemp = Psi(t+dt/2) ->  Step 2: psi = Psi(t+dt)"""

    global hred, M

    # To make the code even faster, all of these temporary arrays could be created already. So no step needs to be taken
    # Declare all necessary arrays. Trying to minimize need to allocate memory
    tempBX = np.empty(len(psi)-2, dtype=np.complex128)
    tempBY = np.empty(len(psi[0])-2, dtype=np.complex128)

    tridX = np.empty((3,len(psi)-2), dtype=np.complex128)
    tridY = np.empty((3,len(psi[0])-2), dtype=np.complex128)

    # Gamma is used for tridiag, it is allocated here to avoid repeated unnecessary memory allocation
    gamma = np.empty((max(len(psi)-1, len(psi[0])-1),), dtype=np.complex128)

    # Parameters for calcualtions
    dx = (X[-1]-X[0])/(len(X))
    dy = (Y[-1]-Y[0])/(len(Y))
    rx = 1j*dt*hred/(2*M*dx**2)
    cent_rx = 2*(1+rx)
    ry = 1j*dt*hred/(2*M*dy**2)
    cent_ry = 2*(1+ry)

    # We set tridiagonal matrices. The main diagonal depends on the potential, which depends on position, so it's done later
    tridX[0][:] = -rx
    #tridX[1][:] = 2*(1+rx) + 1j * dt * V(x,y)
    tridX[2][:] = -rx
    tridY[0][:] = -ry
    # tridY[1][:] = 2*(1+ry) + 1j * dt * V(x,y)
    tridY[2][:] = -ry

    # First half step. At t we only do derivatives in y direction. At t+dt/2 in x direction. Thus the name, alternating
    # direction implicit method. Important to remember that at boundaries wave is fixed (infinite potential wall)
    for ity in range(1, len(psi[0])-1):
        for itx in range(1, len(psi)-1):
            tridX[1][itx-1] = cent_rx + 1j * dt/2 * potential(X[itx], Y[ity], t+dt/2, extra_param=extra_param)
            tempBX[itx-1] = (psi[itx][ity-1] + psi[itx][ity+1])*ry \
                            + (2*(1-ry) - 1j * dt/2 * potential(X[itx], Y[ity], t, extra_param=extra_param))*psi[itx][ity]
        tridiag(gamma, tridX, tempBX, psiTemp[1:-1,ity])
    psiTemp[:,0] = psi[:,0]
    psiTemp[:, len(psi[0]) - 1] = psi[:,len(psi[0]) - 1]
    psiTemp[0,:] = psi[0,:]
    psiTemp[len(psi)-1,:] = psi[len(psi)-1,:]

    # We do the final half step. Again, at t+dt/2 we only do derivatives in x direction, but now at t+dt in x direction
    for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            tridY[1][ity-1] = cent_ry + 1j * dt/2 * potential(X[itx], Y[ity], t+dt, extra_param=extra_param)
            tempBY[ity-1] = (psiTemp[itx-1][ity] + psiTemp[itx+1][ity])*rx \
                            + (2*(1-rx) - 1j * dt/2 * potential(X[itx], Y[ity], t+dt/2, extra_param=extra_param))\
                              *psiTemp[itx][ity]
        tridiag(gamma, tridY, tempBY, psi[itx,1:-1])


@jit
def potential0(x, y, t=0., extra_param=None):
    return 0.


# https://numba.pydata.org/numba-doc/latest/user/jitclass.html
class QuantumSystem2D:
    def __init__(self, Nx, Ny, x0, y0, xf, yf, initState, t = 0., potential = potential0, extra_param=None):
        self.Nx, self.Ny = Nx, Ny
        self.x0, self.y0 = x0, y0
        self.xf, self.yf = xf, yf
        self.dx, self.dy = (xf-x0)/(Nx-1), (yf-y0)/(Ny-1)
        self.X, self.Y = np.linspace(x0, xf, Nx, dtype=np.float64), np.linspace(y0, yf, Ny, dtype=np.float64)
        self.Px, self.Py = 2*np.pi*np.sort(np.fft.fftfreq(Nx))/self.dx, 2*np.pi*np.sort(np.fft.fftfreq(Ny))/self.dy
                           # Maybe in general it needs to be multiplied by hred
        self.t = t  # By default t=0.
        self.psi = np.empty((Nx, Ny), dtype=np.complex128)  # Will hold psi
        self.psiMod = np.empty((Nx, Ny), dtype=np.float64)  # Will hold |psi|^2
        self.psiCopy = np.copy(self.psi)                    # Will hold necessary intermediate step matrices
                                                            # Total memory cost ~2.5 psi ~= 2.5*Nx*Ny*128 bits
        self.Xmesh, self.Ymesh = np.meshgrid(self.X, self.Y, copy=False, indexing='ij')
        self.Pxmesh, self.Pymesh = np.meshgrid(self.Px, self.Py, copy=False, indexing='ij')

        if callable(initState):
            self.psi[:][:] = initState(self.Xmesh, self.Ymesh)
            #set2DMatrix(self.X, self.Y, initState, self.psi, t=t)
            # Potentially faster, but needs jit, not as flexible,
            # and regardless it's a one time thing
        else:
            if initState.shape != self.psi.shape: exit(print("Error: Initial state shape different from system's"))
            self.psi[:][:] = initState[:][:]

        self.psi = self.psi/np.sqrt(self.norm()) # Initial state gets automatically normalized

        self.potential = potential
        self.op = np.array([[1., 0.], [0., 0.], [0., 0.]])

        self.extra_param = extra_param
        # The system is a rectangular "box". Equivalent to having infinite potential walls at the boundary

    def evolveStep(self, dt):
        crankNikolson2DSchrodingerStep(self.X, self.Y, self.t, dt, self.potential,
                                       self.psi, self.psiCopy, extra_param=self.extra_param)
        self.t += dt

    def momentumSpace(self):
        fourierTransform2D(self.X, self.Y, self.psi, self.psiCopy)
        self.psiMod[:][:] = abs2(self.psiCopy)        # Seems faster from tests
        #abs2Matrix(self.psiCopy, self.psiMod)

    def modSquared(self):
        self.psiMod[:][:] = abs2(self.psi)            # Seems faster from tests. At high N abs2Matrix is better
        #abs2Matrix(self.psi, self.psiMod)

    def norm(self):
        return expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy)

    def kineticEnergy(self):
        #kineticEnergy(self.op, (self.x0, self.y0), (self.dx, self.dy), onlyUpdate=False)
        #return expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy, self.op)
        return expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy, kineticEnergy)

    def potentialEnergy(self):
        # Possible bottleneck
        abs2MatrixMultiplied(self.X, self.Y, self.potential, self.psi, self.psiMod,
                             t=self.t, extra_param=self.extra_param)

        #self.psiMod[:][:] = abs2(self.psi[:][:]) * self.potential(self.Xmesh[:][:], self.Ymesh[:][:], self.t)
        # Can't include IFs this first way apparently
        return simpson_complex_list2D(self.dx, self.dy, self.psiMod)

    def approximateEigenstate(self):
        """
        Iterates some time until, approximately, only the lowest energy eigenvector remains significant.
        Right now it actually just iterates some steps.
        Problem, currently it's not always compatible with jit (time is assumed to be float).
        Potential is real, but program interprets it as complex
        """
        t0 = self.t
        for _ in range(50):
            for __ in range(10):
                self.evolveStep(-0.0625j)
            self.psi[:][:] *= 1./np.sqrt(self.norm())

        self.t = float(t0)
    def expectedValueOp(self, operator):
        return expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy,
                                       operator=operator, t=self.t, extra_param=self.extra_param)

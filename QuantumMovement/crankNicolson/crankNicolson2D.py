"""
Resolució de l'equació de Schrodinger dependent del temps no relativista en 2D:
- Joan Ainaud Fondevila

Funcionament general:
La funció d'ona psi(x, y, t) va descrita per arrays numpy  x -> x_i,  y -> y_j
La evolució va a un pas: S'obté l'array que descriu psi(x, y, t+dt) a partir de l'anterior psi(x, y, t)
Mètode Crank Nicolson ADI: 2 mig passos dt/2, alternant direcció de derivades. Així el cost és lineal


Tot fet en Python, optimitzat amb @jit i numpy per anar més ràpid a operacions, sobretot bucles (python en sí és lent)

Es podria mirar de paral·lelitzar (ex. per càlculs com energia/norma/etc)

Numba global interpreter lock: nogil=True
Parallelize
Numba: Parallel = True


Si resulta molt important aconseguir el màxim de rendiment es pot fer servir Fortran/C
https://numpy.org/doc/stable/user/c-info.python-as-glue.html

Fortran to Python: https://numpy.org/doc/stable/f2py/
"""

# IMPORTS
import types
import numpy as np
import numba
from numba import jit
from numba import njit
import random
from scipy.linalg import solve_banded#, solveh_banded
from scipy.fft import dstn
from scipy.fft import idstn






###### ####### ######---------------------------------------
###### UNITATS ######---------------------------------------
###### ####### ######---------------------------------------

# Problema amb les unitats:
#  - Per com funcionen funcions un cop fet @jit, les variables globals no s'actualitzen
#    (vegis l'abús d'extra_param a tot arreu per poder simular variables globals)
#  - Per tant, no es poden canviar unitats. Ex. fer electro, després fer àtom. etc.
#  - Es farà per un electró

hred = 1  # Unitats naturals, hred es refereix a h reduïda: h/2pi
          # 1.05457182 × 10-34 m2 kg / s = 0.6582119561 eV · fs
M = 1
# Mass Electron: 9.1093837 × 10-31 kg

# Aquestes unitats són compatibles amb que:
# Agafant unit. d'energia eV
# 1 de distància correspon a 2.76 Å
# 1 de temps correspon a 0.658 fs

# Una manera més "bonica" però aproximada és fent d'unitats:
# 1 d'energia correspon a            2 eV
# 1 de distància a        1.9519 ~   2 Å
# 1 de temps a            0.3291 ~ 1/3 fs


# ELECTRÓ: Si fem servir unitats estàndard:
#  - distancia (Å),   temps (fs),    energia (eV)





# It's hard to test when is a method better than another
# Sometimes abs2 is faster: Mod[:,:] = abs2(psi)
# Sometimes, when changing psi before, or with very big matrices, abs2Matrix is faster (jit functions have overhead).
@numba.vectorize([numba.float64(numba.complex128), numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


@jit(nopython=True)
def abs2Matrix(psi, result):
    for i in range(len(psi)):
        for j in range(len(psi[0])):
            result[i,j] = psi[i,j].real ** 2 + psi[i,j].imag ** 2

@jit(nopython=True)
def abs2MatrixMultiplied(X, Y, func, psi, result, t=0., extra_param=np.array([])):
    for i in range(len(psi)):
        for j in range(len(psi[0])):
            result[i,j] = (psi[i,j].real ** 2 + psi[i,j].imag ** 2) * func(X[i], Y[j], t=t, extra_param=extra_param)

@jit(nopython=True)
def abs2MatrixMultipliedExpected(X, Y, func, psi, t=0., extra_param=np.array([])):
    # Same as integrating with trapezoids, as at boundaries psi is 0
    sumTotal = 0.
    for i in range(len(psi)):
        sumPartial = 0.
        for j in range(len(psi[0])):
            sumPartial += (psi[i,j].real ** 2 + psi[i,j].imag ** 2) * func(X[i], Y[j], t=t, extra_param=extra_param)
        sumTotal += sumPartial
    return sumTotal*(Y[1]-Y[0])*(X[1]-X[0])

@jit(nopython=True)
def set2DMatrix(X, Y, func, psi, t=0., extra_param=np.array([])):
    for i in range(len(X)):
        for j in range(len(Y)):
            psi[i,j] = func(X[i],Y[j], t, extra_param)


@jit(nopython=True)
def euclidNorm(psi, dx, dy):
    # Careful not to add them all at once. Small terms would be ignored
    sumTotal = 0.
    sumPartial = 0.
    for i in range(len(psi)):
        sumPartial = 0.
        for j in range(len(psi[0])):
            sumPartial += abs2(psi[i,j])
        sumTotal += sumPartial
    return sumTotal*dx*dy

# We can just do dx*dy*np.sum(abs2(psi))   using numpy


def generateAsDistribution(pdf):
    """Given a 2D matrix representing a pdf (doesn't need to be normalized, but IT MUST BE POSITIVE)
    a random position is chosen following that pdf
    Process, using marginal probabilities:
     - Generate x following marginal distribution
     - Generate y at given x
    Returns the indices i, j"""
    Xcdf = np.cumsum(np.sum(pdf, axis=1))
    rand = random.random() * Xcdf[-1]
    i = np.searchsorted(Xcdf, rand, side='left') # Already in numpy. Binary search, fast
    Ycdf = np.cumsum(pdf[i])
    rand = random.random() * Ycdf[-1]
    j = np.searchsorted(Ycdf, rand, side='left')

    return i, j

def generateAsDiscreteDistribution(pdf):
    """Given a 2D matrix representing a pdf (doesn't need to be normalized, but IT MUST BE POSITIVE)
    a random position is chosen following that pdf
    Due to discretization this is just generating a discrete distribution, the cell.
    The genereated points are thus discrete. A way to imporove is:
        - Without interpolation: We have chosen cell. Then now chose random point within cell (uniform)
        - Interpolating: We have chosen cell, chose point within cell according to interpolation (more probable nearer cells with higher probabilities)"""
    cumsum = np.cumsum(pdf)
    rand = random.random() * cumsum[-1]  # if it's normalized, cumsum[-1] = ∫∫pdf = 1
    ij = np.searchsorted(cumsum, rand, side='left') # Already in numpy. Binary search, fast

    j = ij%len(pdf[0])
    i = ij//len(pdf[0])

    return i, j


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
        integral = psi[pos,0]
        for it in range(1, size-3, 2):
            integral += 4 * psi[pos,it]
            integral += 2 * psi[pos,it+1]
        if size % 2 == 1:
            integral += 4 * psi[pos,size-2]
            integral += psi[pos,size-1]
        else:
            integral += 4 * psi[pos,size-3]
            integral += 2.5 * psi[pos,size-2]
            integral += 1.5 * psi[pos,size-1]

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

# -> np.complex128
@jit(nopython=True)
def simpson_complex_list1D(dx, psi):
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
def simpson_complex_list2D(dx, dy, psi):
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

@jit(nopython=True)
def trapez_complex_list1D(dx, psi):
    integral = psi[0]/2.
    for it in range(1, len(psi)-1):
        integral += psi[it]
    integral += psi[-1]/2.
    integral = integral * dx
    return integral


@jit(nopython=True)
def trapez_complex_list2D(dx, dy, psi):
    integral = trapez_complex_list1D(dy, psi[0]) / 2.
    for it in range(1, len(psi) - 1):
        integral += trapez_complex_list1D(dy, psi[it])
    integral += trapez_complex_list1D(dy, psi[-1])/2.
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
            result += operator[it2,0] * psi[pos_minus] + operator[it2,1] * psi[pos_plus]

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
            result[pos] = operator[0,0]*psi[pos]
            atBoundary = False
            for i in range(len(pos)):
                if pos[i] == psi.shape[i] - 1 or pos[i] == 0: atBoundary = True
            if not atBoundary:
                for it2 in range(1, len(operator)):  # it2 indicates dimension. 1: x, 2: y, 3: z, etc
                    pos_minus = tuple([pos[i] - (i == it2 - 1) for i in range(len(pos))])
                    pos_plus = tuple([pos[i] + (i == it2 - 1) for i in range(len(pos))])
                    result[pos] += operator[it2,0] * psi[pos_minus] + operator[it2,1] * psi[pos_plus]
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
                    result[pos] += op[it2,0] * psi[pos_minus] + op[it2,1] * psi[pos_plus]
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
def applyOperator2D(X, Y, psi, result, operator, t=0., extra_param=np.array([]), doConjugate = False):
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
                result[i,j] = operator[0,0]*psi[i,j] + operator[1,0] * psi[i-1,j] + operator[1,1] * psi[i+1,j] + \
                                                        operator[2,0] * psi[i,j-1] + operator[2,1] * psi[i,j+1]
        result[0, :] = psi[0, :]*operator[0,0]
        result[len(psi)-1, :] = psi[len(psi)-1, :] * operator[0,0]
        result[:, 0] = psi[:, 0] * operator[0,0]
        result[:, len(psi[0])-1] = psi[:, len(psi[0])-1] * operator[0,0]
    else:
        for i in range(1, len(psi)-1):
            for j in range(1, len(psi[0]-1)):
                result[i,j] = operator[0,0]*psi[i,j] + operator[1,0] * psi[i-1,j] + operator[1,1] * psi[i+1,j] + \
                                                        operator[2,0] * psi[i,j-1] + operator[2,1] * psi[i,j+1]
                result[i,j] = result[i,j] * np.conj(psi[i,j])
        result[0, :] = abs2(psi[0, :])*operator[0,0]
        result[len(psi)-1, :] = abs2(psi[len(psi)-1, :]) * operator[0,0]
        result[:, 0] = abs2(psi[:, 0]) * operator[0,0]
        result[:, len(psi[0])-1] = abs2(psi[:, len(psi[0])-1]) * operator[0,0]


@jit(nopython=True)
def applyOperator2DFunc(X, Y, psi, result, operator, t=0., extra_param=np.array([]), doConjugate=False):
    if len(X) != len(psi) or len(Y) != len(psi[0]): raise ValueError("X and Y coordinates don't match shape of Psi")
    dx = (X[-1] - X[0]) / (len(psi) - 1)
    dy = (Y[-1] - Y[0]) / (len(psi[0]) - 1)
    op = np.array([[1., 0.], [0., 0.], [0., 0.]])
    operator(op, (X[0], Y[0]), (dx, dy), t, extra_param=extra_param, onlyUpdate=False)
    if not doConjugate:
        for i in range(1, len(psi) - 1):
            for j in range(1, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0,0] * psi[i, j] + op[1,0] * psi[i - 1, j] + op[1,1] * psi[i + 1, j] + \
                               op[2,0] * psi[i, j - 1] + op[2,1] * psi[i, j + 1]
        for i in range(len(psi)):
            for j in (0, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0,0] * psi[i, j]
        for j in range(1, len(psi[0]) - 1):
            for i in (0, len(psi) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0,0] * psi[i, j]
        #if doConjugate: result[...] = result[...] * np.conj(psi[...])
    else:
        for i in range(1, len(psi) - 1):
            for j in range(1, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0,0] * psi[i, j] + op[1,0] * psi[i - 1, j] + op[1,1] * psi[i + 1, j] + \
                               op[2,0] * psi[i, j - 1] + op[2,1] * psi[i, j + 1]
                result[i, j] = result[i, j] * np.conj(psi[i, j])
        for i in range(len(psi)):
            for j in (0, len(psi[0]) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0,0] * abs2(psi[i, j])
        for j in range(1, len(psi[0]) - 1):
            for i in (0, len(psi) - 1):
                operator(op, (X[i], Y[j]), (dx, dy), t, extra_param=extra_param)
                result[i, j] = op[0,0] * abs2(psi[i, j])


#@jit(nopython=True)
def expectedValueOperator2D(X, Y, psi, result, operator = None, t = 0., extra_param=np.array([]), doConjugate=True):
    if np.shape(result) != np.shape(psi): raise ValueError("Psi and result must have the same shape")
    if operator is None:            # Calculate total probability, should be 1
        np.conjugate(psi, out=result)
        np.multiply(psi, result, out=result)
        return np.trapz(np.trapz(result)) * (X[-1]-X[0])/(len(psi)-1) * (Y[-1]-Y[0])/(len(psi[0])-1) # Must use same sumation as norm()!
        #return simpson_complex_list2D((X[-1]-X[0])/(len(psi)-1), (Y[-1]-Y[0])/(len(psi[0])-1), result)    #dx, dy
    applyOperator2D(X, Y, psi, result, operator, t=t, extra_param=extra_param, doConjugate=doConjugate)
    if doConjugate: return np.trapz(np.trapz(result)) * (X[-1]-X[0])/(len(X)-1) * (Y[-1]-Y[0])/(len(Y)-1)
    else: return np.trapz(np.trapz(np.multiply(np.conj(psi),result))) * (X[-1]-X[0])/(len(X)-1) * (Y[-1]-Y[0])/(len(Y)-1)
    #return simpson_complex_list2D((X[-1]-X[0])/(len(psi)-1), (Y[-1]-Y[0])/(len(psi[0])-1), result)        #dx, dy


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
        #np.multiply(psi, result, out=result)
        psi *= result
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
    result[:,:] = np.fft.fft2(psi[:,:])*( (X[1]-X[0])*(Y[1]-Y[0]) /(2*np.pi) )#* np.sqrt((len(X)+1)/len(X) * (len(Y)+1)/len(Y)) )
                                                          #, norm="ortho")     # Older 1.10.0- numpy versions may not allow ortho
                                                          # normalization, but it ends up normalized with respect to dx and dy! not px and py
                                                          # Equal to dividing by sqrt(lenX * lenY). It's better to just do it
    result[:,:] = np.fft.fftshift(result[:,:])            # Center freq (by default 0 freq is on bottom left)
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
def kineticEnergy(op, X, dx, t=0, extra_param=np.array([]), dir=-1, onlyUpdate=True):
    # -h^2/2m ∇^2
    # if dir != -1,  dir indicates only one direction dir=1: x, dir=2: y, etc
    global hred, M
    if onlyUpdate: return # The operator doesn't need to be updated, as dx and dy constant -> operator constant
    if dir == -1:
        laplac = 0.
        for delta in dx: laplac += 1./delta**2
        op[0,0] = hred*hred/M * laplac
        for it in range(1,len(op)):
            op[it,0] = -hred*hred/(2*M) / dx[it-1]**2
            op[it,1] = op[it,0]
    else:
        op[0,0] = +hred*hred/(M)/(dx[dir])**2
        for it in range(1, len(op)):
            op[it,0] = -hred * hred / (2 * M) / dx[it] ** 2
            op[it,1] = op[it,0]
        op[dir,0] = -hred * hred / (2 * M) / dx[dir] ** 2
        op[dir,1] = op[dir,0]


def totalEnergyOpGenerator(potential, preCalcLaplac):
    @jit
    def totalEnergy(op, X, dx, t=0, extra_param=np.array([]), dir=-1, onlyUpdate=True):
        # -h^2/2m ∇^2 + V(X,t)
        # if dir != -1,  dir indicates only one direction dir=1: x, dir=2: y, etc
        global hred, M
        if onlyUpdate:  # Only potential term changes
            op[0, 0] = hred * hred / M * preCalcLaplac + potential(X[0], X[1], t=t, extra_param=extra_param)
        if dir == -1:
            laplac = 0.
            for delta in dx: laplac += 1./delta**2
            op[0,0] = hred*hred/M * laplac + potential(X[0], X[1], t=t, extra_param=extra_param)
            for it in range(1,len(op)):
                op[it,0] = -hred*hred/(2*M) / dx[it-1]**2
                op[it,1] = op[it,0]
        else:
            op[0,0] = +hred*hred/(M)/(dx[dir])**2 + potential(X[0], X[1], t=t, extra_param=extra_param)
            for it in range(1, len(op)):
                op[it,0] = -hred * hred / (2 * M) / dx[it] ** 2
                op[it,1] = op[it,0]
            op[dir,0] = -hred * hred / (2 * M) / dx[dir] ** 2
            op[dir,1] = op[dir,0]

    return totalEnergy


#@jit
def potentialEnergyGenerator(potential):
    def potentialEnergy(op, X, dx, t=0, dir=-1, onlyUpdate=True):
        # V(x,y,t,) Psi
        # if dir != -1,  dir indicates only one direction dir=1: x, dir=2: y, etc. Doesn't matter in this case
        global hred, M
        if onlyUpdate:
            op[0,0] = potential(*X, t)
        else:
            op[0,0] = potential(*X, t)
            for i in range(1, len(op)):
                op[i,0] = 0.
                op[i,1] = 0.
    return potentialEnergy


@jit(nopython=True)
def tridiag(gamma, trid, bVec, xVec):
    """
    Solves tridiagonal system. Tridiag is a matrix of the form (A- = trid[0,:], A = trid[1,:], A+ = trid[2,:]
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
    #if trid[1,0] == 0.: raise ValueError("matriu tridiagonal invàlida pel mètode tridiagonal, 0 a la diagonal")
    beta = trid[1,0]
    xVec[0]=bVec[0]/beta
    for j in range(1,len(bVec)):
        gamma[j]=trid[2,j-1]/beta
        beta=trid[1,j]-trid[0,j]*gamma[j]
        #if beta == 0: raise ValueError("Matriu singular")     # It is better to check == 0 with tolerance
        xVec[j]=(bVec[j]-trid[0,j]*xVec[j-1])/beta

    for j in range(len(bVec)-1):
        xVec[len(bVec)-2-j] = xVec[len(bVec)-2-j] - gamma[len(bVec)-j-1]*xVec[len(bVec)-j-1]

@jit(nopython=True)
def diagonals(gamma, diagonals, bVec, xVec):
    """
    Solves tridiagonal band system.
    Sadly it's really really really computation heavy, but most importantly memory heavy

        matrix:     D-m(0)  ....  D-1(0) |D 0(0) D+1(0)  ....  D+m(0)  ....   ....   ....  |
                           D-m(1)  ....  |D-1(1) D 0(1) D+1(1)  ....  D+m(1)  ....   ....  |    * PSI    =   bVec
                                         | ....   ....   ....   ....   ....   ....   ....  |
                                         |D-m(m)  ....  D-1(m) D 0(m) D+1(m)  ....  D+m(m) |
                                         | ....   ....   ....   ....   ....   ....   ....  |
                                         | ....   ....  D-m(n)  ....   ....  D-1(n) D 0(n) | D+1(n)  ....  D+m(n)
    :param gamma: numpy vector of size n+1. Must be already created to avoid allocating memory every time tridiag is called
    :param trid: tridiagonal matrix
    :param bVec: Right side vector
    :param xVec: Unknown vector to find
    :return:
    """
    #if trid[1,0] == 0.: raise ValueError("matriu tridiagonal invàlida pel mètode tridiagonal, 0 a la diagonal")
    beta = trid[1,0]
    xVec[0]=bVec[0]/beta
    for j in range(1,len(bVec)):
        gamma[j]=trid[2,j-1]/beta
        beta=trid[1,j]-trid[0,j]*gamma[j]
        #if beta == 0: raise ValueError("Matriu singular")     # It is better to check == 0 with tolerance
        xVec[j]=(bVec[j]-trid[0,j]*xVec[j-1])/beta

    for j in range(len(bVec)-1):
        xVec[len(bVec)-2-j] = xVec[len(bVec)-2-j] - gamma[len(bVec)-j-1]*xVec[len(bVec)-j-1]


@jit(nopython=True)
def tridiagReal(gamma, trid, bVec, xVec):
    """tridiag Analog for real matrices"""
    #if trid[1,0] == 0.: raise ValueError("matriu tridiagonal invàlida pel mètode tridiagonal, 0 a la diagonal")
    beta = trid[1,0]
    xVec[0]=bVec[0]/beta
    for j in range(1,len(bVec)):
        gamma[j]=trid[2,j-1]/beta
        beta=trid[1,j]-trid[0,j]*gamma[j]
        #if beta == 0: raise ValueError("Matriu singular")     # It is better to check == 0 with tolerance
        xVec[j]=(bVec[j]-trid[0,j]*xVec[j-1])/beta

    for j in range(len(bVec)-1):
        xVec[len(bVec)-2-j] = xVec[len(bVec)-2-j] - gamma[len(bVec)-j-1]*xVec[len(bVec)-j-1]


@jit(nopython=True)
def crankNicolson2DSchrodingerStepLegacy(X, Y, t, dt, potential, psi, psiTemp, extra_param=np.array([])):
    """ Solves System:  i h_red d/dt psi = [-h_red^2 /2 (d^2/dx^2 + d^2/dy^2)  + V ] psi
    Step 0: Psi(t)  ->   Step 1: psiTemp = Psi(t+dt/2) ->  Step 2: psi = Psi(t+dt)"""
    """ FUNNY THING: THE METHOD IS CONCEPTUALLY WRONG, BUT AT THE SAME TIME THE ERROR RESOLVES ITSELF, SO THIS IS CORRECT """

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
    dx = (X[-1]-X[0])/(len(X)-1)
    dy = (Y[-1]-Y[0])/(len(Y)-1)
    rx = 1j*dt*hred/(2*M*dx**2)
    cent_rx = 2*(1+rx)
    ry = 1j*dt*hred/(2*M*dy**2)
    cent_ry = 2*(1+ry)

    # We set tridiagonal matrices. The main diagonal depends on the potential, which depends on position, so it's done later
    tridX[0,:] = -rx
    #tridX[1,:] = 2*(1+rx) + 1j * dt * V(x,y)
    tridX[2,:] = -rx
    tridY[0,:] = -ry
    # tridY[1,:] = 2*(1+ry) + 1j * dt * V(x,y)
    tridY[2,:] = -ry

    # First half step. At t we only do derivatives in y direction. At t+dt/2 in x direction. Thus the name, alternating
    # direction implicit method. Important to remember that at boundaries wave is fixed (infinite potential wall)
    for ity in range(1, len(psi[0])-1):
        for itx in range(1, len(psi)-1):
            tridX[1,itx-1] = cent_rx + 1j * dt/2 /hred * potential(X[itx], Y[ity], t+dt/2, extra_param=extra_param)
            tempBX[itx-1] = (psi[itx,ity-1] + psi[itx,ity+1])*ry \
                            + (2*(1-ry) - 1j * dt/2 /hred * potential(X[itx], Y[ity], t, extra_param=extra_param)) * \
                            psi[itx,ity]
        #tempBX[1] += rx*psi[0,ity]                                At boundary psi should be 0!
        #tempBX[len(psi)-1] += rx * psi[len(psi)-1,ity]
        tridiag(gamma, tridX, tempBX, psiTemp[1:-1,ity])
    psiTemp[:,0] = psi[:,0]
    psiTemp[:, len(psi[0]) - 1] = psi[:,len(psi[0]) - 1]
    psiTemp[0,:] = psi[0,:]
    psiTemp[len(psi)-1,:] = psi[len(psi)-1,:]

    # We do the final half step. Again, at t+dt/2 we only do derivatives in x direction, but now at t+dt in x direction
    for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            tridY[1,ity-1] = cent_ry + 1j * dt/2 /hred * potential(X[itx], Y[ity], t+dt, extra_param=extra_param)
            tempBY[ity-1] = (psiTemp[itx-1,ity] + psiTemp[itx+1,ity])*rx \
                            + (2*(1-rx) - 1j * dt/2 /hred * potential(X[itx], Y[ity], t+dt/2, extra_param=extra_param))\
                              *psiTemp[itx,ity]
        #tempBY[1] += ry * psi[itx,0]                                  At boundary psi should be 0!
        #tempBY[len(psi[0]) - 1] += rx * psi[itx,len(psi[0]) - 1]
        tridiag(gamma, tridY, tempBY, psi[itx,1:-1])


@jit(nopython=True)
def crankNicolson2DSchrodingerStepVaryingPotential(X, Y, t, dt, potential, psi, psiTemp, extra_param=np.array([])):
    """ Solves System:  i h_red d/dt psi = [-h_red^2 /2 (d^2/dx^2 + d^2/dy^2)  + V ] psi
    Step 0: Psi(t)  ->   Step 1: psiTemp = Psi(t+dt/2) ->  Step 2: psi = Psi(t+dt)"""
    """ THIS METHOD DOES THE EXACT SAME AS LEGACY, BUT HERE IT'S CONCEPTUALLY RIGHT. Missing dt/2 in rx and ry definition
    which was compensated later by missing 2* multiplying rx and ry"""
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
    dx = (X[-1]-X[0])/(len(X)-1)
    dy = (Y[-1]-Y[0])/(len(Y)-1)
    rx = 1j*dt/2*hred/(2*M*dx**2)
    cent_rx = 2*(1+2*rx)  # *2d_x ~ d_x + d_y
    ry = 1j*dt/2*hred/(2*M*dy**2)
    cent_ry = 2*(1+2*ry)  # *2d_y ~ d_x + d_y

    # We set tridiagonal matrices. The main diagonal depends on the potential, which depends on position, so it's done later
    tridX[0,:] = -2*rx
    #tridX[1,:] = 2*(1+2*rx) + 1j * dt/2 * V(x,y)
    tridX[2,:] = -2*rx
    tridY[0,:] = -2*ry
    # tridY[1,:] = 2*(1+2*ry) + 1j * dt/2 * V(x,y)
    tridY[2,:] = -2*ry

    # First half step. At t we only do derivatives in y direction. At t+dt/2 in x direction. Thus the name, alternating
    # direction implicit method. Important to remember that at boundaries wave is fixed (infinite potential wall)
    for ity in range(1, len(psi[0])-1):
        for itx in range(1, len(psi)-1):
            tridX[1,itx-1] = cent_rx + 1j * dt/2/ hred * potential(X[itx], Y[ity], t+dt/2, extra_param=extra_param)
            tempBX[itx-1] = (psi[itx,ity-1] + psi[itx,ity+1])*2*ry \
                            + (2*(1-2*ry) - 1j * dt/2 /hred * potential(X[itx], Y[ity], t, extra_param=extra_param))*psi[itx,ity]
        #tempBX[1] += rx*psi[0,ity]                                At boundary psi should be 0!
        #tempBX[len(psi)-1] += rx * psi[len(psi)-1,ity]
        tridiag(gamma, tridX, tempBX, psiTemp[1:-1,ity])
    psiTemp[:,0] = psi[:,0]
    psiTemp[:, len(psi[0]) - 1] = psi[:,len(psi[0]) - 1]
    psiTemp[0,:] = psi[0,:]
    psiTemp[len(psi)-1,:] = psi[len(psi)-1,:]

    # We do the final half step. Again, at t+dt/2 we only do derivatives in x direction, but now at t+dt in x direction
    for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            tridY[1,ity-1] = cent_ry + 1j * dt/2 / hred * potential(X[itx], Y[ity], t+dt, extra_param=extra_param)
            tempBY[ity-1] = (psiTemp[itx-1,ity] + psiTemp[itx+1,ity])*2*rx \
                            + (2*(1-2*rx) - 1j * dt/2 / hred * potential(X[itx], Y[ity], t+dt/2, extra_param=extra_param))\
                              *psiTemp[itx,ity]
        #tempBY[1] += ry * psi[itx,0]                                  At boundary psi should be 0!
        #tempBY[len(psi[0]) - 1] += rx * psi[itx,len(psi[0]) - 1]
        tridiag(gamma, tridY, tempBY, psi[itx,1:-1])

@jit(nopython=True)
def crankNicolson2DSchrodingerStepImaginary(X, Y, t, dt, potential, psi, psiTemp, extra_param=np.array([])):
    """ Solves System:  i h_red d/dt psi = [-h_red^2 /2 (d^2/dx^2 + d^2/dy^2)  + V ] psi
    Step 0: Psi(t)  ->   Step 1: psiTemp = Psi(t+dt/2) ->  Step 2: psi = Psi(t+dt)"""
    """ Same as VaryingPotential but Potential is taken only at t, as it's assumed constnat. Actually
        to be used for imaginary time displacements, to find eigenstates. dt is imaginary component (real)"""
    global hred, M

    # To make the code even faster, all of these temporary arrays could be created already. So no step needs to be taken
    # Declare all necessary arrays. Trying to minimize need to allocate memory
    tempBX = np.empty(len(psi)-2, dtype=np.complex128)
    tempBY = np.empty(len(psi[0])-2, dtype=np.complex128)

    tridX = np.empty((3,len(psi)-2), dtype=np.float64)
    tridY = np.empty((3,len(psi[0])-2), dtype=np.float64)

    # Gamma is used for tridiag, it is allocated here to avoid repeated unnecessary memory allocation
    gamma = np.empty((max(len(psi)-1, len(psi[0])-1),), dtype=np.float64)

    # Parameters for calcualtions
    dx = (X[-1]-X[0])/(len(X)-1)
    dy = (Y[-1]-Y[0])/(len(Y)-1)
    rx = -dt*hred/(2*M*dx**2)  #    1j * dt = - |dt|
    cent_rx = 2*(1+rx)  # *2d_x ~ d_x + d_y
    ry = -dt*hred/(2*M*dy**2)
    cent_ry = 2*(1+ry)  # *2d_y ~ d_x + d_y

    # We set tridiagonal matrices. The main diagonal depends on the potential, which depends on position, so it's done later
    tridX[0,:] = -rx
    #tridX[1,:] = 2*(1+rx) - dt/2 * V(x,y)
    tridX[2,:] = -rx
    tridY[0,:] = -ry
    # tridY[1,:] = 2*(1+ry) - dt/2 * V(x,y)
    tridY[2,:] = -ry

    # First half step. At t we only do derivatives in y direction. At t+dt/2 in x direction. Thus the name, alternating
    # direction implicit method. Important to remember that at boundaries wave is fixed (infinite potential wall)
    for ity in range(1, len(psi[0])-1):
        for itx in range(1, len(psi)-1):
            tridX[1,itx-1] = cent_rx - dt/2 / hred * potential(X[itx], Y[ity], t, extra_param=extra_param)
            tempBX[itx-1] = (psi[itx,ity-1] + psi[itx,ity+1])*ry \
                            + (2*(1-ry) + dt/2 /hred * potential(X[itx], Y[ity], t, extra_param=extra_param))*psi[itx,ity]
        #tempBX[1] += rx*psi[0,ity]                                At boundary psi should be 0!
        #tempBX[len(psi)-1] += rx * psi[len(psi)-1,ity]
        tridiagReal(gamma, tridX, tempBX, psiTemp[1:-1,ity])
    psiTemp[:,0] = psi[:,0]
    psiTemp[:, len(psi[0]) - 1] = psi[:,len(psi[0]) - 1]
    psiTemp[0,:] = psi[0,:]
    psiTemp[len(psi)-1,:] = psi[len(psi)-1,:]

    # We do the final half step. Again, at t+dt/2 we only do derivatives in x direction, but now at t+dt in x direction
    for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            tridY[1,ity-1] = cent_ry - dt/2 / hred * potential(X[itx], Y[ity], t, extra_param=extra_param)
            tempBY[ity-1] = (psiTemp[itx-1,ity] + psiTemp[itx+1,ity])*rx \
                            + (2*(1-rx) + dt/2 / hred * potential(X[itx], Y[ity], t, extra_param=extra_param))\
                              *psiTemp[itx,ity]
        #tempBY[1] += ry * psi[itx,0]                                  At boundary psi should be 0!
        #tempBY[len(psi[0]) - 1] += rx * psi[itx,len(psi[0]) - 1]
        tridiagReal(gamma, tridY, tempBY, psi[itx,1:-1])


@jit(nopython=True)
def crankNicolson2DSchrodingerStepFastest(X, Y, t, dt, potential, psi, psiTemp, psiMod, extra_param=np.array([])):
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
    dx = (X[-1]-X[0])/(len(X)-1)
    dy = (Y[-1]-Y[0])/(len(Y)-1)
    rx = 1j*dt*hred/(2*M*dx**2)
    cent_rx = 2*(1+rx)
    ry = 1j*dt*hred/(2*M*dy**2)
    cent_ry = 2*(1+ry)

    # We set tridiagonal matrices. The main diagonal depends on the potential, which depends on position, so it's done later
    tridX[0,:] = -rx
    #tridX[1,:] = 2*(1+rx) + 1j * dt * V(x,y)
    tridX[2,:] = -rx
    tridY[0,:] = -ry
    # tridY[1,:] = 2*(1+ry)
    tridY[2,:] = -ry

    set2DMatrix(X, Y, potential, psiMod, t=t+dt/2, extra_param=extra_param)

    # First half step. At t we only do derivatives in y direction. At t+dt/2 in x direction. Thus the name, alternating
    # direction implicit method. Important to remember that at boundaries wave is fixed (infinite potential wall)
    for ity in range(1, len(psi[0])-1):
        for itx in range(1, len(psi)-1):
            tridX[1,itx-1] = cent_rx + 1j * dt/hred * psiMod[itx, ity]
            tempBX[itx-1] = (psi[itx,ity-1] + psi[itx,ity+1])*ry \
                            + (2*(1-ry))*psi[itx,ity]
        #tempBX[1] += rx*psi[0,ity]                                At boundary psi should be 0!
        #tempBX[len(psi)-1] += rx * psi[len(psi)-1,ity]
        tridiag(gamma, tridX, tempBX, psiTemp[1:-1,ity])
    psiTemp[:,0] = psi[:,0]
    psiTemp[:, len(psi[0]) - 1] = psi[:,len(psi[0]) - 1]
    psiTemp[0,:] = psi[0,:]
    psiTemp[len(psi)-1,:] = psi[len(psi)-1,:]

    # We do the final half step. Again, at t+dt/2 we only do derivatives in x direction, but now at t+dt in x direction
    for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            tridY[1,ity-1] = cent_ry
            tempBY[ity-1] = (psiTemp[itx-1,ity] + psiTemp[itx+1,ity])*rx \
                            + (2*(1-rx) - 1j * dt/hred * psiMod[itx,ity])\
                              *psiTemp[itx,ity]
        #tempBY[1] += ry * psi[itx,0]                                  At boundary psi should be 0!
        #tempBY[len(psi[0]) - 1] += rx * psi[itx,len(psi[0]) - 1]
        tridiag(gamma, tridY, tempBY, psi[itx,1:-1])


@njit
def crankNicolson2DSchrodingerExactSetup(X, Y, t, dt, potential, psi, psiTemp, psiMod, diagonals, extra_param=np.array([])):
    # Parameters for calculations
    dx = (X[-1] - X[0]) / (len(X) - 1)
    dy = (Y[-1] - Y[0]) / (len(Y) - 1)
    rx = 1j * dt * hred / (2 * M * dx ** 2)
    ry = 1j * dt * hred / (2 * M * dy ** 2)

    N = len(Y)

    cent_r = 2 * (1 + ry + rx)

    set2DMatrix(X, Y, potential, psiMod, t=t + dt / 2, extra_param=extra_param)

    diagonals[0][:] = -rx
    diagonals[-1][:] = -rx
    diagonals[N-1][:] = -ry
    diagonals[N+1][:] = -ry


    for i in range(len(X)):            # All extreme value don't change.
        diagonals[N][0 + i*N] = 1.
        diagonals[N][N-1 + i * N] = 1.

        diagonals[N-1][0 + (i-1)*N + 1] = 0.
        diagonals[N+1][N-1 + (i+1) * N - 1] = 0.

    for j in range(len(Y)):
        diagonals[N][j + 0*N] = 1.
        diagonals[N][j + (len(X)-1)*N] = 1.

        diagonals[ 0][j + (0) * N + N] = 0.
        diagonals[-1][j + (len(X)-1) * N - N] = 0.

    # We actually still should remove some things:
    # - terms corresponding to y derivatives for (constant) x borders
    # - terms corresponding to x derivatives for (constant) y borders
    # But precisely because the borders have psi=0, they don't matter
    for i in range(1, len(X)-1):
        diagonals[0][0 + i * N + N] = 0.
        diagonals[-1][N-1 + i * N - N] = 0.

    for j in range(1, len(Y)-1):
        diagonals[N - 1][j + (0) * N + 1] = 0.
        diagonals[N + 1][j + (len(X)-1) * N - 1] = 0.

    psiTemp[:,0] = 0.
    psiTemp[:,-1] = 0.
    psiTemp[0, :] = 0.
    psiTemp[-1, :] = 0.
    for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            diagonals[N,ity + itx*N] = cent_r + 1j * dt/hred * psiMod[itx, ity]
            psiTemp[itx, ity] = (psi[itx-1,ity] + psi[itx+1,ity])*rx \
                            + (psi[itx,ity-1] + psi[itx,ity+1])*ry \
                            + (2*(1-rx-ry) - 1j * dt/hred * psiMod[itx,ity]) * psi[itx,ity]


def crankNicolson2DSchrodingerStepExact(X, Y, t, dt, potential, psi, psiTemp, psiMod, extra_param=np.array([])):
    """We solve the CrankNicolson step without approximating further,
    that is, we solve the system as is with Gaussian Elimination"""
    # We can use solver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html#scipy.linalg.solveh_banded

    diagonals = np.zeros((2*len(Y)+1, (len(X))*(len(Y))), dtype=np.complex128)
    crankNicolson2DSchrodingerExactSetup(X, Y, t, dt, potential, psi, psiTemp, psiMod, diagonals, extra_param)

    psi[:,:] = solve_banded((len(Y), len(Y)), diagonals, psiTemp.ravel(), overwrite_ab=True, overwrite_b=True, check_finite=True).reshape((len(X), len(Y)))


#@njit  # WE can't use numba with fourier transforms
def crankNicolson2DSchrodingerStepFourier(X, Y, t, dt, potential, psi, psiMom, psiMod, extra_param=np.array([])):
    """We solve the CrankNicolson step factorizing the operators and switching to momentum space for momentum oeprators"""
    # We can use solver: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solveh_banded.html#scipy.linalg.solveh_banded
    set2DMatrix(X, Y, potential, psiMod, t=t + dt / 2, extra_param=extra_param)
    dx = (X[-1] - X[0]) / (len(X) - 1)
    dy = (Y[-1] - Y[0]) / (len(Y) - 1)
    Nx = len(X)
    Ny = len(Y)
    """for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            psiTemp[itx, ity] = psi[itx, ity] * (1-)"""
    psi *= (1 - 1j*dt/(2*hred)/2 * psiMod)

    Px2, Py2 = np.square(2 * hred * np.pi * (np.fft.fftfreq(Nx, dx))), np.square(2 * hred * np.pi * np.fft.fftfreq(Ny, dy))

    psiMom[:,:] = np.fft.fft2(psi)#dstn(psi)#

    Px2m, Py2m = np.meshgrid(Px2, Py2, copy=False, indexing='ij')

    # Pxm*Pym = grid with P^2 at each point
    psiMom *= (1 - 1j*dt/(2*hred) / (2*M) * (Px2m+Py2m))

    psi[:,:] = np.fft.ifft2(psiMom)#idstn(psiMom)#

    psi *= (1 - 1j*dt/(2*hred)/2 * psiMod)/(1 + 1j*dt/(2*hred)/2 * psiMod)

    psiMom[:, :] = np.fft.fft2(psi)#dstn(psi)#

    psiMom /= (1 + 1j*dt/(2*hred) / (2*M) * (Px2m+Py2m))

    psi[:, :] = np.fft.ifft2(psiMom)#idstn(psiMom)#

    psi /= (1 + 1j*dt/(2*hred)/2 * psiMod)



def crankNicolson2DSchrodingerStepClosedBoxEigen(X, Y, t, dt, potential, psi, psiEigen, psiMod, extra_param=np.array([])):
    """We solve the CrankNicolson step factorizing the operators and switching to momentum space for momentum oeprators"""
    set2DMatrix(X, Y, potential, psiMod, t=t + dt / 2, extra_param=extra_param)

    Nx = len(X)
    Ny = len(Y)
    """for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            psiTemp[itx, ity] = psi[itx, ity] * (1-)"""
    psi *= (1 - 1j*dt/(2*hred) * psiMod)

    Px2, Py2 = np.array([n*n for n in range(1, Nx+1 -2)])*np.pi**2 * hred**2/(2*M*(X[-1]-X[0])**2), np.array([n*n for n in range(1, Ny+1 -2)])*np.pi**2 * hred**2/(2*M*(Y[-1]-Y[0])**2)


    psiEigen[:,:] = dstn(psi[1:-1,1:-1], type=1)

    Px2m, Py2m = np.meshgrid(Px2, Py2, copy=False, indexing='ij')

    # Pxm*Pym = grid with P^2 at each point
    psiEigen *= (1 - 1j*dt/(2*hred) * (Px2m+Py2m))/(1 + 1j*dt/(2*hred) * (Px2m+Py2m))

    psi[1:-1, 1:-1] = idstn(psiEigen, type=1)

    psi /= (1 + 1j*dt/(2*hred) * psiMod)



@jit
def potential0(x, y, t=0., extra_param=np.array([])):
    return 0.


def func1(x, y, t=0., extra_param=np.array([])):
    return 1.

@jit
def potentialBarrier(x, y, t, extra_param):
    return np.exp(-(x ** 2) / 0.1) * 5 / np.sqrt(0.1 * np.pi)

@jit
def potentialBarrierYCustom(x, y, t, extra_param):
    return extra_param[1] * np.exp(-((x-extra_param[0]) ** 2) / 0.1) / np.sqrt(0.1 * np.pi) * (0 if abs(y) < extra_param[2] else 1)

@jit
def potentialGravity(x, y, t, extra_param):
    return y*1

@jit
def potentialClosing(x, y, t, extra_param):
    global L
    # Heavyside. Analytic approximation: 1/(1 + e^-2kr). Larger k, better approximation
    r2 = x*x + y*y
    k = 10
    return 100*1/(1+np.exp(-2*k*( r2-(L/2)**2 + (L/2.02)**2 * np.sin(0.5*t)  ) ))
    #if r2 > (L/2.)**2 - (L/2.02)**2 * np.sin(0.5*t): return 1000.
    #return 0.


######################################################
# OPEN AND CLOSE RADIAL WALLS

@jit
def potentialClosingManual(x, y, t, extra_param):
    global L
    # extra_param = [radius, rateOfChangeRadius, timeofLastChange]
    r = np.sqrt(x*x + y*y)
    k = 4.
    return 100*1/(1+np.exp(-2*k*( r- (extra_param[0] + extra_param[1]*(t-extra_param[2]))  ) ))

@jit
def potentialWellMoving(x, y, t, extra_param):
    k = 10
    return 100 * 1 / (1 + np.exp(-2 * k * ((x-t)**2 + (y-t)**2 - (L / 3) ** 2)))

#@jit([numba.float64(numba.float64, numba.float64, numba.float64)], nopython=True)
@njit
def potentialWell(x, y, t, extra_param):
    k = 10
    return 100 * 1 / (1 + np.exp(-2 * k * ((x)**2 + (y)**2 - (L / 3) ** 2)))

@njit
def potentialHarmonicWell(x, y, t, extra_param=np.array([])):
    global kHarm
    res = 1/2. * kHarm*( x**2 + y**2)
    if res > 100.: return 100
    return res

@njit
def potentialHarmonic(x, y, t, extra_param=np.array([1.])):
    return 1/2. * extra_param[0] * (x*x + y*y)


# These also work as px and py
@njit
def xFunc(x, y, t, extra_param=np.array([])):
    return x

@njit
def yFunc(x, y, t, extra_param=np.array([])):
    return y

@njit
def kineticEnergyMomentum(px, py, t, extra_param=np.array([])):
    return (px**2 + py**2)/(2*M)


# extra_param[0] holds the mean
@njit
def xVarFunc(x, y, t, extra_param=np.array([])):
    return (x-extra_param[0])*(x-extra_param[0])

@njit
def yVarFunc(x, y, t, extra_param=np.array([])):
    return (y-extra_param[0])*(y-extra_param[0])

def gaussian00(x,y,t=0., extra_param=np.array([])):
    return np.exp(-(x*x + y*y)/2) / np.sqrt(2*np.pi)

def gaussianPacket(x, x0, sigma, p0, extra_param=np.array([])):
    global hred
    return 1./(2*np.pi*sigma**2)**(0.25) * np.exp(-1./4. * ((x-x0)/sigma)**2) * np.exp(1j/hred * p0*(x))#-x0/2)) ??? Apunts mec quantica


# GENERATOR FOR GAUSSIAN INITIAL STATES
def gaussian2D(x0, sigmax, px, y0, sigmay, py):
    def result(x, y, t=0., extra_param=np.array([])):
        return gaussianPacket(x, x0, sigmax, px) * gaussianPacket(y, y0, sigmay, py)
    return result


#example1
def inicial(x, p_0, t=0., extra_param=np.array([])):
    return np.exp(-(x - 7) ** 2 / 4) / (2 * np.pi) ** (0.25) * np.exp(- 1j/hred * p_0 * x)

#example2
def inicial2D(x, y, t=0., extra_param=np.array([])):
    global p_0
    return inicial(x, p_0)*inicial(y, 0.5*p_0)


def eigenvectorsHarmonic1D(x, x0, n, k):
    #not normalized
    x = x * np.sqrt(np.sqrt(M*k)/hred)
    return np.power(M*k, 1/8.)/np.sqrt(2**n * np.math.factorial(n) * np.sqrt(hred*np.pi)) * np.exp(-(x-x0)**2/2.)*\
           np.polynomial.hermite.hermval(x, [0]*(n-1) + [1])

# GENERATOR FOR HARMONIC OSCILLATOR EIGENVECTORS
def eigenvectorHarmonic2DGenerator(x0, nx, y0, ny, k):
    def result(x, y, t=0., extra_param=np.array([])):
        return eigenvectorsHarmonic1D(x, x0, nx, k)*eigenvectorsHarmonic1D(y, y0, ny, k)
    return result



@jit #@numba.vectorize([numba.complex128(numba.complex128, numba.complex128), numba.float64(numba.float64, numba.float64)])
def innerProduct2D(a, b, dx, dy):
    # returns <a|b>,   orthogonal basis, norm dx * dy
    return np.conj(a).ravel().dot(b.ravel()) * dx * dy

def interpolate2D(psiNew, x0, xf, y0, yf, psiOld, x0Old, xfOld, y0Old, yfOld):
    """
    Interpolate old system into new system.
    (iI, jI+1)   ·---o-------·      (iI+1, jI+1)
                     |
                     O
                     |
                     |
    (iI, jI)     ·---o-------·      (iI+1, jI)
    Example, we first interpolate in the x direction obtaining both o
    and then finally find the interpolation at point O by interpolating in y direction
    """
    # i: x0 + i dx  -> iInterp = (x-x0Old)/(xfOld-x0Old)
    dx = (xf - x0) / (len(psiNew   ) - 1)
    dy = (yf - y0) / (len(psiNew[0]) - 1)
    dxOld = (xfOld - x0Old) / (len(psiOld   ) - 1)
    dyOld = (yfOld - y0Old) / (len(psiOld[0]) - 1)
    for i in range(len(psiNew)):
        iInterp = (x0 + i*dx - x0Old)/dxOld
        for j in range(len(psiNew[0])):
            jInterp = (y0 + j * dy - x0Old) / dyOld
            if iInterp < 0 or len(psiOld)-1 <= iInterp or jInterp < 0 or len(psiOld[0])-1 <= jInterp:
                psiNew[i,j] = 0.
            else:
                iI = int(iInterp)
                hi = iInterp - iI
                jI = int(jInterp)
                hj = jInterp - jI
                interpX0 = (1-hi)*psiOld[iI, jI    ] + hi * psiOld[iI + 1, jI    ] # Corresponds to lower o
                interpX1 = (1-hi)*psiOld[iI, jI + 1] + hi * psiOld[iI + 1, jI + 1] # Corresponds to upper o
                psiNew[i,j] = (1-hj) * interpX0 + hj * interpX1


# https://numba.pydata.org/numba-doc/latest/user/jitclass.html
class QuantumSystem2D:
    # there are Nx+1 points x0, x1, ..., XNx
    def __init__(self, Nx=200, Ny=200, x0=-10., y0=-10., xf=+10., yf=+10.,
                 initState = gaussian00, t = 0., potential = potential0, extra_param=np.array([1.]),
                 x0Old = -10., xfOld=10., y0Old=-10., yfOld=10., mass=M, step='fastest', renormStep=False, customOperator=None):
                      # These are used for interpolating
        global M
        M = mass
        self.mass = mass
        self.Nx, self.Ny = Nx, Ny
        self.x0, self.y0 = x0, y0
        self.xf, self.yf = xf, yf
        self.dx, self.dy = (xf-x0)/(Nx), (yf-y0)/(Ny)
        self.X, self.Y = np.linspace(x0, xf, Nx+1, dtype=np.float64), np.linspace(y0, yf, Ny+1, dtype=np.float64)
        self.Px, self.Py = 2*hred*np.pi*np.sort(np.fft.fftfreq(Nx+1,self.dx)), 2*hred*np.pi*np.sort(np.fft.fftfreq(Ny+1,self.dy))
                           # Maybe in general it needs to be multiplied by hred
        self.t = t  # By default t=0.
        self.extra_param = extra_param.view()

        self.step = step
        self.renormStep = renormStep
        self.customOperator = customOperator

        self.psi = np.empty((Nx+1, Ny+1), dtype=np.complex128)  # Will hold psi
        self.psiMod = np.empty((Nx+1, Ny+1), dtype=np.float64)  # Will hold |psi|^2
        self.psiCopy = np.copy(self.psi)                    # Will hold necessary intermediate step matrices
                                                            # Total memory cost ~2.5 psi ~= 2.5*Nx*Ny*128 bits
        self.psiMom = np.copy(self.psi)                     # One extra for momentum, not really necessary. But useful for uncertainty
        self.Xmesh, self.Ymesh = np.meshgrid(self.X, self.Y, copy=False, indexing='ij') # Copy True or False? Causes some problems sometimes???
        self.Pxmesh, self.Pymesh = np.meshgrid(self.Px, self.Py, copy=False, indexing='ij')

        self.psiEigen = np.zeros((Nx-1,Ny-1), dtype=np.complex128)
        self.Kx, self.Ky = np.array([n*n for n in range(1, Nx)])*np.pi**2 * hred**2/(2*M*(xf-x0)**2), np.array([n*n for n in range(1, Ny)])*np.pi**2 * hred**2/(2*M*(yf-y0)**2)
                      # Kinetic energy Eigenvalues for closed box. "only for interior points"
        self.Kxmesh, self.Kymesh = np.meshgrid(self.Kx, self.Ky, copy=False, indexing='ij')

        if callable(initState):
            self.psi[:,:] = initState(self.Xmesh, self.Ymesh, t = self.t, extra_param=self.extra_param)
            #set2DMatrix(self.X, self.Y, initState, self.psi, t=t)
            # Potentially faster, but needs jit, not as flexible,
            # and regardless it's a one time thing
        else:
            if initState.shape == self.psi.shape and x0==x0Old and y0==y0Old and xf==xfOld and yf==yfOld:
                self.psi[:,:] = initState[:,:]
            else:
                interpolate2D(self.psi, self.x0, self.xf, self.y0, self.yf,
                              initState, x0Old, xfOld, y0Old, yfOld)
        self.psi[:,0] = 0.
        self.psi[:,Ny] = 0.
        self.psi[0,:] = 0.
        self.psi[Nx,:] = 0.
        #self.psi[:,:] = self.psi[:,:]/np.sqrt(self.norm()) # Initial state gets automatically normalized
        self.renorm()

        self.potential = potential
        self.op = np.array([[1., 0.], [0., 0.], [0., 0.]])

        self.totalEnergyOp = totalEnergyOpGenerator(self.potential, 1./self.dx**2 + 1./self.dy**2)

        self.eigenstates = []

        self.momentumSpace()

        # The system is a rectangular "box". Equivalent to having infinite potential walls at the boundary

    def changePotential(self, newPotential):
        self.potential = newPotential
        self.totalEnergyOp = totalEnergyOpGenerator(self.potential, 1. / self.dx ** 2 + 1. / self.dy ** 2)
        self.eigenstates.clear()

    def evolveStep(self, dt):
        #crankNicolson2DSchrodingerStepLegacy(self.X, self.Y, self.t, dt, self.potential,
        #                                     self.psi, self.psiCopy, extra_param=self.extra_param)
        if self.step == 'eigen':
            crankNicolson2DSchrodingerStepClosedBoxEigen(self.X, self.Y, self.t, dt, self.potential,
                                                         self.psi, self.psiEigen, self.psiMod, extra_param=self.extra_param)
        if self.step == 'exact':
            crankNicolson2DSchrodingerStepExact(self.X, self.Y, self.t, dt, self.potential,
                                                self.psi, self.psiCopy, self.psiMod, extra_param=self.extra_param)
        else:
            if self.customOperator is None:
                crankNicolson2DSchrodingerStepFastest(self.X, self.Y, self.t, dt, self.potential,
                                                             self.psi, self.psiCopy, self.psiMod, extra_param=self.extra_param)
            else:
                crankNicolson2DHalfStepSchrodinger(self.X, self.Y, self.t, dt, self.potential, self.customOperator,
                                        self.psi, self.psiCopy, self.psiMod, extra_param=self.extra_param)
        self.t += dt
        if self.renormStep: self.psi /= np.sqrt(self.norm())

    def evolveImagStep(self, dt):
        """Potential will be taken as constant. Useful for imaginary time displacements, approximate eigenstates. NEGATIVE!"""
        crankNicolson2DSchrodingerStepImaginary(self.X, self.Y, self.t, dt, self.potential,
                                                self.psi, self.psiCopy, extra_param=self.extra_param)

        # To avoid crashing in obtaining very very low values or very very high values
        # we normalize here
        self.psi[:,:] = self.psi[:,:]/np.sqrt(self.norm())

    def momentumSpace(self):
        fourierTransform2D(self.X, self.Y, self.psi, self.psiMom)
        if hred != 1: self.psiMom *= 1./hred

    def momentumSpaceModSquared(self):
        fourierTransform2D(self.X, self.Y, self.psi, self.psiMom)
        if hred != 1: self.psiMom *= 1./hred
        self.psiMod[:,:] = abs2(self.psiMom)        # Seems faster from tests
        #abs2Matrix(self.psiMom, self.psiMod)

    def modSquared(self):
        self.psiMod[:,:] = abs2(self.psi)            # Seems faster from tests. At high N abs2Matrix is better
        #abs2Matrix(self.psi, self.psiMod)

    def norm(self):
        return euclidNorm(self.psi, self.dx, self.dy)
        #return expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy)     #Simpson

    def renorm(self):
        #self.psi[:,:] = self.psi[:,:] / np.sqrt(self.norm())
        #np.multiply(self.psi, 1./np.sqrt(self.norm()),out=self.psi)
        self.psi *= 1./np.sqrt(self.norm())

    def kineticEnergy(self):
        #kineticEnergy(self.op, (self.x0, self.y0), (self.dx, self.dy), onlyUpdate=False)
        #return expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy, self.op)
        if self.step == 'eigen':
            return self.dx*self.dy/(4*self.Nx*self.Ny) * np.sum(abs2(dstn(self.psi[1:-1,1:-1], type=1))*(self.Kxmesh + self.Kymesh))

        if self.step == 'fourier':
            return self.expectedValueOpCentralMomentum(kineticEnergyMomentum)

        return self.expectedValueOp(kineticEnergy)#expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy, kineticEnergy) # time and extra_param are irrelevant

    def potentialEnergy(self):
        # Possible bottleneck
        return abs2MatrixMultipliedExpected(self.X, self.Y, self.potential, self.psi,
                                            t=self.t, extra_param=self.extra_param)

        ####abs2MatrixMultiplied(self.X, self.Y, self.potential, self.psi, self.psiMod,
        ####                     t=self.t, extra_param=self.extra_param)

        #self.psiMod[:,:] = abs2(self.psi[:,:]) * self.potential(self.Xmesh[:,:], self.Ymesh[:,:], self.t)
        # Can't include IFs this first way apparently
        # Must be same summation type as norm
        ####return np.trapz(np.trapz(self.psiMod))*self.dx*self.dy
        #return simpson_complex_list2D(self.dx, self.dy, self.psiMod)

    def approximateEigenstate(self, maxiter = 100, callback=None, tol = 1e-4, resetInit=True):
        """
        Iterates some time until, approximately, the newest lowest energy eigenstate remains
        Returns true if it was found u
        """
        #print(len(self.eigenstates))

        if resetInit: self.setState(func1)

        for it in range(maxiter):
            for E, eigenstate in self.eigenstates:
                self.substractComponent(eigenstate)
            for __ in range(10):
                self.evolveImagStep(-2**(-6))
            self.renorm()

            isEigen, E = self.isEigenstate(tol=tol)
            if isEigen:
                #print("good")
                #insort(self.eigenstates,(E,self.psi.copy()),key=lambda eig: eig[0]) # sort by Energy. We add elements already sorted. key param for python 3.10>=
                #self.eigenstates.append((E, self.psi.copy()))
                #self.eigenstates.sort(key=lambda eig: eig[0]) # sort by Energy, first element
                for i in range(len(self.eigenstates)):
                    if E < self.eigenstates[i][0]:
                        self.eigenstates.insert(i, (E, self.psi.copy()))
                        return True
                self.eigenstates.append((E, self.psi.copy()))
                return True
            if callback is not None:
                callback(it/maxiter)
        #print(E)

        return False



    def expectedValueOp(self, operator, doConjugate=True):
        return expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy,
                                       operator=operator, t=self.t, extra_param=self.extra_param, doConjugate=doConjugate)

    def expectedValueOpCentral(self, func):
        return abs2MatrixMultipliedExpected(self.X, self.Y, func, self.psi,
                                            t=self.t, extra_param=self.extra_param)

    def expectedValueOpMomentum(self, operator, forceFourier=False, doConjugate=True):
        if forceFourier: self.momentumSpace() # Makes sure psiMom is updated
        return expectedValueOperator2D(self.Px, self.Py, self.psiMom, self.psiCopy,
                                       operator=operator, t=self.t, extra_param=self.extra_param, doConjugate=doConjugate)

    def expectedValueOpCentralMomentum(self, func, forceFourier=False):
        if forceFourier: self.momentumSpace() # Makes sure psiMom is updated
        return abs2MatrixMultipliedExpected(self.Px, self.Py, func, self.psiMom,
                                            t=self.t, extra_param=self.extra_param)

    def expectedX(self):
        return self.expectedValueOpCentral(xFunc)#jit(lambda x, y, t, extra_param: x))

    def varX(self, expectedX = None):
        if expectedX is None: expectedX = self.expectedX()
        copy = self.extra_param[0]
        self.extra_param[0] = expectedX
        res = np.real(self.expectedValueOpCentral(xVarFunc))
        self.extra_param[0] = copy
        return np.sqrt(res)

    def expectedY(self):
        return self.expectedValueOpCentral(yFunc)#jit(lambda x, y, t, extra_param: y))

    def varY(self, expectedY = None):
        if expectedY is None: expectedY = self.expectedY()
        copy = self.extra_param[0]
        self.extra_param[0] = expectedY
        res = np.real(self.expectedValueOpCentral(yVarFunc))
        self.extra_param[0] = copy
        return np.sqrt(res)

    def expectedPx(self, forceFourier=False):
        # Making use of momentum space. Is basically same as regular
        return self.expectedValueOpCentralMomentum(xFunc, forceFourier)

    def varPx(self, expectedPx = None):
        if expectedPx is None: expectedPx = self.expectedPx()
        copy = self.extra_param[0]
        self.extra_param[0] = expectedPx
        res = np.real(self.expectedValueOpCentralMomentum(xVarFunc))
        self.extra_param[0] = copy
        return np.sqrt(res)

    def expectedPxPsi(self):
        # Central derivative
        return np.real(self.expectedValueOp( -1j * hred * np.array([[0., 0.], [-1./(2.*self.dx), +1./(2.*self.dx)] , [0.,0.]]) ))

    def expectedPy(self, forceFourier=False):
        # Making use of momentum space. Is basically same as regular
        return self.expectedValueOpCentralMomentum(yFunc, forceFourier)

    def varPy(self, expectedPy = None):
        if expectedPy is None: expectedPy = self.expectedPy()
        copy = self.extra_param[0]
        self.extra_param[0] = expectedPy
        res = np.real(self.expectedValueOpCentralMomentum(yVarFunc))
        self.extra_param[0] = copy
        return np.sqrt(res)

    def expectedPyPsi(self):
        # Central derivative
        return np.real(self.expectedValueOp( -1j * hred * np.array([[0., 0.], [0.,0.], [-1./(2.*self.dy), +1./(2.*self.dy)]]) ))

    def setState(self, state, x0Old = -10., xfOld=10., y0Old=-10., yfOld=10.):
        if callable(state):
            self.psi[:,:] = state(self.Xmesh, self.Ymesh, t = self.t, extra_param=self.extra_param)
        else:
            if type(state) is dict:
                x0Old = state["x0"]; xfOld = state["xf"]; y0Old = state["y0"]; yfOld = state["yf"]
                state = state["psi"]

            if state.shape == self.psi.shape and self.x0 == x0Old and self.y0 == y0Old and self.xf == xfOld and self.yf == yfOld:
                self.psi[:, :] = state[:, :]
            else:
                interpolate2D(self.psi, self.x0, self.xf, self.y0, self.yf,
                              state, x0Old, xfOld, y0Old, yfOld)
        self.psi[:, 0] = 0.; self.psi[:, self.Ny] = 0.; self.psi[0, :] = 0.; self.psi[self.Nx, :] = 0.
        #self.psi[:, :] = self.psi[:, :] / np.sqrt(self.norm())


    def setTempState(self, state):
        if callable(state):
            self.psiCopy[:,:] = state(self.Xmesh, self.Ymesh, t = self.t, extra_param=self.extra_param)
        else:
            exit(print("This method is only for states as functions"))

    def setPotential(self, potential):
        set2DMatrix(self.X, self.Y, potential, self.psiMod,
                    t=self.t, extra_param=self.extra_param)

    def isEigenstate(self, tol=1e-4):
        """Checks if the current state is an eigenstate up to some tolerance
        |p> eigenstate <==>  H|p> = E|p>  <==> H|p> - E|p> = 0
        We check:  || H|p> - E|p> || ~ 0
        || (H|p>)/E - |p> || < tol
        0 is a trivial eigenstate, but also if || |p> || ~ 0 we can get false positives. |p> is normalized just in case"""
        E = np.real(self.expectedValueOp(self.totalEnergyOp, doConjugate=False))#expectedValueOperator2D(self.X, self.Y, self.psi, self.psiCopy, self.totalEnergyOp,
                    #                        t=self.t, extra_param=self.extra_param, doConjugate=False))
        # After this |psiCopy> = H |psi>
        val = euclidNorm(self.psiCopy/E - self.psi, self.dx, self.dy) / euclidNorm(self.psi, self.dx, self.dy) #normalize
        #print("E =", E, "    || H|p> - E|p> || =",val)
        return (val < tol), E

    def substractComponent(self, component):
        # Psi can be expressed in a basis:
        # |Psi> = c1 |c1> + c2 |c2> + ...
        # We substract here a component, valid for any basis where basis vectors are orthogonal to c
        # |Psi> = ... + c |c> + ...
        # |Psi> - <c|Psi> |c> = ... + 0 |c> + ..

        if type(component) is dict:
            x0Old = component["x0"]; xfOld = component["xf"]; y0Old = component["y0"]; yfOld = component["yf"]
            component = component["psi"]

            interpolate2D(self.psiCopy, self.x0, self.xf, self.y0, self.yf,
                          component, x0Old, xfOld, y0Old, yfOld)

            component = self.psiCopy

        elif self.psi.shape != component.shape:
            print("Can't substract component, different shape!")
            return False

        c = innerProduct2D(component, self.psi, self.dx, self.dy)
        self.psi[:,:] = self.psi[:,:] - c*component[:,:]
        #np.subtract(self.psi,c*component, out=self.psi)   # what even is faster?

        return True


class ClassicalParticle:
    def __init__(self, QSystem):
        self.QSystem = QSystem
        self.time = self.QSystem.t

        self.x = self.QSystem.expectedX()
        self.y = self.QSystem.expectedY()

        if QSystem.step != 'exact' or QSystem.step != 'fastest':
            self.px = QSystem.expectedPx(forceFourier=True)
            self.py = QSystem.expectedPy(forceFourier=True)
        else:
            self.px = self.QSystem.expectedPxPsi()
            self.py = self.QSystem.expectedPyPsi()

        self.m = self.QSystem.mass

        self.yVec = np.array([self.x, self.px, self.y, self.py])

        """# Debug Energy
        f = open("debug.out", 'w')
        f.write("")
        f.close()
        self.f = open("debug.out", 'a')"""

    def evolveStep(self, dt):
        self.yVec = RungeKutta4(self.time, dt, self.yVec, self.deriv)

        # Bounce on walls
        if self.yVec[0] < self.QSystem.x0:
            self.yVec[0] = self.QSystem.x0 + self.QSystem.x0 - self.yVec[0]
            self.yVec[1] = -self.yVec[1]
        elif self.yVec[0] > self.QSystem.xf:
            self.yVec[0] = self.QSystem.xf - self.yVec[0] + self.QSystem.xf
            self.yVec[1] = -self.yVec[1]
        if self.yVec[2] < self.QSystem.y0:
            self.yVec[2] = self.QSystem.y0 + self.QSystem.y0 - self.yVec[2]
            self.yVec[3] = -self.yVec[3]
        elif self.yVec[2] > self.QSystem.yf:
            self.yVec[2] = self.QSystem.yf - self.yVec[2] + self.QSystem.yf
            self.yVec[3] = -self.yVec[3]
        self.x = self.yVec[0]; self.px = self.yVec[1]; self.y = self.yVec[2]; self.py = self.yVec[3]
        self.time += dt

        # Debug Energy
        #self.f.write("{0}\t{1}\t{2}\t{3}\n".format(self.time, self.kineticEnergy(), self.potentialEnergy(), self.totalEnergy()))

    def deriv(self, t, yin):
        # yin: [x, px, y, py]
        res = np.empty(4, dtype=np.float64)
        res[0] = yin[1]/self.m
        res[1] = (self.QSystem.potential(yin[0]-self.QSystem.dx, yin[2], t=t, extra_param=self.QSystem.extra_param)
                 -self.QSystem.potential(yin[0]+self.QSystem.dx, yin[2], t=t, extra_param=self.QSystem.extra_param) )\
                 /(2.*self.QSystem.dx)
        res[2] = yin[3] / self.m
        res[3] = (self.QSystem.potential(yin[0], yin[2] - self.QSystem.dy, t=t, extra_param=self.QSystem.extra_param)
                  - self.QSystem.potential(yin[0], yin[2] + self.QSystem.dy, t=t, extra_param=self.QSystem.extra_param)) \
                 / (2 * self.QSystem.dy)
        return res

    def kineticEnergy(self):
        return (self.px*self.px + self.py*self.py)/(2*self.m)

    def potentialEnergy(self):
        return self.QSystem.potential(self.x, self.y, self.time, extra_param =self.QSystem.extra_param)

    def totalEnergy(self):
        return self.kineticEnergy() + self.potentialEnergy()


def RungeKutta4(t, dt, yin, deriv):
    k1 = deriv(t     , yin        )
    k2 = deriv(t+dt/2, yin+dt*k1/2)
    k3 = deriv(t+dt/2, yin+dt*k2/2)
    k4 = deriv(t+dt  , yin+dt*k3  )
    return yin + dt*(k1 + 2*k2 + 2*k3 + k4)/6.


class localOperator:
    """
    A local operator is an operator that only acts on a point and its closest neighbours
    i.e. It's useful because we can express any combination:
    L = fxx(x,y,t) ∂^2_x + fyy(x,y,t) ∂^2_y + fx(x,y,t) ∂_x + fy(x,y,t) ∂_y + F(x,y,t)
    It just simplfies creating an operator, to be used for defining the hamiltonian
    """
    pass


# We define operators here differently. [0,0] Center element [dir, 0] center element of dir, [dir, 1] forward dir, [dir, -1] backward dir
@jit(nopython=True)
def crankNicolson2DHalfStepSchrodinger(X, Y, t, dt, potential, operator, psi, psiTemp, psiMod, extra_param=np.array([])):
    """ Solves System:   d/dt psi = [Operator - i/h V ] psi
    We already assume here h = m = 1
    Step 0: Psi(t)  ->   Step 1: psiTemp = Psi(t+dt/2) ->  Step 2: psi = Psi(t+dt)"""


    # To make the code even faster, all of these temporary arrays could be created already. So no step needs to be taken
    # Declare all necessary arrays. Trying to minimize need to allocate memory
    tempBX = np.empty(len(psi)-2, dtype=np.complex128)
    tempBY = np.empty(len(psi[0])-2, dtype=np.complex128)

    tridX = np.empty((3,len(psi)-2), dtype=np.complex128)
    tridY = np.empty((3,len(psi[0])-2), dtype=np.complex128)

    # Gamma is used for tridiag, it is allocated here to avoid repeated unnecessary memory allocation
    gamma = np.empty((max(len(psi)-1, len(psi[0])-1),), dtype=np.complex128)

    # Parameters for calcualtions
    dx = (X[-1]-X[0])/(len(X)-1)
    dy = (Y[-1]-Y[0])/(len(Y)-1)

    op = np.array([[1., 0.,0.], [0., 0.,0.], [0., 0.,0.]], dtype=np.complex128)
    operator(op, X[0], Y[0], dx, dy, t, extra_param=extra_param, onlyUpdate=False)

    set2DMatrix(X, Y, potential, psiMod, t=t+dt/2, extra_param=extra_param)

    # First half step. At t we only do derivatives in y direction. At t+dt/2 in x direction. Thus the name, alternating
    # direction implicit method. Important to remember that at boundaries wave is fixed (infinite potential wall)
    for ity in range(1, len(psi[0])-1):
        for itx in range(1, len(psi)-1):
            operator(op, X[itx], Y[ity], dx, dy, t+dt/2., extra_param=extra_param)
            tridX[0, itx - 1] = -dt*op[1,-1]
            tridX[2, itx - 1] = -dt*op[1, 1]
            tridX[1,itx-1] = 2 - dt*(op[0,0]+op[1,0] - 1j * psiMod[itx, ity])   #PROBLEM, PSIMOD POTENTIAL SHOULD INCLUDE IMAGINARY PART IF NEC. BUT NOT LIKE THIS!!!! 1J??? ALSO NEEDS ARBITRARY - sign
            tempBX[itx-1] = (psi[itx,ity-1]*op[2,-1] + psi[itx,ity+1]*op[2,1])*dt \
                            + (2 + dt*op[2,0])*psi[itx,ity]
        tridiag(gamma, tridX, tempBX, psiTemp[1:-1,ity])

    psiTemp[:,0] = psi[:,0]
    psiTemp[:, len(psi[0]) - 1] = psi[:,len(psi[0]) - 1]
    psiTemp[0,:] = psi[0,:]
    psiTemp[len(psi)-1,:] = psi[len(psi)-1,:]

    # We do the final half step. Again, at t+dt/2 we only do derivatives in x direction, but now at t+dt in y direction
    for itx in range(1, len(psi)-1):
        for ity in range(1, len(psi[0])-1):
            operator(op, X[itx], Y[ity], dx, dy, t+dt/2., extra_param=extra_param)
            tridY[0, ity - 1] = -dt * op[2,-1]
            tridY[2, ity - 1] = -dt * op[2, 1]
            tridY[1,ity-1] = 2-dt*op[2,0]
            tempBY[ity-1] = (psiTemp[itx-1,ity]*op[1,-1] + psiTemp[itx+1,ity]*op[1,1])*dt \
                            + (2 + dt*(op[0,0]+op[1,0] - 1j * psiMod[itx,ity])) *psiTemp[itx,ity]

        tridiag(gamma, tridY, tempBY, psi[itx,1:-1])



@njit
def aharonovBohmOperator(op, x, y, dx, dy, t=0, extra_param=np.array([1,1,1,1,0.2]), dir=-1, onlyUpdate=True):
    # -i/h [H-V] = -ih/2m [∂^2_x + ∂^2_y + i alpha / r^2 * (-y∂x + x∂y) - a^2/4r^2]
    alpha = extra_param[4]
    #if onlyUpdate: return # The operator changes at every point
    invR2 = 1/(x*x+y*y+1e-10)

    op[0,0] = 1j/2*(- alpha**2 * invR2)

    op[1, 0] = 1j*(-1./dx**2)
    op[1, 1] = 1j/2*(1 / dx**2 + 2j*alpha * invR2 * (-y*1/(2*dx)))
    op[1,-1] = 1j/2*(1 / dx**2 + 2j*alpha * invR2 * (+y*1/(2*dx)))

    op[2, 0] = 1j*(-1./dy**2)
    op[2, 1] = 1j/2*(1 / dy**2 + 2j*alpha * invR2 * (+x*1/(2*dy)))
    op[2,-1] = 1j/2*(1 / dy**2 + 2j*alpha * invR2 * (-x*1/(2*dy)))



@jit
def slit(x, n, width, dist):
    return x <= n/2*width + dist*(n-1)/2 and (x-width/2*(n%2)+dist/2*((n+1)%2))/(dist+width) % 1 >= dist/(dist+width)


parameters = np.array([2, 1, 1, 0.5]) # [n, width, dist, GruixWall]

@njit
def slitpotential(x, y, t, extra_param=np.array([])):
    return 400. if (abs(x) < parameters[3] / 2 and not slit(abs(y), parameters[0], parameters[1], parameters[2])) \
        else 0.

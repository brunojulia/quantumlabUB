"""
Joan Ainaud Fondevila

Main code to run modules (no Kivy)

The important part is at the end. TWO EXAMPLES: module = 'closeOpen' or 'moveParticle'

The process to create a module is:
1. Define QuantumSystem using class:
    QuantumSystem2D(Nx, Ny, x0, y0, xf, yf, initState, t0=0., potential=potential0, extra_param=None)
    Needs:
    - Parameters
    - Initial state (can be given either as a function or a matrix to be copied)
    - Potential to use. extra_param are passed to the Potential, and due to @jit not allowing global variables,
      they represent such global variables in case they are needed (necessary for interactivity)

IMPORTANT:
    Potentials need to be of the shape:
    > potential(x, y, t, extra_param=None)

    And be decorated with @jit or @njit (for example, when defining them)

    @njit
    def potentialExample(...):
        ...

2. Create animation using QuantumAnimation class
    - Interactivity is added here with optional argument:
    extraCommands = [(action1, command1), (action2, command2), ...]
    for instance, an action could be 'key_press_event' and actions are functions, for instance pausing if key='p'
    Pausing is already implemented by default

3. To play the animation live, and interact, write plt.show() after creating animation.

4. The animation can be saved, for example as a gif:
    writergif = matplotlib.animation.PillowWriter(fps=20)
    animation.animation.save("./Results/testResult.gif", writer=writergif)
    plt.close()

In this example, animation is the Quantum Animation: animation = QuantumAnimation(width, height, ...)
"""

import numpy as np
import scipy
from numba import jit
from numba import njit
import numba
import time
import crankNicolson2D as mathPhysics
import animate
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import matplotlib.animation


#####################
##### DEBUGGING #####
#####################

#import warnings     # For debugging. Warnings stop the program, and thus also show where they occur
#warnings.filterwarnings('error')
#warnings.filterwarnings('error', category=FutureWarning)


def timer(func):
    """
    Takes a function and returns another function which does the same but also prints execution time in console
    :param func: Function to be modified
    :return: New function which prints execution time
    """
    def wrapper(*args, **kwargs):
        t0 = time.time()
        val = func(*args, **kwargs)
        print("Process took: ", time.time()-t0, " (s)")
        return val
    return wrapper

#mathPhysics.applyOperator = timer(mathPhysics.applyOperator)
#mathPhysics.applyOperator2D = timer(mathPhysics.applyOperator2D)
#mathPhysics.expectedValueOperator2D = timer(mathPhysics.expectedValueOperator2D)
#mathPhysics.crankNikolson2DSchrodingerStep = timer(mathPhysics.crankNikolson2DSchrodingerStep)
#mathPhysics.abs2 = timer(mathPhysics.abs2)




#####################
##### PROGRAMA ######
#####################

##################################
# USEFUL INITIAL STATE FUNCTIONS #
##################################

def gaussianPacket(x, x0, sigma, p0):
    return 1./(2*np.pi*sigma**2)**(0.25) * np.exp(-1./4. * ((x-x0)/sigma)**2) * np.exp(1j/mathPhysics.hred * p0*(x))#-x0/2)) ??? Apunts mec quantica


# GENERATOR FOR GAUSSIAN INITIAL STATES
def gaussian2D(x0, sigmax, px, y0, sigmay, py):
    def result(x, y):
        return gaussianPacket(x, x0, sigmax, px) * gaussianPacket(y, y0, sigmay, py)
    return result


#example1
def inicial(x, p_0):
    return np.exp(-(x - 7) ** 2 / 4) / (2 * np.pi) ** (0.25) * np.exp(- 1j * p_0 * x)

#example2
def inicial2D(x, y):
    global p_0
    return inicial(x, p_0)*inicial(y, 0.5*p_0)


def eigenvectorsHarmonic1D(x, x0, n, k):
    x = x * np.sqrt(np.sqrt(k))
    return np.power(k, 1/8.)/np.sqrt(2**n * np.math.factorial(n) * np.sqrt(np.pi)) * np.exp(-(x-x0)**2/2.)*\
np.polynomial.hermite.hermval(x, [0]*(n-1) + [1])

# GENERATOR FOR HARMONIC OSCILLATOR EIGENVECTORS
def eigenvectorHarmonic2DGenerator(x0, nx, y0, ny, k):
    def result(x, y):
        return eigenvectorsHarmonic1D(x, x0, nx, k)*eigenvectorsHarmonic1D(y, y0, ny, k)
    return result


################
## PARAMETRES ##
################

L = 25.
#p_0 = 5

n = 2
X0 = (-L,)*n
x0, y0 = -L, -L
Xf = (L,)*n
xf, yf = L, L

Nx = 201
Ny = 201
t = 0.

# Example of creating a gaussian initial state
inicial2D = gaussian2D(0., 1., 0.,
                       0*0.7*L, 1., 0.)

# Example
kHarm = 8.
inicial2D = eigenvectorHarmonic2DGenerator(0., 2, 0., 2, kHarm)
                                        #  x0  nx y0  ny

################
## POTENTIALS ##
################

@jit
def potential0(x, y, t, extra_param):
    return 0.


@jit
def potentialBarrier(x, y, t, extra_param):
    return np.exp(-(x ** 2) / 0.1) * 5 / np.sqrt(0.1 * np.pi)

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

#extra_param = np.array([L/2, 0., 0.])  # [potentialx0, potentialy0, potentialvx, potentialvy, potentialt0]
# FUNCTION THAT TAKES USER INPUT
def openCloseWellKeyboard(event):
    # Extra param is: [radius, radiusVelocity, timeOfLastChange]
    # Up and Down to increase/decrease radius
    # Constant decreasing
    global extra_param
    if event.key == 'up' or 'down' or 'left' or 'right':
        t = testSystem.t
        extra_param[0] = extra_param[0] + extra_param[1] * (t - extra_param[2])
        extra_param[2] = t
    if event.key == 'up':
        extra_param[1] += 0.1
    if event.key == 'down':
        extra_param[1] -= 0.1
    print("Rate of change of radius: ", extra_param[1])

########################################################################
#


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
def potentialHarmonicWell(x, y, t, extra_param):
    global kHarm
    res = 1/2. * kHarm*( x**2 + y**2)
    if res > 100.: return 100
    return res


######################################################
# MOVE HARMONIC TRAP


@njit([numba.float64(numba.float64, numba.float64, numba.float64, numba.float64[:])])
def potentialHarmonicWellMoving(x, y, t, extra_param):
    global kHarm
    res = 1/2. * kHarm *( (x-extra_param[0]-extra_param[2]*(t-extra_param[4]))**2
              + (y-extra_param[1]-extra_param[3] * (t - extra_param[4]))**2
              )
    if res > 100.: return 100.
    return res

# EXTRA PARAM:
# extra_param = np.array([0., 0., 0., 0., 0.])  # [potentialx0, potentialy0, potentialvx, potentialvy, potentialt0]

# FUNCTION TO INTERACT WITH USER INPUT
def moveHarmonicWellKeyboard(event):
    global extra_param
    if event.key == 'up' or 'down' or 'left' or 'right':
        t = testSystem.t
        extra_param[0] = extra_param[0] + extra_param[2]*(t-extra_param[4])
        extra_param[1] = extra_param[1] + extra_param[3] * (t - extra_param[4])
        extra_param[4] = t
    if event.key == 'up':
        extra_param[3] += 0.25
    if event.key == 'down':
        extra_param[3] -= 0.25
    if event.key == 'left':
        extra_param[2] -= 0.25
    if event.key == 'right':
        extra_param[2] += 0.25


# FUNCTION TO DRAW A GOAL CIRCLE
goalRadius = 2.
goalX = x0 + goalRadius + random.random()*(xf-x0 - 2*goalRadius)
goalY = y0 + goalRadius + random.random()*(yf-y0 - 2*goalRadius)
goalCircle = plt.Circle((goalX, goalY), goalRadius, alpha=0.2, color='black')
firstDraw = True
drawnCircle = None
def drawCircle():
    global goalCircle, firstDraw, drawnCircle
    if firstDraw:
        drawnCircle = animation.axPsi.add_patch(goalCircle)
        firstDraw = False
    else: return drawnCircle

######################################################
#

##########################################
############## Main code #################
##########################################

# Bottleneck: Drawing matplotlib.
# For example, when the axis need to be readjusted, so the whole plot needs to be redrawn, slowest step

# Run here actual module.
# Two cases

#module = "closeOpen", "moveParticle"   CHOSE ONE OF THE TWO

module = 'none'

Nx = 201
Ny = 201

L = 10.
#p_0 = 5

n = 2
X0 = (-L,)*n
x0, y0 = -L, -L
Xf = (L,)*n
xf, yf = L, L

inicial2D = gaussian2D(7, 1., -5., 7., 1., -2.5)
testSystem = mathPhysics.QuantumSystem2D(Nx, Ny, *X0, *Xf, inicial2D,
                                             potential=potentialGravity)

animation = animate.QuantumAnimation(testSystem, width=12, height=7, duration = 10, dtSim=0.01, dtAnim=0.04,
                                         showPotential=True, updatePotential=True,
                                         showMomentum=True, showEnergy=True, showNorm=True,
                                         scalePsi=True, scaleMom=True)

#writergif = matplotlib.animation.PillowWriter(fps=20) #fps=25
#animation.animation.save("./Results/FallTestWithNewCN.gif", writer=writergif)
#plt.close()
plt.show()

if module=='moveParticle':
    # Change harmonic trap speed with up left right and down. Try to move the particle inside the circle
    # PROBLEMS:
    #   Momentum can get very high -> Bad Simulation

    # After 10 seconds, a "score" is shown in console

    kHarm = 8.
    inicial2D = eigenvectorHarmonic2DGenerator(0., 2, 0., 2, kHarm)
    extra_param = np.array([0., 0., 0., 0., 0.])  # [potentialx0, potentialy0, potentialvx, potentialvy, potentialt0]
    testSystem = mathPhysics.QuantumSystem2D(Nx, Ny, *X0, *Xf, inicial2D,
                                             potential=potentialHarmonicWellMoving, extra_param=extra_param)

    ### Example of complete animation. Slower
    animation = animate.QuantumAnimation(testSystem, width=12, height=7, duration = 10, dtSim=0.01, dtAnim=0.04,
                                         showPotential=True, updatePotential=True,
                                         showMomentum=True, showEnergy=True, showNorm=True,
                                         scalePsi=True, scaleMom=True,
                                         extraCommands=[('key_press_event', moveHarmonicWellKeyboard)],
                                         extraUpdates=[drawCircle])
    ### Simpler but faster animation
    """animation = animate.QuantumAnimation(testSystem, width=6.4, height=4.8, duration = 10, dtSim=0.01, dtAnim=0.04,
                                         showPotential=True, updatePotential=True,
                                         extraCommands=[('key_press_event', moveHarmonicWellKeyboard)],
                                         extraUpdates=[drawCircle])"""


    print("10 seconds! Move the particle inside the circle")
    t0 = time.time()
    plt.show()
    print("FPS: ", animation.frame/(time.time()-t0))

    @jit
    def isInsideCircle(op, X, dx, t=0, extra_param=None, dir=-1, onlyUpdate=True):
        global hred, M
        global goalX, goalY
        if not onlyUpdate:
            op[0][0] = 0.
            op[1][:] = 0.
            op[2][:] = 0.
        if (X[0] - goalX) ** 2 + (X[0] - goalX) ** 2 > goalRadius**2: op[0][0] = 0.
        else: op[0][0] = 1.

    print("Probability of finding the particle inside the circle (SCORE):")
    print(np.real(testSystem.expectedValueOp(isInsideCircle)/testSystem.norm()))

if module=='closeOpen':
    # Increase / Decrease radius ratio of change with Up and Down keys

    extra_param = np.array([L / 2, 0., 0.])

    inicial2D = gaussian2D(0., 2., 0.,
                           0., 2., 0.)
    testSystem = mathPhysics.QuantumSystem2D(Nx, Ny, *X0, *Xf, inicial2D,
                                             potential=potentialClosingManual, extra_param=extra_param)

    ### Example of complete animation. Slower
    animation = animate.QuantumAnimation(testSystem, width=12, height=7, duration = None, dtSim=0.01, dtAnim=0.04,
                                         showPotential=True, updatePotential=True,
                                         showMomentum=True, showEnergy=True, showNorm=True,
                                         scalePsi=True, scaleMom=True,
                                         extraCommands=[('key_press_event', openCloseWellKeyboard)])
    ### Simpler but faster animation
    """animation = animate.QuantumAnimation(testSystem, width=6.4, height=4.8, duration=None, dtSim=0.01, dtAnim=0.04,
                                         showPotential=True, updatePotential=True,
                                         extraCommands=[('key_press_event', openCloseWellKeyboard)])"""

    t0 = time.time()
    plt.show()
    print("FPS: ", animation.frame / (time.time() - t0))


#writergif = matplotlib.animation.PillowWriter(fps=20) #fps=25
#animation.animation.save("./Results/testResult.gif", writer=writergif)
#plt.close()


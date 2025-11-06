from cranknicolson.cn2d import crank_nicolson2D

import numpy as np
import threading
import json
from scipy.signal import square

import matplotlib.pyplot as plt

class DSexperiment(object):
    """docstring for DSexperiment."""
    def __init__(self, Lx = 15.0, Nx = 300, Ny = 200, Vo = 200, n = 2, sx = .25, sy = 2, d = 4, measurepos = 50, measurewidth = 5):
        #Mesh parameters
        self.Lx = Lx
        self.Nx = Nx
        self.Ny = Ny

        self.dx = 2*Lx/Nx
        self.Ly = Ny*self.dx/2

        #Slits parameters
        self.n = n
        self.sx = sx
        self.sy = sy
        self.d = d

        #Grid coordinates
        self.x, self.y = np.meshgrid(np.arange(-self.Lx, self.Lx, self.dx), np.arange(-self.Ly, self.Ly, self.dx))

        #psi0
        self.times = [0]
        self.psit = [np.zeros(self.x.shape, dtype = np.complex128)]
        self.Pt = [np.zeros(self.x.shape, dtype = np.float64)]

        #V
        self.Vo = Vo
        self.V = None
        self.compute_potential()

        #measures
        self.mp = measurepos
        self.mw = measurewidth
        self.old_mp = self.mp
        self.old_mw = self.mw

        self.measurements = []

    def compute_potential(self):
        xb = 2.5
        duty = self.d/(self.d+self.sy)
        delta = self.n*(self.d+self.sy)

        V = (1+square(2*np.pi*self.y/(self.d+self.sy) + np.pi*(self.n%2 + duty), duty))/2
        self.V = self.Vo*np.maximum(np.heaviside(self.y-delta/2, 1) + np.heaviside(-(self.y+delta/2), 1), V)*np.uint8(np.abs(self.x-xb) < self.sx/2)


    def update_slits(self, n = None, sx = None, sy = None, d = None):
        changed = False
        if n is not None:
            changed = changed or int(self.n) != int(n)
            self.n = n
        if sx is not None:
            changed = changed or float(self.sx) != float(sx)
            self.sx = sx
        if sy is not None:
            changed = changed or float(self.sy) != float(sy)
            self.sy = sy
        if d is not None:
            changed = changed or float(self.d) != float(d)
            self.d = d

        if changed:
            self.compute_potential()

        return changed

    def update_measure_screen(self, mp = None, mw = None):
        changed = False
        if mp is not None:
            changed = changed or float(self.mp) != float(mp)
            self.mp = mp
        if mw is not None:
            changed = changed or float(self.mw) != float(mw)
            self.mw = mw


        return changed

    def compute_py(self, force = False):
        if self.old_mp != self.mp or self.old_mw != self.mw or force:
            self.py = np.sum(np.sum(self.Pt, axis = 0)[:,self.mp:(self.mp+self.mw)], axis = 1)
            self.old_mp = self.mp
            self.old_mw = self.mw


    def set_gaussian_psi0(self, x0 = 10, y0 = 0, p0x = 20, p0y = 0, s = 2):
        """
        sets psit[0] to a gaussian wavepacket
        """
        r2 = (self.x-x0)**2 + (self.y-y0)**2
        self.psit = [np.exp(-1j*(p0x*self.x + p0y*self.y))*np.exp(-r2/(4*s**2))/(2*s**2*np.pi)**(.5)]
        self.Pt = [np.absolute(self.psit[0])**2]

    def compute_evolution(self, tmax = 2.5, dt = 0.01, update_callback = None, done_callback = None, parallel = True):
        """
        Computes the evolution of the experiment
        """
        self.tmax = tmax
        self.dt = dt
        self.update_callback = update_callback
        self.done_callback = done_callback

        self.compute_potential()

        if parallel:
            CNThread(1, "CrankNicolsonThread", self).start()
        else:
            CNThread(1, "CrankNicolsonThread", self).run()

    def measure(self, N = 1):
        self.compute_py()

        new_measures = []
        M = 1.5*np.max(self.py)

        while len(new_measures) < N:
            i = np.random.randint(0, len(self.py))
            y = M*np.random.random()
            if y < self.py[i]:
                new_measures.append( (self.mp + np.random.randint(0, self.mw), i, np.random.normal(0, 1, None)))

        self.measurements += new_measures

    def clear_measurements(self):
        self.measurements = []

    def save_to_files(self, filename = "experiment"):
        with open(filename + "_parameters.json", "w") as outFile:
            dic = [self.Lx, self.Nx, self.Ny, self.Vo, self.n, self.sx, self.sy, self.d]
            json.dump(dic, outFile)

        np.save(filename + "_Pt.npy", self.Pt,)


def create_experiment_from_files(filename):
    with open(filename + "_parameters.json", "r") as inFile:
        param = json.load(inFile)

    exp = DSexperiment(*param)
    exp.Pt = np.load(filename + "_Pt.npy")
    exp.compute_py(force = True)

    return exp

class CNThread(threading.Thread):
    """
    This class represents the thread that runs parallel to the app in order to
    compute the simulation while updating the UI on the app
    """
    def __init__(self, threadID, threadName, experiment):
        threading.Thread.__init__(self)
        self.exp = experiment

    def run(self):
        self.exp.psit, self.exp.times = crank_nicolson2D(self.exp.x, self.exp.y, self.exp.psit[0], self.exp.V, tmax = self.exp.tmax, dt = self.exp.dt, callback = self.exp.update_callback)
        self.exp.Pt = np.absolute(self.exp.psit)**2
        mp, mw = self.exp.mp, self.exp.mw
        self.exp.compute_py(force = True)
        self.exp.done_callback()

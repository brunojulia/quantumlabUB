import matplotlib
from matplotlib import animation
import matplotlib.pyplot as plt
import matplotlib.patheffects as peffects
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

if(__name__ == "crankNicolson.animate"): import crankNicolson.crankNicolson2D as mathPhysics
else: import crankNicolson2D as mathPhysics         #   Problems with making it work outside of folder
import numpy as np
import time
import matplotlib.style as mplstyle
#mplstyle.use('fast')
"""On performance:
    https://matplotlib.org/stable/users/explain/performance.html"""


matplotlib.rcParams['keymap.back'].remove('left')
matplotlib.rcParams['keymap.forward'].remove('right')
#fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

class QuantumAnimation:              # inches
    def __init__(self, QSystem, width=6.4, height=4.8,
                 dtSim=0.01, dtAnim = 0.05, duration = None, realTime = True,
                 showEnergy = False, showNorm = False, showMomentum = False, showPotential = False, updatePotential = False,
                 potentialCutoff = None, psiCutoff = None, scalePsi=False, scaleMom=False,
                 extraCommands = None, extraUpdates = None, isKivy = False, stepsPerFrame = 0,
                 debugTime = False, callbackProgress = True):
        """
        Declaration of a Quantum Animation
        :param QSystem: Quantum System. The physics happen here, see crankNikolson2D QuantumSystem class.
        :param width: width of figure (inches)
        :param height: height of figure (inches)
        :param dtSim: dt of Simulation. The system will be evolved (Crank-Nicolson 2D ADI) with this dt.
        :param dtAnim: dt of Animation. Should be a multiple of dtSim. Every frame corresponds to dtAnim, and thus
        the quantum system is evolved dtAnim/dtSim times each frame.
        :param duration: Duration of animation.
        :param showEnergy: set to True to show Kinetic/Potential/Total Energy
        :param showNorm: set to True to show how the norm evolves (doesn't always remain constant)
        :param showMomentum: set to True to show the system in momentum space
        :param showPotential: set to True to show the potential acting on the system
        :param updatePotential: set to update the potential in case it will evolve. Important for games
        :param potentialCutoff: Don't draw potential at points where it's value is below cutoff
        :param psiCutoff: Don't draw psi at points where it's value (module) is below cutoff
        :param scalePsi: set to True to rescale colorscale of position space. Helps see if the particle dissipates, but
        makes it harder to see how small is the probability of each section in absolute terms
        :param scaleMom: set to True to rescale colorsacale of momentum space.
        :param extraCommands: Trigger actions, for example on click, and command to execute:
        extraCommands = [(action1, command1), (action2, command2), ...]. command: function. Action, ex: 'key_press_event'
        :param extraUpdates: Run some extra functions every update
        :param isKivy: Links canvas to Kivy Backend and disables automatic animation
        """

        self.dtAnim = dtAnim
        self.dtSim = dtSim
        self.stepsPerFrame = stepsPerFrame  # If 0, that means try "real time" with dtAnim
        self.imagdt = False

        if(int(dtSim/dtAnim) != dtSim//dtAnim): print("WARNING: dtAnim is not a multiple of dtSim, can't do real time")
        self.QSystem = QSystem
        self.width = width
        self.height = height
        self.frame = 0

        self.duration = duration
        self.realTime = realTime
        if(duration == None): self.frames = None
        elif realTime: self.frames = int(duration/dtAnim +1)
        else: self.frames = int(duration/ (dtAnim if stepsPerFrame==0 else dtSim*stepsPerFrame))
        # Simulation time. Careful...

        self.showEnergy = showEnergy
        self.showNorm = showNorm
        self.showMomentum = showMomentum
        self.showPotential = showPotential
        self.updatePotential = updatePotential
        self.potentialCutoff = potentialCutoff
        self.psiCutoff = psiCutoff
        self.scalePsi = scalePsi
        self.scaleMom = scaleMom

        self.paused = False
        self.text = None

        self.debugTime = debugTime
        self.callbackProgress = callbackProgress

        self.extraCommands = extraCommands
        self.extraUpdates = extraUpdates
        # Extra Commands: [(action1, command1), (action2, command2), ...]

        self.TList = []
        self.KList = []
        self.VList = []
        self.EList = []
        self.NormList = []
        self.potentialMat = np.ndarray((self.QSystem.Nx+1, self.QSystem.Ny+1), dtype=np.float64)
        # Nx+1 points!!! x0, x1, ..., xNx
        self.isKivy = isKivy

        if not self.isKivy:
            self.fig = plt.figure(figsize=(width, height))
        else:
            self.fig = plt.figure()
            FigureCanvasKivyAgg(self.fig)   # Canvas is now Kivy Canvas
             # If we do a Kivy Canvas, we can't use FuncAnimation which implements blitting automatically,
             # so we need to implement blitting manually. To do this, we need to keep track of when we update the figure
            #self.drawEvent = self.fig.canvas.mpl_connect('draw_event', self.on_draw) # For Blitting, if it worked?
            #self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

        #fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

        if not self.isKivy: self.pauseEvent = self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        if self.extraCommands != None:
            for action, command in self.extraCommands:
                self.fig.canvas.mpl_connect(action, command)

        # Code
        self.axEnergy = None
        self.axNorm = None
        self.axMomentum = None

        self.datPsi = None
        self.datPot = None
        self.datMom = None
        self.lineK = None
        self.lineV = None
        self.lineE = None
        self.lineN = None

        self.reset_plot()

        if not self.isKivy:
            self.animation = animation.FuncAnimation(self.fig, self.update, interval=dtAnim * 1000, blit=True,
                                                     frames=self.frames)


    def reset_plot(self, width = None, height = None,
                   showEnergy = None, showNorm = None, showMomentum = None, showPotential = None,
                   scalePsi=None, scaleMom=None):
        if width != None: self.width = width
        if height != None:  self.height = height
        if showEnergy != None: self.showEnergy = showEnergy
        if showNorm != None: self.showNorm = showNorm
        if showMomentum != None: self.showMomentum = showMomentum
        if showPotential != None: self.showPotential = showPotential
        if scalePsi != None: self.scalePsi = scalePsi
        if scaleMom != None: self.scaleMom = scaleMom

        extraSubplots = 0
        if self.showEnergy: extraSubplots += 1
        if self.showNorm: extraSubplots += 1
        if self.showMomentum: extraSubplots += 1

        self.fig.clf()                      #Clear figure
        self.fig.figsize = (self.width, self.height)

        self.axPsi = self.fig.add_subplot(1, 2+(extraSubplots>0), (1, 2))
        if self.showEnergy:  self.axEnergy = self.fig.add_subplot(3, 3, 3)     #(2, 6, 11)
        else: self.axEnergy = None
        if self.showNorm: self.axNorm = self.fig.add_subplot(3, 3, 6)          #(2, 6, 12)
        else: self.axNorm = None
        if self.showMomentum: self.axMomentum = self.fig.add_subplot(3, 3, 9)   #(2, 3, 3)
        else: self.axMomentum = None


        # First drawing
        self.QSystem.modSquared()
        title = "Espai de posicions"
        if self.scalePsi: title += " (color reescalat)"
        self.axPsi.set_title(title)
        self.axPsi.set_xlabel("x")
        self.axPsi.set_ylabel("y")
        self.datPsi = self.axPsi.imshow(self.QSystem.psiMod.T, origin='lower',
                            extent=(self.QSystem.x0, self.QSystem.xf, self.QSystem.y0, self.QSystem.yf),
                            aspect='equal', cmap="viridis", interpolation=None)#, animated=True) # Doesn't do anything for imshow?
        if self.showPotential:
            mathPhysics.set2DMatrix(self.QSystem.X, self.QSystem.Y,
                                    self.QSystem.potential, self.potentialMat,
                                    t=self.QSystem.t, extra_param=self.QSystem.extra_param)
            self.datPot = self.axPsi.imshow(self.potentialMat.T, origin='lower',
                                extent=(self.QSystem.x0, self.QSystem.xf, self.QSystem.y0, self.QSystem.yf),
                                aspect='equal', cmap='gist_heat', alpha=0.3, interpolation=None)#, animated=True)
            #self.fig.colorbar(datPot, ax=ax1, label="Potencial: Força constant cap a baix")

        if self.showEnergy:
            self.axEnergy.set_title("Energia")
            if len(self.TList) == 0: self.axEnergy.set_xlim(self.QSystem.t, self.QSystem.t+5)
            self.axEnergy.set_ylim(min(np.min(self.potentialMat) - 1., 0.),
                                   np.real(self.QSystem.potentialEnergy() + self.QSystem.kineticEnergy() + 1 -
                                           min(np.min(self.potentialMat) - 1., 0.)))
            self.lineK, = self.axEnergy.plot(self.TList, self.KList, label="K")#, animated=True)  #Doesn't work with Kivy?
            self.lineV, = self.axEnergy.plot(self.TList, self.VList, label="V")#, animated=True)
            self.lineE, = self.axEnergy.plot(self.TList, self.EList, label="E")#, animated=True)
            self.axEnergy.grid()
            self.axEnergy.legend()

        if self.showNorm:
            self.axNorm.set_title("Normalització")
            self.axNorm.set_xlabel("t (s)")
            if len(self.TList) == 0: self.axNorm.set_xlim(self.QSystem.t, self.QSystem.t+5) #0, 5
            ####
            self.axNorm.set_ylim(1 - 0.001, 1. + 0.001)
            ##### SI es vol optimitzar fent blitting s'ha d'evitar fer canvis de límit
            self.lineN, = self.axNorm.plot(self.TList, self.NormList)#, animated=True)  #Doesn't work wit hKivy?

        if self.showMomentum:
            self.QSystem.momentumSpace()
            title = "Espai de moments"
            if self.scaleMom: title += " (color reescalat)"
            self.axMomentum.set_title(title)
            self.axMomentum.set_xlabel("Px")
            self.axMomentum.set_ylabel("Py")
            self.datMom = self.axMomentum.imshow(self.QSystem.psiMod.T, origin='lower',
                                extent=(self.QSystem.Px[0], self.QSystem.Px[-1],
                                        self.QSystem.Py[0], self.QSystem.Py[-1]),
                                cmap='hot', aspect='equal', interpolation=None)#, animated = True)

        plt.tight_layout()
        if self.isKivy: self.fig.canvas.draw()

    def reset_lists(self):
        self.KList = []
        self.VList = []
        self.EList = []
        self.NormList = []
        self.TList = []
        self.potentialMat = np.ndarray((self.QSystem.Nx+1, self.QSystem.Ny+1), dtype=np.float64)

    def on_key_press(self, event):
        if event.key == 'p':
            if self.paused:
                #self.animation.resume()
                self.text.remove()
                self.text = None
                if not self.isKivy: self.animation.event_source.start()
            else:
                #self.animation.pause()

                self.text = self.fig.text(0.5, 0.5, 'Paused!', dict(size=30, fontweight=800, color='white'),
                                          horizontalalignment='center', verticalalignment='center',
                                          path_effects=[peffects.withStroke(linewidth=4, foreground="black")])
                self.fig.canvas.draw()
                if not self.isKivy: self.animation.event_source.stop()

            self.paused = not self.paused

    def lineRedraw(self, ax, line, datax, datay, frame):
        # It is assumed there is a point of data for each frame
        line.set_data(datax, datay)
        if datax[frame] > ax.get_xlim()[1]:
            ax.set_xlim(ax.get_xlim()[0], 2 * ax.get_xlim()[1])
            self.redraw = True
        if datay[frame] < ax.get_ylim()[0]:
            ax.set_ylim(datay[frame] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05, ax.get_ylim()[1])
            self.redraw = True
        if datay[frame] > ax.get_ylim()[1]:
            ax.set_ylim(ax.get_ylim()[0], datay[frame] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05)
            self.redraw = True

    def update(self, frame):
        if self.duration is not None and self.callbackProgress and frame%10 == 0: print("{:5.2f} %".format(100*frame/self.frames))
        self.redraw = False # Will try to not redraw, only blit, except when necessary (bounds change, etc.)
        t0 = time.time()
        if self.frames != None:
            #if frame == self.frames-1:
            if self.QSystem.t >= self.duration:
                self.fig.canvas.mpl_disconnect(self.pauseEvent)
                if not self.isKivy: self.animation.event_source.stop()
                if self.text != None: self.text.remove()
                self.text = self.fig.text(0.5, 0.5, 'TIME!', dict(size=30, fontweight=800, color='white'),
                                          horizontalalignment='center', verticalalignment='center',
                                          path_effects=[peffects.withStroke(linewidth=4, foreground="black")])

                self.fig.canvas.draw()
        changes = []
        self.frame = frame

        for _ in range(max(1,int( self.dtAnim / self.dtSim)) if self.stepsPerFrame == 0 else self.stepsPerFrame):
            if self.imagdt: self.QSystem.evolveImagStep(self.dtSim)
            else:           self.QSystem.evolveStep(self.dtSim)

        if self.extraUpdates != None:
            for action in self.extraUpdates:
                updated = action()
                if updated != None:
                    changes.append(updated)

        if self.debugTime: print("Crank-Nicolson took:  {:12.8f}".format(time.time()-t0), " (s)", end=' <> ')

        self.QSystem.modSquared()
        if self.psiCutoff != None:
            self.QSystem.psiMod[self.QSystem.psiMod < self.psiCutoff] = None
        if self.scalePsi: self.datPsi.set_clim(vmax=np.max(self.QSystem.psiMod.T))

        self.datPsi.set_data(self.QSystem.psiMod.T)
        changes.append(self.datPsi)

        self.TList.append(self.QSystem.t)
        if self.showEnergy:
            self.KList.append(np.real(self.QSystem.kineticEnergy()))
            self.VList.append(np.real(self.QSystem.potentialEnergy()))
            self.EList.append(self.VList[-1] + self.KList[-1])

            self.lineRedraw(self.axEnergy, self.lineK, self.TList, self.KList, frame)  # lineK.set_data(TList, KList)
            self.lineRedraw(self.axEnergy, self.lineV, self.TList, self.VList, frame)  # lineV.set_data(TList, VList)
            self.lineRedraw(self.axEnergy, self.lineE, self.TList, self.EList, frame)  # lineE.set_data(TList, EList)

            changes.append(self.lineK)
            changes.append(self.lineV)
            changes.append(self.lineE)
        else:
            self.KList.append(None)
            self.VList.append(None)
            self.EList.append(None)

        if self.showNorm:
            self.NormList.append(np.real(self.QSystem.norm()))
            self.lineRedraw(self.axNorm, self.lineN, self.TList, self.NormList, frame)  # lineN.set_data(TList, NormList)
            changes.append(self.lineN)
        else:
            self.NormList.append(None)

        if self.showMomentum:
            self.QSystem.momentumSpace()
            if self.scaleMom: self.datMom.set_clim(vmax=np.max(self.QSystem.psiMod.T))
            self.datMom.set_data(self.QSystem.psiMod.T)
            changes.append(self.datMom)

        if self.showPotential:
            if self.updatePotential:
                mathPhysics.set2DMatrix(self.QSystem.X, self.QSystem.Y,
                                        self.QSystem.potential, self.potentialMat,
                                        t=self.QSystem.t, extra_param=self.QSystem.extra_param)
            if self.potentialCutoff != None:
                self.potentialMat[self.potentialMat < self.potentialCutoff] = None

            self.datPot.set_data(self.potentialMat.T)
            changes.append(self.datPot)

        if self.debugTime: print("+ Energy/Norm/etc took:  {:12.8f}".format(time.time()-t0), " (s)", end=' <> ')

        if self.redraw and self.isKivy:
            self.fig.canvas.draw()
            return []

        """if self.isKivy:
            self.fig.canvas.draw()
            return []"""
        # Blitting
        return changes

    def on_draw(self, event):    #Blitting doesn't seem to work. Won't plot anything. Not clear if it even works faster
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.fig.draw_artist(self.datPsi)
        if self.showPotential:
            self.fig.draw_artist(self.datPot)
        if self.showEnergy:
            self.fig.draw_artist(self.lineE)
            self.fig.draw_artist(self.lineK)
            self.fig.draw_artist(self.lineV)
        if self.showNorm:
            self.fig.draw_artist(self.lineN)
        if self.showMomentum:
            self.fig.draw_artist(self.datMom)


    def manualUpdate(self):
        if self.frame == self.frames: return
        t0 = time.time()

        """self.fig.canvas.restore_region(self.bg)

        changedArtists = self.update(self.frame)
        if self.redraw:
            self.fig.canvas.draw()
        else:
            for artistChanged in changedArtists:
                #print("Frame: ", self.frame, ",  artist: ", artistChanged)
                self.fig.draw_artist(artistChanged)
                #pass

            self.fig.canvas.blit(self.fig.bbox)

        self.fig.canvas.flush_events()"""
        self.update(self.frame)
        self.fig.canvas.draw()

        self.frame += 1
        #plt.pause(0.001)
        if self.debugTime: print("+ Ploting took:  {:12.8f}".format(time.time()-t0), " (s),   FPS = ", 1./(time.time()-t0))

    def resetSystem(self, QSystem):
        self.QSystem = QSystem
        self.frame = 0
        self.reset_lists()
        self.reset_plot()

    def saveAnimation(self, outputName, type="gif"):
        if type == "gif":
            writergif = matplotlib.animation.PillowWriter(fps=int(1/self.dtAnim))
            self.animation.save("./{}.gif".format(outputName), writer=writergif)
        elif type == "mp4":
            FFwriter = animation.FFMpegWriter(fps=int(1/self.dtAnim))
            self.animation.save('./{}.mp4'.format(outputName), writer=FFwriter)
        else:
            print("Output format not supported")






def lineRedraw(fig, ax, line, datax, datay, frame):
    #It is assumed there is a point of data for each frame
    line.set_data(datax, datay)
    scale = False
    if datax[frame] > ax.get_xlim()[1]:
        ax.set_xlim(ax.get_xlim()[0], 2 * ax.get_xlim()[1])
        scale = True
    if datay[frame] < ax.get_ylim()[0]:
        ax.set_ylim(datay[frame] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05, ax.get_ylim()[1])
        scale = True
    if datay[frame] > ax.get_ylim()[1]:
        ax.set_ylim(ax.get_ylim()[0], datay[frame] + (ax.get_ylim()[1]-ax.get_ylim()[0])*0.05)
        scale = True
    if scale:
        fig.canvas.draw()


#!!!!! WE REDRAW THE STUPID CANVAS MULTIPLE TIMES!!!!




if __name__ == "__main__":
    """ SIMPLEST GAME CODE
    #SIMPLEST PLOT. Marginally faster than general one at same conditions
    fig = plt.figure()
    
    potentialMat = np.ndarray((Nx+1, Ny+1), dtype=np.float64)
    
    mathPhysics.set2DMatrix(testSystem.X, testSystem.Y,
                            testSystem.potential, potentialMat,
                            t=testSystem.t, extra_param=testSystem.extra_param)
    testSystem.modSquared()
    
    plt.xlabel("x")
    plt.ylabel("y")
    im1 = plt.imshow(testSystem.psiMod.T, origin='lower', extent=(testSystem.x0, testSystem.xf, testSystem.y0, testSystem.yf),
               aspect='equal', cmap="viridis", interpolation=None, animated=True)
    im2 = plt.imshow(potentialMat.T, origin='lower', extent=(testSystem.x0, testSystem.xf, testSystem.y0, testSystem.yf),
               aspect='equal', cmap='gist_heat', alpha=0.3, interpolation=None, animated=False)
    
    plt.tight_layout()
    dt = 0.01
    # Updating data
    counter = 0
    t0 = 0.
    def update(frame):
        global counter, t0
        counter += 1
        print("Temps entre updates: ", time.time() - t0, "(s)")
        t0 = time.time()
        for _ in range(int(0.04/dt)):
            testSystem.evolveStep(dt)
    
    
        testSystem.modSquared()
        im1.set_data(testSystem.psiMod.T)
    
        mathPhysics.set2DMatrix(testSystem.X, testSystem.Y,
                                testSystem.potential, potentialMat,
                                t=testSystem.t, extra_param=testSystem.extra_param)
        im2.set_data(potentialMat.T)
    
        return (im1, im2)
    
    
    ani = FuncAnimation(fig, update, interval=1, blit=True)
    #writergif = matplotlib.animation.PillowWriter(fps=25) #fps=25
    #ani.save("falling.gif", writer=writergif)
    
    t0 = time.time()
    plt.show()
    print(counter/(time.time()-t0), " (FPS)")
    #plt.close()"""
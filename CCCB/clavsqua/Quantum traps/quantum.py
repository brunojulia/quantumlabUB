"""
Jan Albert Iglesias
10/02/2019
"""

import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from kivy.clock import Clock
from kivy.properties import NumericProperty, ObjectProperty, StringProperty
from kivy.core.window import Window
import warnings
from matplotlib import rc
import timeev as te
warnings.filterwarnings("error")

#Initialization of the plots:
fig_qua = Figure()
aqu = fig_qua.add_subplot(111)
aqu2 = aqu.twinx() #To have two different scales.

#Main Layout:
class Quantum(BoxLayout):
    time_label = StringProperty()
    but1 = StringProperty()
    but2 = StringProperty()
    but3 = StringProperty()
    language = StringProperty()

    def __init__(self, **kwargs):
        super(Quantum, self).__init__(**kwargs)

        #Language:
        self.language = "CAT"
        self.time_label = "Temps"
        self.but1 = "Sobre el pic"
        self.but2 = "En el pic"
        self.but3 = "Sota el pic"
        self.energy_label = "Energia"
        self.prob_label = "Probabilitat"

        "Quantum definitions"
        self.a = -20.
        self.b = 20.
        self.N = 1000
        self.deltax = (self.b - self.a)/float(self.N)

        te.factor = 10 #Factor for the applied potential.
        self.height_qua = 15 #Default
        self.sigma_qua = 2*te.factor/(np.sqrt(2*np.pi)*self.height_qua)
        self.mu_qua = 0
        self.xarr_qua = np.arange(self.a, self.b + self.deltax*0.1, self.deltax)
        self.m_qua = 1/(2*3.80995) #The value contains hbar^2.
        te.hbar = 4.136 #eV·fs (femtosecond)
        self.k_qua = 0.2 #Default
        self.xo = -2.3  #Default
        te.sigma0 = 0.4 #Default.
        self.oldtop2_qua = self.height_qua
        self.oldk2_qua = self.k_qua

        #Clock (Quantum):
        Clock.schedule_interval(self.psiupdate, 1/60)
        self.dtime0_qua = 1/30
        self.vel_qua = 1
        self.dtime_qua = self.vel_qua*self.dtime0_qua
        self.time_qua = 0.0
        self.tmax_qua = 10
        self.oldtime1_qua = self.time_qua + 1

        #Time
        self.timeslide_qua.disabled = True  #It is not allowed to move the time bar.

        #Plots (Quantum)
        self.canvas_qua = FigureCanvasKivyAgg(fig_qua)
        self.pot_qua, = aqu.plot(self.xarr_qua, te.pot(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua), 'k--', label = "V(x)")
        self.filled_pot_qua = aqu.fill_between(self.xarr_qua, te.pot(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua), color = (0.5,0.5,0.5,0.5))
        self.energy_qua, = aqu.plot([], [], '-.', color = 'g', label = "<E>")
        aqu.plot([], [], 'r-',  label = r'$|\Psi(x)|^{2}$') #Fake one, just for the legend.
        aqu.axis([-5, 5, 0, 2*te.factor])
        aqu.set_xlabel("x (" + u"\u212B" + ")")
        aqu.set_ylabel(self.energy_label + " (eV)", color = 'k')
        aqu.tick_params('y', colors = 'k')
        aqu.legend(loc=1)

            #The wavefunction is plotted in a different scale.
        self.psi_qua, = aqu2.plot(self.xarr_qua, np.abs(te.psi(self.xo, self.xarr_qua))**2, 'r-')
        self.filled_qua = aqu2.fill_between(self.xarr_qua, np.abs(te.psi(self.xo, self.xarr_qua))**2, color = (1,0,0,0.2))
        aqu2.axis([-5, 5, 0, 1.5])
        aqu2.set_ylabel(self.prob_label, color = 'k')
        aqu2.tick_params('y', colors = 'r')
        self.panel2.add_widget(self.canvas_qua)

        #First computations:
        self.demo1_qua_btn()


#Quantum functions:
    #Plotting:
    def plotpot(self):
        #It is only activated if some value changes.
        a = self.oldtop2_qua != self.height_qua
        b = self.oldk2_qua != self.k_qua

        if a or b:
            self.pot_qua.set_data(self.xarr_qua, te.pot(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua))
            self.filled_pot_qua.remove()
            self.filled_pot_qua = aqu.fill_between(self.xarr_qua, te.pot(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua), color = (0.5,0.5,0.5,0.5))

            self.oldtop2_qua = self.height_qua
            self.oldk2_qua = self.k_qua

    def plotpot1(self):
        #It is only activated if some value changes.
        a = self.oldtop2_qua != self.height_qua
        b = self.oldk2_qua != self.k_qua

        if a or b:
            self.pot_qua.set_data(self.xarr_qua, te.pot1(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua))
            self.filled_pot_qua.remove()
            self.filled_pot_qua = aqu.fill_between(self.xarr_qua, te.pot1(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua), color = (0.5,0.5,0.5,0.5))

            self.oldtop2_qua = self.height_qua
            self.oldk2_qua = self.k_qua

    def plotpsi(self, t):
        if self.oldtime1_qua != t:
            psit = np.abs(te.psiev(self.evalsbasis, self.coef_x_efuns, t))**2
            self.psi_qua.set_data(self.xarr_qua, psit)
            self.filled_qua.remove()
            self.filled_qua = aqu2.fill_between(self.xarr_qua, psit, color = (1,0,0,0.2))
            self.canvas_qua.draw()
            self.oldtime1_qua = t


    #Playing:
    def psiupdate(self, dt):
        self.time_qua = self.timeslide_qua.value + self.dtime_qua
        if self.time_qua >= self.tmax_qua:
            self.time_qua = 0
        self.timeslide_qua.value = self.time_qua
        self.plotpsi(self.time_qua)

    def reset(self):
        #Sets time to 0
        self.timeslide_qua.value = 0
        self.time_qua = 0
        self.tmax_qua = 10

    def demo1_qua_btn(self):
        self.reset()
        #This version does not compute anything. It just reads the precomputed matrices.
        self.coef_x_efuns = np.load("Demo1_qua/vecs.npy")
        self.evalsbasis = np.load("Demo1_qua/vals.npy")

        #Potential:
        self.height_qua = 20
        self.sigma_qua = 2*te.factor/(np.sqrt(2*np.pi)*self.height_qua)
        self.k_cla = 0.2
        self.plotpot1()

        #Plots the enrgy:
        self.energy = np.load("Demo1_qua/ene.npy")
        self.energy_qua.set_data(self.xarr_qua, 0*self.xarr_qua + self.energy)
        self.energy_qua.set_color('g')
        self.energy_qua.set_label("<E>")
        aqu.legend(loc=1)
        self.canvas_qua.draw()

    def demo2_qua_btn(self):
        self.reset()
        #This version does not compute anything. It just reads the precomputed matrices.
        self.coef_x_efuns = np.load("Demo2_qua/vecs.npy")
        self.evalsbasis = np.load("Demo2_qua/vals.npy")

        #Potential:
        self.height_qua = 10
        self.sigma_qua = 2*te.factor/(np.sqrt(2*np.pi)*self.height_qua)
        self.k_cla = 0.2
        self.plotpot()

        #Plots the enrgy:
        self.energy = np.load("Demo2_qua/ene.npy")
        self.energy_qua.set_data(self.xarr_qua, 0*self.xarr_qua + self.energy)
        self.energy_qua.set_color('g')
        self.energy_qua.set_label("<E>")
        aqu.legend(loc=1)
        self.canvas_qua.draw()

    def demo3_qua_btn(self):
        self.reset()
        #This version does not compute anything. It just reads the precomputed matrices.
        self.coef_x_efuns = np.load("Demo3_qua/vecs.npy")
        self.evalsbasis = np.load("Demo3_qua/vals.npy")

        #Potential:
        self.height_qua = 13
        self.sigma_qua = 2*te.factor/(np.sqrt(2*np.pi)*self.height_qua)
        self.k_cla = 0.2
        self.plotpot()

        psit = np.abs(te.psiev(self.evalsbasis, self.coef_x_efuns, 0))**2
        self.psi_qua.set_data(self.xarr_qua, psit)
        self.filled_qua.remove()
        self.filled_qua = aqu2.fill_between(self.xarr_qua, psit, color = (1,0,0,0.2))
        self.canvas_qua.draw()

        #Plots the enrgy:
        self.energy = np.load("Demo3_qua/ene.npy")
        self.energy_qua.set_data(self.xarr_qua, 0*self.xarr_qua + self.energy)
        self.energy_qua.set_color('g')
        self.energy_qua.set_label("<E>")
        aqu.legend(loc=1)
        self.canvas_qua.draw()

    def changelanguage(self):
        if self.language == "CAT":
            self.language = "ESP"
            self.time_label = "Tiempo"
            self.but1 = "Sobre el pico"
            self.but2 = "En el pico"
            self.but3 = "Bajo el pico"
            self.energy_label = "Energía"
            self.prob_label = "Probabilidad"
            aqu.set_ylabel(self.energy_label + " (eV)", color = 'k')
            aqu2.set_ylabel(self.prob_label, color = 'k')

        elif self.language == "ESP":
            self.language = "ENG"
            self.time_label = "Time"
            self.but1 = "Above the peak"
            self.but2 = "On the peak"
            self.but3 = "Beneath the peak"
            self.energy_label = "Energy"
            self.prob_label = "Probability"
            aqu.set_ylabel(self.energy_label + " (eV)", color = 'k')
            aqu2.set_ylabel(self.prob_label, color = 'k')

        elif self.language == "ENG":
            self.language = "CAT"
            self.time_label = "Temps"
            self.but1 = "Sobre el pic"
            self.but2 = "En el pic"
            self.but3 = "Sota el pic"
            self.energy_label = "Energia"
            self.prob_label = "Probabilitat"
            aqu.set_ylabel(self.energy_label + " (eV)", color = 'k')
            aqu2.set_ylabel(self.prob_label, color = 'k')



class quantumApp(App):
    def build(self):
        self.title = "Quantum traps"
        a = Quantum()
        return a

if __name__ == "__main__":
    Window.fullscreen = 'auto'
    quantumApp().run()

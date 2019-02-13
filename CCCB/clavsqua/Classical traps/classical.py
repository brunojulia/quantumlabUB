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
import rollball as rob
warnings.filterwarnings("error")

#Initialization of the plots:
fig_cla = Figure()
acl = fig_cla.add_subplot(111)

rob.g = 9.806

#Main Layout:
class Classical(BoxLayout):
    time_label = StringProperty()
    but1 = StringProperty()
    but2 = StringProperty()
    but3 = StringProperty()
    language = StringProperty()

    def __init__(self, **kwargs):
        super(Classical, self).__init__(**kwargs)

        #Language:
        self.language = "CAT"
        self.time_label = "Temps"
        self.but1 = "Sobre el pic"
        self.but2 = "En el pic"
        self.but3 = "Sota el pic"
        self.energy_label = "Energia"
        self.prob_label = "Probabilitat"

        "Classical definitions"
        self.time_cla = 0.
        self.height_cla = 0.3
        self.sigma_cla =  1./(np.sqrt(2*np.pi)*self.height_cla)
        self.mu_cla = 0
        rob.m = 5

        self.R = 0.2

        self.k_cla = 0.5
        self.tmax_cla = 10

        self.xo_cla = 1.5
        self.yin0 = np.array([self.xo_cla,0.0])  #Check if this is needed.
        self.timeslide_cla.disabled = True  #It is not allowed to move the time bar.

        self.xarr_cla = xarr_cla = np.arange(-20, 20 + (20 - (-20))/float(1000)*0.1, (20 - (-20))/float(1000)) #Why not?
        #It initially used xarr_qua. But since the programs were separated, it uses this. It is only for plotting purposes.

        #Flux controllers.
        self.oldtop2_cla = self.height_cla
        self.oldk2_cla = self.k_cla

        #Clock (Classical):
        self.time_cla = 0.
        self.dtime0_cla = 0.01 #Default time step (used by RK4 and as default playing velocity).

        self.dtime_cla = 1.2*self.dtime0_cla #Defines the playing velocity = 1.

        self.oldtime1_cla = self.time_cla + 1
        self.oldrow = -1

        #Plots (Classical):
        self.canvas_cla = FigureCanvasKivyAgg(fig_cla)
        self.panel1.add_widget(self.canvas_cla)

        acl.axis('scaled')
        acl.set_xlabel("x (m)")
        acl.set_ylabel("y (m)")
        acl.axis([-2.5, 2.5, 0 , 3])

        self.ground_cla, = acl.plot(self.xarr_cla, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla), 'k--')
        self.filled_cla = acl.fill_between(self.xarr_cla, 0, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla), color = (0.5,0.5,0.5,0.5))
        self.ballcm, = acl.plot([], [], 'ro', ms=1)
        self.ballperim, = acl.plot([], [], 'r-', lw=1)
        self.filled_ball_cla = acl.fill_between([], []) #Empty one. Needs to be here in order to use self.name.remove() later.
        self.balldots, = acl.plot([], [], 'ro', ms=1)
        self.E_cla, = acl.plot([],[], 'g-.', lw=1, label = "E")
        acl.legend(loc=1)

        #First computations:
        self.demo1_cla_btn()
        Clock.schedule_interval(self.ballupdate, self.dtime0_cla)

#Classical functions:
    #Changing parameters:
    def plotground(self):
        #It is only activated if some value changes.
        a = self.oldtop2_cla != self.height_cla
        b = self.oldk2_cla != self.k_cla

        if a or b:
            self.oldtop2_cla = self.height_cla
            self.oldk2_cla = self.k_cla

            #Changes and plots the new ground.
            self.ground_cla.set_data(self.xarr_cla, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla))
            self.filled_cla.remove()
            self.filled_cla = acl.fill_between(self.xarr_cla, 0, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla),
            color = (0.5,0.5,0.5,0.5))



    #Plotting.
    def plotball_0(self):
        x = self.supermatrix_cla[0, 1]
        XXcm = rob.xcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)
        YYcm = rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)

        gamma = np.arange(0, 2*np.pi + 0.5*0.02/self.R, 0.02/self.R)
        half_gamma = np.arange(0, np.pi, 0.02/self.R)
        XXr = XXcm + self.R*np.cos(gamma)
        YYr = YYcm + self.R*np.sin(gamma)

        Xdots = []
        Ydots = []
        for i in [-1./2., 0., 1./2., 1.]:
            Xdots.append(XXcm + self.R*np.sin(self.angle[0] + i*np.pi)/2.)
            Ydots.append(YYcm + self.R*np.cos(self.angle[0] + i*np.pi)/2.)

        self.ballperim.set_data(XXr, YYr)
        self.filled_ball_cla.remove()
        self.filled_ball_cla = acl.fill_between(XXcm + self.R*np.cos(half_gamma), YYcm - self.R*np.sin(half_gamma),
        YYcm + self.R*np.sin(half_gamma), color = (1,0,0,0.2))
        self.ballcm.set_data(XXcm, YYcm)
        self.balldots.set_data(Xdots, Ydots)
        self.canvas_cla.draw()


    def plotball(self, t):
        if self.oldtime1_cla != t:
            if t >= self.supermatrix_cla[-1,0]:
                row = - 1

            else:
                row = int(t/self.dtime0_cla)

            if self.oldrow != row:
                x = self.supermatrix_cla[row, 1]
                XXcm = rob.xcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)
                YYcm = rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)

                gamma = np.arange(0, 2*np.pi + 0.5*0.02/self.R, 0.02/self.R)
                half_gamma = np.arange(0, np.pi, 0.02/self.R)
                XXr = XXcm + self.R*np.cos(gamma)
                YYr = YYcm + self.R*np.sin(gamma)

                Xdots = []
                Ydots = []
                for i in [-1./2., 0., 1./2., 1.]:
                    Xdots.append(XXcm + self.R*np.sin(self.angle[row] + i*np.pi)/2.)
                    Ydots.append(YYcm + self.R*np.cos(self.angle[row] + i*np.pi)/2.)

                self.ballperim.set_data(XXr, YYr)
                self.filled_ball_cla.remove()
                self.filled_ball_cla = acl.fill_between(XXcm + self.R*np.cos(half_gamma), YYcm - self.R*np.sin(half_gamma),
                YYcm + self.R*np.sin(half_gamma), color = (1,0,0,0.2))
                self.ballcm.set_data(XXcm, YYcm)
                self.balldots.set_data(Xdots, Ydots)
                self.canvas_cla.draw()

                self.oldrow = row
            self.oldtime1_cla = t

    def plotE_cla(self):
        #Plots the initial energy as a height (maximum height).
        x = np.arange(-2.6, 2.6, 0.1)
        y = self.Eheight + 0*x
        self.E_cla.set_data(x,y)
        self.canvas_cla.draw()

    #Playing.
    def ballupdate(self, dt):
        self.time_cla = self.timeslide_cla.value + self.dtime_cla

        #It starts again if the time reaches the top.
        if self.time_cla >= self.tmax_cla:
            self.time_cla = 0

        self.timeslide_cla.value = self.time_cla
        self.plotball(self.time_cla)

    def reset_cla(self):
        #Sets time to 0
        self.timeslide_cla.value = 0
        self.time_cla = 0

    def demo1_cla_btn(self):
        self.reset_cla()

        self.height_cla = 0.8
        self.sigma_cla =  1./(np.sqrt(2*np.pi)*self.height_cla)
        self.k_cla = 0.8
        self.plotground()

        #This version does not compute anything. It just reads the precomputed matrices.
        self.supermatrix_cla = np.load("Demo1_cla/super.npy")
        self.angle = np.load("Demo1_cla/ang.npy")

        #Plots the maximum height:
        self.Eheight = np.load("Demo1_cla/ene.npy")
        self.plotE_cla()

    def demo2_cla_btn(self):
        self.reset_cla()

        self.height_cla = 0.9
        self.sigma_cla =  1./(np.sqrt(2*np.pi)*self.height_cla)
        self.k_cla = 0.8
        self.plotground()

        #This version does not compute anything. It just reads the precomputed matrices.
        self.supermatrix_cla = np.load("Demo2_cla/super.npy")
        self.angle = np.load("Demo2_cla/ang.npy")

        #Plots the maximum height:
        self.Eheight = np.load("Demo2_cla/ene.npy")
        self.plotE_cla()

    def demo3_cla_btn(self):
        self.reset_cla()

        self.height_cla = 1
        self.sigma_cla =  1./(np.sqrt(2*np.pi)*self.height_cla)
        self.k_cla = 0.8
        self.plotground()

        #This version does not compute anything. It just reads the precomputed matrices.
        self.supermatrix_cla = np.load("Demo3_cla/super.npy")
        self.angle = np.load("Demo3_cla/ang.npy")

        #Plots the maximum height:
        self.Eheight = np.load("Demo3_cla/ene.npy")
        self.plotE_cla()

    def changelanguage(self):
        if self.language == "CAT":
            self.language = "ESP"
            self.time_label = "Tiempo"
            self.but1 = "Sobre el pico"
            self.but2 = "En el pico"
            self.but3 = "Bajo el pico"

        elif self.language == "ESP":
            self.language = "ENG"
            self.time_label = "Time"
            self.but1 = "Above the peak"
            self.but2 = "On the peak"
            self.but3 = "Beneath the peak"

        elif self.language == "ENG":
            self.language = "CAT"
            self.time_label = "Temps"
            self.but1 = "Sobre el pic"
            self.but2 = "En el pic"
            self.but3 = "Sota el pic"


class classicalApp(App):
    def build(self):
        self.title = "Classical traps"
        a = Classical()
        return a

if __name__ == "__main__":
    Window.fullscreen = 'auto'
    classicalApp().run()

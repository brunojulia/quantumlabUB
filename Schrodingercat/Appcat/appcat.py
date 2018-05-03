"""
Jan Albert Iglesias
3/5/2018
"""

import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from kivy.clock import Clock
from kivy.properties import NumericProperty
from  kivy.uix.popup import Popup
from matplotlib import rc
import rollball as rob
import timeev as te

fig_cla = Figure()
acl = fig_cla.add_subplot(111)
fig_qua = Figure()
aqu = fig_qua.add_subplot(111)
fig_norm = Figure()
anorm = fig_norm.add_subplot(111)
fig_ene = Figure()
aen = fig_ene.add_subplot(111)

rob.g = 9.806

"""
xvec = np.arange(-5,5,0.01)
plt.plot(xvec, rob.fground(0, 1./(np.sqrt(2*np.pi)*1), 1.3, xvec), label = "0")
plt.plot(xvec, rob.dfground(0, 1./(np.sqrt(2*np.pi)*1), 1.3, xvec), label = "1")
plt.plot(xvec, rob.d2fground(0, 1./(np.sqrt(2*np.pi)*1), 1.3, xvec), label = "2")
plt.plot(xvec, rob.d3fground(0, 1./(np.sqrt(2*np.pi)*1), 1.3, xvec), label = "3")
plt.legend()
plt.show()
"""

class Appcat(BoxLayout):
    time_qua = NumericProperty()
    tmax_qua = NumericProperty()
    tmax_cla = NumericProperty()
    xo = NumericProperty()


    def __init__(self, **kwargs):
        super(Appcat, self).__init__(**kwargs)

        #Quantum definitions:
        self.a = -15.
        self.b = 15.
        self.N = 900
        self.deltax = (self.b - self.a)/float(self.N)

        self.sigma_qua = 1./(np.sqrt(2*np.pi)*self.heightslide_qua.value)
        self.mu_qua = 0
        self.xarr_qua = np.arange(self.a, self.b + self.deltax*0.1, self.deltax)
        self.m_qua = 1
        self.masslide_qua.value = self.m_qua
        self.k_qua = 1
        self.kslide_qua.value = self.k_qua
        self.xo = 0
        self.poslide_qua.value = self.xo

        self.switch1_qua = "off"
        self.oldtop1_qua = self.heightslide_qua.value + 1
        self.oldtop2_qua = self.heightslide_qua.value
        self.oldk1_qua = self.kslide_qua.value + 1
        self.oldk2_qua = self.kslide_qua.value
        self.oldmass1_qua = self.masslide_qua.value + 1
        self.oldmass2_qua = self.masslide_qua.value
        self.oldxo1_qua = self.poslide_qua.value + 1
        self.oldxo2_qua = self.poslide_qua.value

        self.startstopbut_qua.background_down = "playblue.png"
        self.startstopbut_qua.background_normal = "play.png"

        #Classical definitions:
        self.time_cla = 0.
        self.heightslide_cla.value = 0.3
        self.sigma_cla =  1./(np.sqrt(2*np.pi)*self.heightslide_cla.value)
        self.mu_cla = 0
        self.m_cla = 10
        self.masslide_cla.value = self.m_cla
        self.rho = 200
        self.R = (3*self.m_cla/(4*np.pi*self.rho))**(1/3.)
        rob.m = self.m_cla
        self.k_cla = 0.5
        self.kslide_cla.value = self.k_cla
        self.yin0 = np.array([1.5,0.0])
        self.yin = self.yin0
        self.poslide_cla.value = 1.5
        self.xarr_cla = self.xarr_qua

        self.switch1_cla = "off"
        self.oldtop1_cla = self.heightslide_cla.value + 1
        self.oldtop2_cla = self.heightslide_cla.value
        self.oldk1_cla = self.kslide_cla.value +1
        self.oldk2_cla = self.kslide_cla.value
        self.oldmass1_cla = self.masslide_cla.value +1
        self.oldmass2_cla = self.masslide_cla.value
        self.oldxo1_cla = self.poslide_cla.value + 1
        self.oldxo2_cla = self.poslide_cla.value

        self.startstopbut_cla.background_down = "playblue.png"
        self.startstopbut_cla.background_normal = "play.png"

        self.supermatrix_cla = np.array([[self.time_cla, self.yin[0], self.yin[1]]]) #3 columns: time, x and xdot
        self.angle = np.array([[-np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))]])
        self.perimeter = 0.

        #Energy. 4 columns; translational, rotational, potential and total.
        trans = 0.5*self.m_cla*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2 + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2)
        rot = 0.2*self.m_cla*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])/self.R - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))**2*self.yin[1]**2
        pot = self.m_cla*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])
        tot = trans + rot + pot
        self.energy = np.array([[trans, rot, pot, tot]])

        #Clock (Quantum):
        Clock.schedule_interval(self.psiupdate, 1/60)
        self.dtime0_qua = 1/30
        self.dtime_qua = self.dtime0_qua
        self.time_qua = 0.0
        self.tmax_qua = 10
        self.oldtime1_qua = self.time_qua + 1
        self.oldtime2_qua = self.time_qua
        self.vel_qua = 1
        self.vel_btn_qua.text = "1x"

        #Clock (Classical):
        Clock.schedule_interval(self.ballupdate, 1/30)
        self.tmax_cla = 3
        self.deltat = 0.01
        self.dtime0_cla = self.deltat
        self.dtime_cla = self.dtime0_cla
        self.tsteps_cla = int(self.tmax_cla/self.deltat)
        self.oldtime1_cla = self.time_cla + 1
        self.oldrow = -1
        self.deltax_cla = 0.001
        self.vel_cla = 1
        self.vel_btn_cla.text = "1x"

        #Norm
        self.norm = []
        self.timevec_qua = []
        psit = np.abs(te.psi(self.xo, self.xarr_qua))**2
        self.norm.append(self.deltax*(np.sum(psit) - psit[0]/2. - psit[self.N]/2.))
        self.timevec_qua.append(self.time_qua)


        #Plots:
        self.canvas_cla = FigureCanvasKivyAgg(fig_cla)
        self.ground_cla, = acl.plot(self.xarr_cla, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla), 'k--')
        self.ballcm, = acl.plot([], [], 'ko', ms=1)
        self.ballperim, = acl.plot([], [], 'k-', lw=1)
        self.balldots, = acl.plot([], [], 'ko', ms=1)
        self.filled = acl.fill_between(self.xarr_cla, 0, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla), color = (0.5,0.5,0.5,0.5))
        acl.axis('scaled')
        acl.set_xlabel("x (m)")
        acl.set_ylabel("y (m)")
        self.panel1.add_widget(self.canvas_cla)
        acl.axis([-2.5, 2.5, 0 , 3])

        self.canvas_qua = FigureCanvasKivyAgg(fig_qua)
        self.pot_qua, = aqu.plot(self.xarr_qua, te.pot(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua), 'g--', label = "V(x)")
        self.psi_qua, = aqu.plot(self.xarr_qua, np.abs(te.psi(self.xo, self.xarr_qua))**2, 'r-', label = r'$|\Psi(x)|^{2}$')
        aqu.axis([-5, 5, 0, 2])
        aqu.set_xlabel("x")
        aqu.legend(loc=1)
        self.panel2.add_widget(self.canvas_qua)

        #First computations:
        self.computevolution()
        self.computevolution_cla()
        self.plotball_0()

    def computevolution(self):
        #Just happens if some value changes:
        a = self.oldtop1_qua != self.heightslide_qua.value
        b = self.oldk1_qua != self.kslide_qua.value
        c = self.oldmass1_qua != self.masslide_qua.value
        d = self.oldxo1_qua != self.poslide_qua.value

        if a or b or c or d:
            print("i'm working!")
            #Definitions:
            Nbasis=150
            coefs = np.zeros(shape = (Nbasis + 1, 1), dtype=complex)
            self.coef_x_efuns = np.zeros(shape = (self.N + 1, Nbasis + 1), dtype=complex)
            self.evalsbasis = np.zeros(shape = (Nbasis + 1, 1))
            psivec = te.psi(self.xo, self.xarr_qua)

            #Solving:
            evals, efuns = te.srindwall(self.a, self.b, self.N, self.m_qua, te.pot, self.mu_qua, self.sigma_qua, self.k_qua)
            for j in range(0, Nbasis+1, 1):
                prod = np.conjugate(psivec)*efuns[:,j]
                coefs[j] = self.deltax*(np.sum(prod) - prod[0]/2. - prod[Nbasis]/2.)
                self.coef_x_efuns[:,j] = coefs[j]*efuns[:,j]
                self.evalsbasis[j] = evals[j]

            self.oldtop1_qua = self.heightslide_qua.value
            self.oldk1_qua = self.kslide_qua.value
            self.oldmass1_qua = self.masslide_qua.value
            self.oldxo1_qua = self.poslide_qua.value
            print("I've worked.")

    def plotpsi(self, t):
        if self.oldtime1_qua != t:
            psit = np.abs(te.psiev(self.evalsbasis, self.coef_x_efuns, t))**2
            self.psi_qua.set_data(self.xarr_qua, psit)
            self.canvas_qua.draw()
            self.oldtime1_qua = t

            if t > self.oldtime2_qua + 1:
                self.norm.append(self.deltax*(np.sum(psit) - psit[0]/2. - psit[self.N]/2.))
                self.timevec_qua.append(t)
                oldtime2_qua = t

    def plotpot(self):
        #It is only activated if some value changes.
        a = self.oldtop2_qua != self.heightslide_qua.value
        b = self.oldk2_qua != self.kslide_qua.value

        if a or b:
            self.sigma_qua = 1./(np.sqrt(2*np.pi)*self.heightslide_qua.value)
            self.k_qua = self.kslide_qua.value
            self.pot_qua.set_data(self.xarr_qua, te.pot(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua))
            self.canvas_qua.draw()

            self.reset()

            self.oldtop2_qua = self.heightslide_qua.value
            self.oldk2_qua = self.kslide_qua.value

    def change_mass(self):
        if self.oldmass2_qua != self.masslide_qua.value:
            self.m_qua = self.masslide_qua.value
            self.oldmass2_qua = self.masslide_qua.value
            self.reset()

    def change_xo(self, xoo):
        if self.oldxo2_qua != self.poslide_qua.value:
            self.reset()
            self.oldxo2_qua = self.poslide_qua.value
            self.xo = self.poslide_qua.value
            self.psi_qua.set_data(self.xarr_qua, np.abs(te.psi(xoo, self.xarr_qua))**2)
            self.canvas_qua.draw()

    def psiupdate(self, dt):
        if self.switch1_qua == "on":
            self.time_qua = self.timeslide_qua.value + self.dtime_qua
            if self.time_qua >= self.tmax_qua:
                self.tmax_qua = self.time_qua
            self.timeslide_qua.value = self.time_qua
            self.plotpsi(self.time_qua)

    def start_stop(self):
        if self.switch1_qua == "off":
            self.startstopbut_qua.background_normal = "pause.png"
            self.startstopbut_qua.background_down = "playblue.png"
            self.switch1_qua = "on"
        elif self.switch1_qua == "on":
            self.startstopbut_qua.background_normal = "play.png"
            self.startstopbut_qua.background_down = "pauseblue.png"
            self.switch1_qua = "off"

    def reset(self):
        #Sets time to 0
        self.timeslide_qua.value = 0
        self.time_qua = 0
        self.tmax_qua = 10
        self.switch1_qua = "off"
        self.startstopbut_qua.background_normal = "play.png"
        self.startstopbut_qua.background_down = "pauseblue.png"
        self.norm = []
        self.timevec_qua = []
        psit = np.abs(te.psi(self.xo, self.xarr_qua))**2
        self.norm.append(self.deltax*(np.sum(psit) - psit[0]/2. - psit[self.N]/2.))
        self.timevec_qua.append(self.time_qua)

    def reset_btn(self):
        self.reset()
        self.plotpsi(0)

    def velocity_btn(self):
        self.vel_qua = self.vel_qua + 1
        if self.vel_qua == 5:
            self.vel_qua = 1
        self.dtime_qua = self.vel_qua*self.dtime0_qua
        self.vel_btn_qua.text = str(self.vel_qua) + "x"

#Classical functions:
    def extend(self, t):
        x0 = self.yin[0]
        self.yin = rob.RK4(self.R, self.mu_cla, self.sigma_cla, self.k_cla, t, self.deltat, self.yin, rob.frollingball)
        self.supermatrix_cla = np.concatenate((self.supermatrix_cla, [[t, self.yin[0], self.yin[1]]]))

        self.perimeter = self.perimeter + rob.trapezoidal(self.mu_cla, self.sigma_cla, self.k_cla, x0, self.yin[0], self.deltax_cla, rob.groundperim)
        theta = self.perimeter/self.R
        beta = np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))
        self.angle = np.concatenate((self.angle, [[theta - beta]]))

        trans = 0.5*self.m_cla*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2 + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2)
        rot = 0.2*self.m_cla*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])/self.R - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))**2*self.yin[1]**2
        pot = self.m_cla*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])
        tot = trans + rot + pot
        self.energy = np.concatenate((self.energy, [[trans, rot, pot, tot]]))

    def computevolution_cla(self):
        #Just happens if some value changes:
        a = self.oldtop1_cla != self.heightslide_cla.value
        b = self.oldk1_cla != self.kslide_cla.value
        c = self.oldmass1_cla != self.masslide_cla.value
        d = self.oldxo1_cla != self.poslide_cla.value

        if a or b or c or d:
            #Erases the previously computed values.
            self.supermatrix_cla = np.array([[self.time_cla, self.yin[0], self.yin[1]]]) #3 columns: time, x and xdot
            self.angle = np.array([[-np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))]])
            self.perimeter = 0.
            self.tsteps_cla = int(self.tmax_cla/self.deltat)
            self.energy = np.zeros(shape=(1,4))
            self.energy[0,0] = 0.5*self.m_cla*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2 + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2)
            self.energy[0,1] = 0.2*self.m_cla*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])/self.R - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))**2*self.yin[1]**2
            self.energy[0,2] = self.m_cla*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])
            self.energy[0,3] = sum(self.energy[0,:])

            print("classical working")
            for i in range(1, self.tsteps_cla, 1):
                t = self.deltat*i
                self.extend(t)

            self.oldtop1_cla = self.heightslide_cla.value
            self.oldk1_cla = self.kslide_cla.value
            self.oldmass1_cla = self.masslide_cla.value
            self.oldxo1_cla = self.poslide_cla.value
            print("classical worked!")
            print(self.energy)


    def plotground(self):
        #It is only activated if some value changes.
        a = self.oldtop2_cla != self.heightslide_cla.value
        b = self.oldk2_cla != self.kslide_cla.value

        if a or b:
            self.sigma_cla = 1./(np.sqrt(2*np.pi)*self.heightslide_cla.value)
            self.k_cla = self.kslide_cla.value
            self.ground_cla.set_data(self.xarr_cla, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla))
            self.filled.remove()
            self.filled = acl.fill_between(self.xarr_cla, 0, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla), color = (0.5,0.5,0.5,0.5))
            self.canvas_cla.draw()

            self.oldtop2_cla = self.heightslide_cla.value
            self.oldk2_cla = self.kslide_cla.value

            self.reset_cla()
            self.angle[0] = -np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))
            self.plotball_0()


    def plotball_0(self):
        x = self.supermatrix_cla[0, 1]
        XXcm = rob.xcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)
        YYcm = rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)

        gamma = np.arange(0, 2*np.pi + 0.5*0.05/self.R, 0.05/self.R)
        XXr = XXcm + self.R*np.sin(gamma)
        YYr = YYcm + self.R*np.cos(gamma)

        Xdots = []
        Ydots = []
        for i in [-1./2., 0., 1./2., 1.]:
            Xdots.append(XXcm + self.R*np.sin(self.angle[0] + i*np.pi)/2.)
            Ydots.append(YYcm + self.R*np.cos(self.angle[0] + i*np.pi)/2.)

        self.ballperim.set_data(XXr, YYr)
        self.ballcm.set_data(XXcm, YYcm)
        self.balldots.set_data(Xdots, Ydots)
        self.canvas_cla.draw()


    def plotball(self, t):
        if self.oldtime1_cla != t:
            if t >= self.supermatrix_cla[-1,0]:
                row = - 1

            else:
                row = int(t/self.deltat)

            if self.oldrow != row:
                x = self.supermatrix_cla[row, 1]
                XXcm = rob.xcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)
                YYcm = rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, x)

                gamma = np.arange(0, 2*np.pi + 0.5*0.05/self.R, 0.05/self.R)
                XXr = XXcm + self.R*np.sin(gamma)
                YYr = YYcm + self.R*np.cos(gamma)

                Xdots = []
                Ydots = []
                for i in [-1./2., 0., 1./2., 1.]:
                    Xdots.append(XXcm + self.R*np.sin(self.angle[row] + i*np.pi)/2.)
                    Ydots.append(YYcm + self.R*np.cos(self.angle[row] + i*np.pi)/2.)

                self.ballperim.set_data(XXr, YYr)
                self.ballcm.set_data(XXcm, YYcm)
                self.balldots.set_data(Xdots, Ydots)
                self.canvas_cla.draw()

                self.oldrow = row
            self.oldtime1_cla = t

    def ballupdate(self, dt):
        if self.switch1_cla == "on":
            self.time_cla = self.timeslide_cla.value + self.dtime_cla

            #Supermatrix gets extended if the timer reaches the top.
            if self.time_cla >= self.tmax_cla:
                t = self.supermatrix_cla[-1, 0]
                self.tmax_cla = self.time_cla

                while t <= self.time_cla:
                    t = t + self.deltat
                    self.extend(t)

            self.timeslide_cla.value = self.time_cla
            self.plotball(self.time_cla)


    def start_stop_cla(self):
        if self.switch1_cla == "off":
            self.startstopbut_cla.background_normal = "pause.png"
            self.startstopbut_cla.background_down = "playblue.png"
            self.switch1_cla = "on"
        elif self.switch1_cla == "on":
            self.startstopbut_cla.background_normal = "play.png"
            self.startstopbut_cla.background_down = "pauseblue.png"
            self.switch1_cla = "off"

    def reset_cla(self):
        #Sets time to 0
        self.switch1_cla = "off"
        self.timeslide_cla.value = 0
        self.time_cla = 0
        self.tmax_cla = 3
        self.startstopbut_cla.background_normal = "play.png"
        self.startstopbut_cla.background_down = "pauseblue.png"
        self.yin = self.yin0


    def reset_cla_btn(self):
        self.reset_cla()
        self.plotball(0)

    def change_mass_cla(self):
        if self.oldmass2_cla != self.masslide_cla.value:
            self.m_cla = self.masslide_cla.value
            rob.m = self.m_cla
            self.oldmass2_cla = self.masslide_cla.value

            #Changes the radius as well.
            self.R = (3*self.m_cla/(4*np.pi*self.rho))**(1/3.)

            self.reset_cla()
            self.plotball_0()

    def change_xo_cla(self):
        if self.oldxo2_cla != self.poslide_cla.value:
            self.oldxo2_cla = self.poslide_cla.value
            self.xo_cla = self.poslide_cla.value
            self.yin0[0] = self.xo_cla
            self.reset_cla()
            self.supermatrix_cla[0, 1] = self.xo_cla
            self.angle[0] = -np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))
            self.plotball_0()

    def velocity_cla_btn(self):
        self.vel_cla = self.vel_cla + 1
        if self.vel_cla == 5:
            self.vel_cla = 1
        self.dtime_cla = self.vel_cla*self.dtime0_cla
        self.vel_btn_cla.text = str(self.vel_cla) + "x"


class Normpopup(Popup):
    def __init__(self, appcat):
        super(Normpopup, self).__init__()
        self.a = appcat
        self.canvas_norm = FigureCanvasKivyAgg(fig_norm)
        self.ids.Panel3_id.add_widget(self.canvas_norm)
        anorm.clear()
        anorm.set_ylim([0, 1.2])
        anorm.set_xlim([0, self.a.tmax_qua])
        anorm.set_xlabel("t (" + u"\u0127" + "/J)")
        anorm.set_ylabel(r'$\int_{-\infty}^{ \infty} |\Psi(t,x)|^{2}$' + "dx")
        self.event = Clock.schedule_interval(self.update, 0.3)
        self.normplot, = anorm.plot([], [], 'g-')
        self.txt = anorm.set_title("")
        self.txt.set_text("t" + r'$_{max}$' + " = " + '%1.1f' %self.a.timevec_qua[-1] + " " + u"\u0127" + "/J"
        + "      norm = " + str(self.a.norm[-1]).ljust(7)[:7])

    def update(self, dt):
        self.normplot.set_data(self.a.timevec_qua, self.a.norm)
        self.canvas_norm.draw()
        anorm.set_xlim([0, self.a.tmax_qua])
        self.txt.set_text("t" + r'$_{max}$' + " = " + '%1.1f' %self.a.timevec_qua[-1] + " " + u"\u0127" + "/J"
        + "      norm = " + str(self.a.norm[-1]).ljust(7)[:7])

    def close(self):
        Clock.unschedule(self.event)


class Energypopup(Popup):
    def __init__(self, appcat):
        super(Energypopup, self).__init__()
        self.a = appcat
        self.canvas_ene = FigureCanvasKivyAgg(fig_ene)
        self.ids.Panel4_id.add_widget(self.canvas_ene)
        aen.clear()
        aen.set_ylim([0, max(self.a.energy[:, 3]) + 10])
        aen.set_xlim([0, self.a.tmax_cla])
        aen.set_xlabel("t (s)")
        aen.set_ylabel("E (J)")
        self.event = Clock.schedule_interval(self.update, 0.3)
        self.totplot, = aen.plot([], [], label = "Total")
        self.transplot, = aen.plot([], [], label = "Translational")
        self.rotplot, = aen.plot([], [], label = "Rotational")
        self.potplot, = aen.plot([], [], label = "Potential")
        aen.legend(loc=5)
        self.txt = aen.set_title("")
        self.txt.set_text("t" + r'$_{max}$' + " = " + '%1.1f' %self.a.supermatrix_cla[-1, 0] + " s"
        + "      E" + r'$_{total}$' + "= " + str(self.a.energy[-1, 3]).ljust(7)[:7])

    def update(self, dt):
        self.totplot.set_data(self.a.supermatrix_cla[:, 0], self.a.energy[:, 3])
        self.transplot.set_data(self.a.supermatrix_cla[:, 0], self.a.energy[:, 0])
        self.rotplot.set_data(self.a.supermatrix_cla[:, 0], self.a.energy[:, 1])
        self.potplot.set_data(self.a.supermatrix_cla[:, 0], self.a.energy[:, 2])
        aen.set_ylim([0, max(self.a.energy[:, 3]) + 10])
        aen.set_xlim([0, self.a.tmax_cla])
        self.txt.set_text("t" + r'$_{max}$' + " = " + '%1.1f' %self.a.supermatrix_cla[-1, 0] + " s"
        + "      E" + r'$_{total}$' + "= " + str(self.a.energy[-1, 3]).ljust(7)[:7])
        self.canvas_ene.draw()

    def close(self):
        Clock.unschedule(self.event)


class appcatApp(App):
    def build(self):
        self.title = "Main App"
        a = Appcat()
        return a

if __name__ == "__main__":
    appcatApp().run()

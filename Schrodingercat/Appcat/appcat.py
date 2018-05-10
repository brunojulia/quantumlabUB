"""
Jan Albert Iglesias
6/5/2018
"""

import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from kivy.clock import Clock
from kivy.properties import NumericProperty, ObjectProperty
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

#Main Layout:
class Appcat(BoxLayout):
    time_qua = NumericProperty()
    tmax_qua = NumericProperty()
    tmax_cla = NumericProperty()
    xo = NumericProperty()
    progressbar_cla = ObjectProperty()


    def __init__(self, **kwargs):
        super(Appcat, self).__init__(**kwargs)

        #Quantum definitions:
        self.a = -15.
        self.b = 15.
        self.N = 900
        self.deltax = (self.b - self.a)/float(self.N)

        te.sigma0 = 1
        self.sigmaslide_qua.value = te.sigma0
        self.sigma_qua = 1./(np.sqrt(2*np.pi)*self.heightslide_qua.value)
        self.mu_qua = 0
        self.xarr_qua = np.arange(self.a, self.b + self.deltax*0.1, self.deltax)
        self.m_qua = 1
        self.k_qua = 1
        self.kslide_qua.value = self.k_qua
        self.xo = 0
        self.poslide_qua.value = self.xo
        te.p0 = 0
        self.velslide_qua.value = te.p0

        self.switch1_qua = "off"
        self.oldtop1_qua = self.heightslide_qua.value + 1
        self.oldtop2_qua = self.heightslide_qua.value
        self.oldk1_qua = self.kslide_qua.value + 1
        self.oldk2_qua = self.kslide_qua.value
        self.oldsigma1_qua = self.sigmaslide_qua.value + 1
        self.oldsigma2_qua = self.sigmaslide_qua.value
        self.oldxo1_qua = self.poslide_qua.value + 1
        self.oldxo2_qua = self.poslide_qua.value
        self.oldvel1_qua = self.velslide_qua.value + 1
        self.oldvel2_qua = self.velslide_qua.value

        self.startstopbut_qua.background_down = "playblue.png"
        self.startstopbut_qua.background_normal = "play.png"

        #Classical definitions:
        self.time_cla = 0.
        self.heightslide_cla.value = 0.3
        self.sigma_cla =  1./(np.sqrt(2*np.pi)*self.heightslide_cla.value)
        self.mu_cla = 0
        self.m_cla = 10
        rob.m = self.m_cla
        self.R = 0.2
        self.radiusslide_cla.value = self.R
        self.k_cla = 0.5
        self.kslide_cla.value = self.k_cla
        self.yin0 = np.array([1.5,0.0])
        self.poslide_cla.value = 1.5
        self.velslide_cla.value = 0.0
        self.xarr_cla = self.xarr_qua

        self.switch1_cla = "off"
        self.oldtop1_cla = self.heightslide_cla.value + 1
        self.oldtop2_cla = self.heightslide_cla.value
        self.oldk1_cla = self.kslide_cla.value +1
        self.oldk2_cla = self.kslide_cla.value
        self.oldradius1_cla = self.radiusslide_cla.value +1
        self.oldradius2_cla = self.radiusslide_cla.value
        self.oldxo1_cla = self.poslide_cla.value + 1
        self.oldxo2_cla = self.poslide_cla.value
        self.oldvel1_cla = self.velslide_cla.value + 1
        self.oldvel2_cla = self.velslide_cla.value
        self.rKswitch.active = False
        self.rKFswitch.active = True

        self.startstopbut_cla.background_down = "playblue.png"
        self.startstopbut_cla.background_normal = "play.png"


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
        self.tmax_cla = 3.0
        self.dtime0_cla = 0.01 #For storing data.
        self.dtime_cla = self.dtime0_cla
        self.oldtime1_cla = self.time_cla + 1
        self.oldrow = -1
        self.deltax_cla = 0.001
        self.vel_cla = 1
        self.vel_btn_cla.text = "1x"

        #Runge-Kutta-Fehlberg:
        rob.eps = 0.000001
        rob.h = 1
        self.yarr = np.zeros(shape=(2,2))
        self.tvec = np.zeros(shape=(2,1))

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
        self.E_cla, = acl.plot([],[], 'r--', lw=1)
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
        self.computed_cla = False
        self.computevolution()
        self.computevolution_cla()
        self.plotball_0()
        self.plotE_cla()

    def computevolution(self):
        #Just happens if some value changes:
        a = self.oldtop1_qua != self.heightslide_qua.value
        b = self.oldk1_qua != self.kslide_qua.value
        c = self.oldsigma1_qua != self.sigmaslide_qua.value
        d = self.oldxo1_qua != self.poslide_qua.value
        e = self.oldvel1_qua != self.velslide_qua.value

        if a or b or c or d or e:
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
            self.oldsigma1_qua = self.sigmaslide_qua.value
            self.oldxo1_qua = self.poslide_qua.value
            self.oldvel1_qua = self.velslide_qua.value
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

    def change_sigma(self):
        if self.oldsigma2_qua != self.sigmaslide_qua.value:
            self.reset()
            te.sigma0 = self.sigmaslide_qua.value
            self.oldsigma2_qua = self.sigmaslide_qua.value
            self.psi_qua.set_data(self.xarr_qua, np.abs(te.psi(self.xo, self.xarr_qua))**2)
            self.canvas_qua.draw()

    def change_xo(self, xoo):
        if self.oldxo2_qua != self.poslide_qua.value:
            self.reset()
            self.oldxo2_qua = self.poslide_qua.value
            self.xo = self.poslide_qua.value
            self.psi_qua.set_data(self.xarr_qua, np.abs(te.psi(xoo, self.xarr_qua))**2)
            self.canvas_qua.draw()

    def change_vel_qua(self):
        if self.oldvel2_qua != self.velslide_qua.value:
            self.reset()
            self.oldvel2_qua = self.velslide_qua.value
            te.p0 = self.velslide_qua.value

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
    def extend(self):
        #Makes one step by RKF.
        self.yarr[0,:] = self.yarr[1,:]
        self.yarr[1,:] = rob.RKF(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.tvec[1], self.yarr[1,:], rob.frollingball)

        self.tvec[0] = self.tvec[1]
        self.tvec[1] = self.tvec[1] + rob.h

        gotin = False

        #Fills all the possible values between the last step and the new one by interpolation.
        while self.lastt < self.tvec[1]:
            x0 = self.yin[0]
            self.yin = rob.interpol(self.tvec, self.yarr, self.lastt)
            self.supermatrix_cla = np.concatenate((self.supermatrix_cla, [[self.lastt, self.yin[0], self.yin[1]]]))

            self.perimeter = self.perimeter + rob.trapezoidal(self.mu_cla, self.sigma_cla, self.k_cla, x0, self.yin[0], self.deltax_cla, rob.groundperim)
            theta = self.perimeter/self.R
            beta = np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))
            self.angle = np.concatenate((self.angle, [[theta - beta]]))

            trans = 0.5*self.m_cla*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2 + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2)
            rot = 0.2*self.m_cla*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])/self.R - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))**2*self.yin[1]**2
            pot = self.m_cla*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])
            tot = trans + rot + pot
            self.energy = np.concatenate((self.energy, [[trans, rot, pot, tot]]))

            gotin = True

            self.lastt = self.lastt + self.dtime0_cla

        if gotin:
            self.timenet = np.append(self.timenet, [self.lastt], axis=0)
            self.energynet = np.concatenate((self.energynet, [[trans, rot, pot, tot]]))


    def computevolution_cla(self):
        if not self.computed_cla:
            #Overwrites the previously computed values.
            self.yin = self.yin0
            rob.h = 1
            self.yarr[1,:] = self.yin0
            self.tvec[1] = 0
            self.lastt = self.dtime0_cla
            self.timenet = np.array([0])

            self.supermatrix_cla = np.array([[self.time_cla, self.yin0[0], self.yin0[1]]]) #3 columns: time, x and xdot
            self.angle = np.array([[-np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))]])
            self.perimeter = 0.

            progress = 0

            #Energy. 4 columns; translational, rotational, potential and total.
            trans = 0.5*self.m_cla*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])*self.yin0[1])**2
            + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])*self.yin0[1])**2)
            rot = 0.2*self.m_cla*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])/self.R
            - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))**2*self.yin0[1]**2
            pot = self.m_cla*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])
            tot = trans + rot + pot
            self.energy = np.array([[trans, rot, pot, tot]])
            self.energynet = self.energy

            print("classical working")
            #Computes about 3 oscillations (or until the ball gets quite out of the screen)
            Period = 2*np.pi/np.sqrt(rob.g*self.k_cla)
            self.tmax_cla = float(4*Period)
            while self.lastt <= self.tmax_cla:
                self.extend()
                if 100*self.lastt/self.tmax_cla >= progress + 10:
                    progress = progress + 10
                    print(progress,"%")

            self.oldtop1_cla = self.heightslide_cla.value
            self.oldk1_cla = self.kslide_cla.value
            self.oldradius1_cla = self.radiusslide_cla.value
            self.oldxo1_cla = self.poslide_cla.value
            self.oldvel1_cla = self.velslide_cla.value
            print("classical worked!")

        self.computed_cla = True

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

            self.computed_cla = False
            self.reset_cla()
            self.angle[0] = -np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))
            self.plotball_0()
            self.plotE_cla()
            self.tmax_cla = 0


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
                row = int(t/self.dtime0_cla)

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

            #It starts again if the time reaches the top.
            if self.time_cla >= self.tmax_cla:
                self.time_cla = 0

            self.timeslide_cla.value = self.time_cla
            self.plotball(self.time_cla)

    def start_stop_cla(self):
        if self.computed_cla:
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
        self.startstopbut_cla.background_normal = "play.png"
        self.startstopbut_cla.background_down = "playblue.png"


    def reset_cla_btn(self):
        self.reset_cla()
        self.plotball(0)

    def change_radius(self):
        if self.oldradius2_cla != self.radiusslide_cla.value:
            self.R = self.radiusslide_cla.value
            self.oldradius2_cla = self.radiusslide_cla.value

            self.computed_cla = False
            self.reset_cla()
            self.plotball_0()
            self.plotE_cla()
            self.tmax_cla = 0

    def change_xo_cla(self):
        if self.oldxo2_cla != self.poslide_cla.value:
            self.oldxo2_cla = self.poslide_cla.value
            self.xo_cla = self.poslide_cla.value
            self.yin0[0] = self.xo_cla
            self.computed_cla = False
            self.reset_cla()
            self.supermatrix_cla[0, 1] = self.xo_cla
            self.angle[0] = -np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))
            self.plotball_0()
            self.plotE_cla()
            self.tmax_cla = 0

    def change_vel_cla(self):
        if self.oldvel2_cla != self.velslide_cla.value:
            self.oldvel2_cla = self.velslide_cla.value
            self.yin0[1] = self.velslide_cla.value
            self.computed_cla = False
            self.reset_cla()
            self.plotE_cla()
            self.tmax_cla = 0

    def velocity_cla_btn(self):
        self.vel_cla = self.vel_cla + 1
        if self.vel_cla == 5:
            self.vel_cla = 1
        self.dtime_cla = self.vel_cla*self.dtime0_cla
        self.vel_btn_cla.text = str(self.vel_cla) + "x"

    def plotE_cla(self):
        #Plots the initial energy.
        trans = 0.5*self.m_cla*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])*self.yin0[1])**2
        + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])*self.yin0[1])**2)
        rot = 0.2*self.m_cla*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])/self.R
        - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))**2*self.yin0[1]**2
        pot = self.m_cla*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])
        tot = trans + rot + pot

        #The energy is transformed to a height (as if all was potential energy).
        height = tot/(self.m_cla*rob.g) - self.R

        x = np.arange(-2.6, 2.6, 0.1)
        y = height + 0*x
        self.E_cla.set_data(x,y)
        self.canvas_cla.draw()


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
        self.ids.switch.active = False
        self.a = appcat
        self.canvas_ene = FigureCanvasKivyAgg(fig_ene)
        self.ids.Panel4_id.add_widget(self.canvas_ene)
        aen.clear()
        aen.set_ylim([0, max(self.a.energynet[:, 3]) + 10])
        aen.set_xlim([0, self.a.tmax_cla])
        aen.set_xlabel("t (s)")
        aen.set_ylabel("E (J)")
        self.event = Clock.schedule_interval(self.update, 0.3)
        aen.plot(self.a.timenet, self.a.energynet[:, 3], label = "Total", color = (1,0,0,1))
        aen.plot(self.a.timenet, self.a.energynet[:, 1], label = "Rotational", color = (0,1,0,1))
        aen.plot(self.a.timenet, self.a.energynet[:, 0], label = "Translational", color = (1,1,0,1))
        aen.plot(self.a.timenet, self.a.energynet[:, 2], label = "Potential", color = (0,0,1,1))
        aen.legend(loc=6)
        self.txt = aen.set_title("")
        self.txt.set_text("t" + r'$_{max}$' + " = " + '%1.1f' %self.a.supermatrix_cla[-1, 0] + " s"
        + "      E" + r'$_{total}$' + "= " + str(self.a.energynet[-1, 3]).ljust(7)[:7])

    def update(self, dt):
        if self.ids.switch.active == False:
            aen.clear()
            aen.set_ylim([0, max(self.a.energynet[:, 3]) + 10])
            aen.set_xlim([0, self.a.tmax_cla])
            aen.set_xlabel("t (s)")
            aen.set_ylabel("E (J)")
            aen.plot(self.a.timenet, self.a.energynet[:, 3], label = "Total", color = (1,0,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 1], label = "Rotational", color = (0,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 0], label = "Translational", color = (1,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 2], label = "Potential", color = (0,0,1,1))
            aen.legend(loc=6)
            self.txt = aen.set_title("")
            self.txt.set_text("t" + r'$_{max}$' + " = " + '%1.1f' %self.a.supermatrix_cla[-1, 0] + " s"
            + "      E" + r'$_{total}$' + "= " + str(self.a.energynet[-1, 3]).ljust(7)[:7])

            self.canvas_ene.draw()

        else:
            aen.clear()
            aen.set_ylim([0, max(self.a.energynet[:, 3]) + 10])
            aen.set_xlim([0, self.a.tmax_cla])
            aen.set_xlabel("t (s)")
            aen.set_ylabel("E (J)")
            aen.plot(self.a.timenet, self.a.energynet[:, 1] + self.a.energynet[:, 0] + self.a.energynet[:, 2], label = "Rotational", color = (0,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 0] + self.a.energynet[:, 2], label = "Translational", color = (1,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 2], label = "Potential", color = (0,0,1,1))

            aen.fill_between(self.a.timenet, 0, self.a.energynet[:, 2], color = (0,0,1,0.6))
            aen.fill_between(self.a.timenet, self.a.energynet[:, 2], self.a.energynet[:, 0] + self.a.energynet[:, 2], color = (1,1,0,0.6))
            aen.fill_between(self.a.timenet, self.a.energynet[:, 0] + self.a.energynet[:, 2], self.a.energynet[:, 1] + self.a.energynet[:, 0] + self.a.energynet[:, 2],
            color = (0,1,0,0.6))

            aen.legend(loc=6)
            self.txt = aen.set_title("")
            self.txt.set_text("t" + r'$_{max}$' + " = " + '%1.1f' %self.a.supermatrix_cla[-1, 0] + " s"
            + "      E" + r'$_{total}$' + "= " + str(self.a.energynet[-1, 3]).ljust(7)[:7])
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

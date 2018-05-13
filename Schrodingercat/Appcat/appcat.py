"""
Jan Albert Iglesias
11/5/2018
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
import warnings
from matplotlib import rc
import rollball as rob
import timeev as te
warnings.filterwarnings("error")

#Initialization of the plots:
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

        "Quantum definitions"
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

        #Norm
        self.norm = []
        self.timevec_qua = []
        psit = np.abs(te.psi(self.xo, self.xarr_qua))**2
        self.norm.append(self.deltax*(np.sum(psit) - psit[0]/2. - psit[self.N]/2.))
        self.timevec_qua.append(self.time_qua)

        #Plots (Quantum)
        self.canvas_qua = FigureCanvasKivyAgg(fig_qua)
        self.pot_qua, = aqu.plot(self.xarr_qua, te.pot(self.mu_qua, self.sigma_qua, self.k_qua, self.xarr_qua), 'g--', label = "V(x)")
        self.psi_qua, = aqu.plot(self.xarr_qua, np.abs(te.psi(self.xo, self.xarr_qua))**2, 'r-', label = r'$|\Psi(x)|^{2}$')
        aqu.axis([-5, 5, 0, 2])
        aqu.set_xlabel("x")
        aqu.legend(loc=1)
        self.panel2.add_widget(self.canvas_qua)



        "Classical definitions"
        self.time_cla = 0.
        self.heightslide_cla.value = 0.3
        self.sigma_cla =  1./(np.sqrt(2*np.pi)*self.heightslide_cla.value)
        self.mu_cla = 0
        rob.m = 5

        self.R = 0.2
        self.radiusslide_cla.value = self.R

        self.k_cla = 0.5
        self.kslide_cla.value = self.k_cla

        self.yin0 = np.array([1.5,0.0])
        self.poslide_cla.value = 1.5
        self.velslide_cla.value = 0.0

        self.xarr_cla = self.xarr_qua

        self.deltax_cla = 0.005 #dx to integrate the ground perimeter.

        self.method = "RKF45"
        self.rkf_button.background_color = (0.5,0.5,0.5,1)
        self.rk_button.background_color = (0.9,0.9,0.9,1)

        #Switches and flux controllers.
        self.switch1_cla = "off"
        self.oldtop2_cla = self.heightslide_cla.value
        self.oldk2_cla = self.kslide_cla.value
        self.oldradius2_cla = self.radiusslide_cla.value
        self.oldxo2_cla = self.poslide_cla.value
        self.oldvel2_cla = self.velslide_cla.value

        self.startstopbut_cla.background_down = "playblue.png"
        self.startstopbut_cla.background_normal = "play.png"

        #Clock (Classical):
        self.time_cla = 0.
        self.dtime0_cla = 0.01 #Default time step (used by RK4 and as default playing velocity).

        self.dtime_cla = self.dtime0_cla #Defines the playing velocity.
        self.vel_cla = 1
        self.vel_btn_cla.text = "1x"

        self.oldtime1_cla = self.time_cla + 1
        self.oldrow = -1

          #Starts and ends the event in order for "self.event_cla" to exist.
        self.event_cla = Clock.schedule_interval(self.ballupdate, self.dtime0_cla)
        Clock.unschedule(self.event_cla)

        #Runge-Kutta-Fehlberg:
        rob.eps = 0.000001 #Tolerance
        rob.h = 1
        self.yarr = np.zeros(shape=(2,2))
        self.tvec = np.zeros(shape=(2,1))


        #Plots (Classical):
        self.canvas_cla = FigureCanvasKivyAgg(fig_cla)
        self.panel1.add_widget(self.canvas_cla)

        acl.axis('scaled')
        acl.set_xlabel("x (m)")
        acl.set_ylabel("y (m)")
        acl.axis([-2.5, 2.5, 0 , 3])

        self.ground_cla, = acl.plot(self.xarr_cla, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla), 'k--')
        self.ballcm, = acl.plot([], [], 'ko', ms=1)
        self.ballperim, = acl.plot([], [], 'k-', lw=1)
        self.balldots, = acl.plot([], [], 'ko', ms=1)
        self.filled = acl.fill_between(self.xarr_cla, 0, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla), color = (0.5,0.5,0.5,0.5))
        self.E_cla, = acl.plot([],[], 'r--', lw=1)


        #First computations:
        self.computed_cla = False
        self.computing_cla = False
        self.computevolution()
        self.triggercompute_cla()
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
    #Computing:
    def extend(self):
        """
        It extends the matrices with the evolution parameters by the specified method.
        """
        if self.method == "RKF45":
            #Makes one step by RKF.
            self.yarr[0,:] = self.yarr[1,:]
            self.yarr[1,:] = rob.RKF(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.tvec[1], self.yarr[1,:], rob.frollingball)

            self.tvec[0] = self.tvec[1]
            self.tvec[1] = self.tvec[1] + rob.h

            gotin = False

            #Fills all the possible values between the last step and the new one by interpolation.
            while self.lastt < self.tvec[1]:
                x0 = self.yin[0]

                #Position & velocity:
                self.yin = rob.interpol(self.tvec, self.yarr, self.lastt)
                self.supermatrix_cla = np.concatenate((self.supermatrix_cla, [[self.lastt, self.yin[0], self.yin[1]]]))

                #Angle:
                self.perimeter = self.perimeter + rob.trapezoidal(self.mu_cla, self.sigma_cla, self.k_cla, x0, self.yin[0], self.deltax_cla, rob.groundperim)
                theta = self.perimeter/self.R
                beta = np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))
                self.angle = np.concatenate((self.angle, [[theta - beta]]))

                gotin = True

                self.lastt = self.lastt + self.dtime0_cla

            if gotin:
                #Energy. It does not use interpolation for the energies.
                trans = 0.5*rob.m*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2
                + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2)
                rot = 0.2*rob.m*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])/self.R
                - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))**2*self.yin[1]**2
                pot = rob.m*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])
                tot = trans + rot + pot

                self.timenet = np.append(self.timenet, [self.lastt - self.dtime0_cla], axis=0)
                self.energynet = np.concatenate((self.energynet, [[trans, rot, pot, tot]]))

        if self.method == "RK4":
            x0 = self.yin[0]

            #Position & velocity:
            self.yin = rob.RK4(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.lastt, self.dtime0_cla, self.yin, rob.frollingball)
            self.supermatrix_cla = np.concatenate((self.supermatrix_cla, [[self.lastt, self.yin[0], self.yin[1]]]))

            #Angle:
            self.perimeter = self.perimeter + rob.trapezoidal(self.mu_cla, self.sigma_cla, self.k_cla, x0, self.yin[0], self.deltax_cla, rob.groundperim)
            theta = self.perimeter/self.R
            beta = np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))
            self.angle = np.concatenate((self.angle, [[theta - beta]]))

            #Energy:
            trans = 0.5*rob.m*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2
            + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])*self.yin[1])**2)
            rot = 0.2*rob.m*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])/self.R
            - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0]))**2*self.yin[1]**2
            pot = rob.m*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin[0])
            tot = trans + rot + pot

            self.timenet = np.append(self.timenet, [self.lastt], axis=0)
            self.energynet = np.concatenate((self.energynet, [[trans, rot, pot, tot]]))

            self.lastt = self.lastt + self.dtime0_cla

    def extend5percent(self, callback):
        """
        It keeps calling extend() until the progreess percentage is increased by 5.
        """
        while 100*self.lastt/self.tmax_cla <= self.progressbar_cla.value + 5:
            self.extend()
        callback()

    def triggercompute_cla(self):
        #Just gets triggered if it has not been computed:
        if not self.computed_cla:
            Computevolution_cla(self.progressbar_cla, self.extend5percent, self)


    #Changing parameters:
    def plotground(self):
        #It is only activated if some value changes.
        a = self.oldtop2_cla != self.heightslide_cla.value
        b = self.oldk2_cla != self.kslide_cla.value

        if a or b:
            self.oldtop2_cla = self.heightslide_cla.value
            self.oldk2_cla = self.kslide_cla.value

            #Changes and plots the newground.
            self.sigma_cla = 1./(np.sqrt(2*np.pi)*self.heightslide_cla.value)
            self.k_cla = self.kslide_cla.value
            self.ground_cla.set_data(self.xarr_cla, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla))
            self.filled.remove()
            self.filled = acl.fill_between(self.xarr_cla, 0, rob.fground(self.mu_cla, self.sigma_cla, self.k_cla, self.xarr_cla),
            color = (0.5,0.5,0.5,0.5))

            #Sets time to 0.
            self.reset_cla()
            self.tmax_cla = 0

            #Plots the ball again.
            self.angle[0] = -np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))
            self.plotball_0()
            self.plotE_cla()

            #Disables some buttons.
            self.compu_button.disabled = False
            self.startstopbut_cla.disabled = True
            self.reset_cla_button.disabled = True
            self.vel_btn_cla.disabled = True
            self.timeslide_cla.disabled = True

            self.computed_cla = False

    def change_radius(self):
        #It is only activated if the value changes.
        if self.oldradius2_cla != self.radiusslide_cla.value:
            self.oldradius2_cla = self.radiusslide_cla.value

            #Changes the radius.
            self.R = self.radiusslide_cla.value

            #Sets time to 0.
            self.reset_cla()
            self.tmax_cla = 0

            #Plots the new ball.
            self.plotball_0()
            self.plotE_cla()

            #Disables some buttons.
            self.compu_button.disabled = False
            self.startstopbut_cla.disabled = True
            self.reset_cla_button.disabled = True
            self.vel_btn_cla.disabled = True
            self.timeslide_cla.disabled = True

            self.computed_cla = False

    def change_xo_cla(self):
        #It is only activated if the value changes.
        if self.oldxo2_cla != self.poslide_cla.value:
            self.oldxo2_cla = self.poslide_cla.value

            #Changes the initial position.
            #Note that it is the position of the contact point.
            self.xo_cla = self.poslide_cla.value
            self.yin0[0] = self.xo_cla

            #Sets time to 0.
            self.reset_cla()
            self.tmax_cla = 0

            #Plots the new ball.
            self.supermatrix_cla[0, 1] = self.xo_cla
            self.angle[0] = -np.arctan(rob.dfground(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))
            self.plotball_0()
            self.plotE_cla()

            #Disables some buttons.
            self.compu_button.disabled = False
            self.startstopbut_cla.disabled = True
            self.reset_cla_button.disabled = True
            self.vel_btn_cla.disabled = True
            self.timeslide_cla.disabled = True

            self.computed_cla = False

    def change_vel_cla(self):
        #It is only activated if the value changes.
        if self.oldvel2_cla != self.velslide_cla.value:
            self.oldvel2_cla = self.velslide_cla.value

            #Changes the initial velocity.
            #Note that it is the velocity of the contact point.
            self.yin0[1] = self.velslide_cla.value

            #Sets time to 0.
            self.reset_cla()
            self.tmax_cla = 0

            #Plots the new ball.
            self.plotball_0()
            self.plotE_cla()

            #Disables some buttons.
            self.compu_button.disabled = False
            self.startstopbut_cla.disabled = True
            self.reset_cla_button.disabled = True
            self.vel_btn_cla.disabled = True
            self.timeslide_cla.disabled = True

            self.computed_cla = False


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
                self.event_cla = Clock.schedule_interval(self.ballupdate, self.dtime0_cla)
                self.startstopbut_cla.background_normal = "pause.png"
                self.startstopbut_cla.background_down = "playblue.png"
                self.switch1_cla = "on"
            elif self.switch1_cla == "on":
                Clock.unschedule(self.event_cla)
                self.startstopbut_cla.background_normal = "play.png"
                self.startstopbut_cla.background_down = "pauseblue.png"
                self.switch1_cla = "off"

    def reset_cla(self):
        #Sets time to 0
        Clock.unschedule(self.event_cla)
        self.switch1_cla = "off"
        self.timeslide_cla.value = 0
        self.time_cla = 0
        self.startstopbut_cla.background_normal = "play.png"
        self.startstopbut_cla.background_down = "playblue.png"


    def reset_cla_btn(self):
        self.reset_cla()
        self.plotball(0)


    def velocity_cla_btn(self):
        self.vel_cla = self.vel_cla + 1
        if self.vel_cla == 5:
            self.vel_cla = 1
        self.dtime_cla = self.vel_cla*self.dtime0_cla
        self.vel_btn_cla.text = str(self.vel_cla) + "x"

    def plotE_cla(self):
        #Plots the initial energy.
        trans = 0.5*rob.m*((rob.dxcm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])*self.yin0[1])**2
        + (rob.dycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])*self.yin0[1])**2)
        rot = 0.2*rob.m*self.R**2*(rob.groundperim(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])/self.R
        - rob.dalpha(self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0]))**2*self.yin0[1]**2
        pot = rob.m*rob.g*rob.ycm(self.R, self.mu_cla, self.sigma_cla, self.k_cla, self.yin0[0])
        tot = trans + rot + pot

        #The energy is transformed to a height (as if all was potential energy).
        height = tot/(rob.m*rob.g) - self.R

        x = np.arange(-2.6, 2.6, 0.1)
        y = height + 0*x
        self.E_cla.set_data(x,y)
        self.canvas_cla.draw()

    def change_RK(self):
        if self.method != "RK4":
            self.method = "RK4"
            self.rk_button.background_color = (0.5,0.5,0.5,1)
            self.rkf_button.background_color = (0.9,0.9,0.9,1)
            self.computed_cla = False
            self.reset_cla()
            self.tmax_cla = 0
            self.plotball_0()

            self.compu_button.disabled = False
            self.startstopbut_cla.disabled = True
            self.reset_cla_button.disabled = True
            self.vel_btn_cla.disabled = True
            self.timeslide_cla.disabled = True


    def change_RKF(self):
        if self.method != "RKF45":
            self.method = "RKF45"
            self.rkf_button.background_color = (0.5,0.5,0.5,1)
            self.rk_button.background_color = (0.9,0.9,0.9,1)
            self.computed_cla = False
            self.reset_cla()
            self.tmax_cla = 0
            self.plotball_0()

            self.compu_button.disabled = False
            self.startstopbut_cla.disabled = True
            self.reset_cla_button.disabled = True
            self.vel_btn_cla.disabled = True
            self.timeslide_cla.disabled = True


class Computevolution_cla(object):
    def __init__(self, progressbar, extendfun, appcat):
        self.a = appcat
        self.pb = progressbar
        self.ext = extendfun

        #Disables everything while computing.
        self.a.heightslide_cla.disabled = True
        self.a.kslide_cla.disabled = True
        self.a.radiusslide_cla.disabled = True
        self.a.poslide_cla.disabled = True
        self.a.velslide_cla.disabled = True
        self.a.rkf_button.disabled = True
        self.a.rk_button.disabled = True
        self.a.energy_button.disabled = True
        self.a.timeslide_cla.disabled = True
        self.a.compu_button.disabled = True
        self.a.startstopbut_cla.disabled = True
        self.a.reset_cla_button.disabled = True
        self.a.vel_btn_cla.disabled = True

        #Overwrites the previously computed values:
        self.a.yin = self.a.yin0
        rob.h = 1
        self.a.yarr[1,:] = self.a.yin0
        self.a.tvec[1] = 0
        self.a.timenet = np.array([0])
        self.a.lastt = self.a.dtime0_cla

        self.a.supermatrix_cla = np.array([[self.a.time_cla, self.a.yin0[0], self.a.yin0[1]]]) #3 columns: time, x and xdot
        self.a.angle = np.array([[-np.arctan(rob.dfground(self.a.mu_cla, self.a.sigma_cla, self.a.k_cla, self.a.yin0[0]))]])
        self.a.perimeter = 0.

        #Energy. 4 columns; translational, rotational, potential and total.
        trans = 0.5*rob.m*((rob.dxcm(self.a.R, self.a.mu_cla, self.a.sigma_cla, self.a.k_cla, self.a.yin0[0])*self.a.yin0[1])**2
        + (rob.dycm(self.a.R, self.a.mu_cla, self.a.sigma_cla, self.a.k_cla, self.a.yin0[0])*self.a.yin0[1])**2)
        rot = 0.2*rob.m*self.a.R**2*(rob.groundperim(self.a.mu_cla, self.a.sigma_cla, self.a.k_cla, self.a.yin0[0])/self.a.R
        - rob.dalpha(self.a.mu_cla, self.a.sigma_cla, self.a.k_cla, self.a.yin0[0]))**2*self.a.yin0[1]**2
        pot = rob.m*rob.g*rob.ycm(self.a.R, self.a.mu_cla, self.a.sigma_cla, self.a.k_cla, self.a.yin0[0])
        tot = trans + rot + pot
        self.a.energynet = np.array([[trans, rot, pot, tot]])

        #Computes about 3 oscillations, or a maximum time.
        #(In order to prevent tmax from being inf.)
        if self.a.k_cla < 0.15:
            self.a.tmax_cla = 10
        else:
            Period = 2*np.pi/np.sqrt(rob.g*self.a.k_cla)
            self.a.tmax_cla = float(3*Period)

        #Calls the extend function, that will make a new step and call the task_complete function when finishes.
        Clock.schedule_once(lambda dt: self.ext(self.task_complete), 0.1)

    def task_complete(self):
        #Checks if there are still steps to be done, calls again the function and updates the progressbar.
        if self.a.lastt <= self.a.tmax_cla + 2*self.a.dtime0_cla:
            self.pb.value = 100*self.a.lastt/self.a.tmax_cla
            Clock.schedule_once(lambda dt: self.ext(self.task_complete), 0.1)

        else:
            self.a.computed_cla = True
            self.pb.value = 0

            #Ables the widgets again.
            self.a.heightslide_cla.disabled = False
            self.a.kslide_cla.disabled = False
            self.a.radiusslide_cla.disabled = False
            self.a.poslide_cla.disabled = False
            self.a.velslide_cla.disabled = False
            self.a.rkf_button.disabled = False
            self.a.rk_button.disabled = False
            self.a.energy_button.disabled = False
            self.a.timeslide_cla.disabled = False
            self.a.startstopbut_cla.disabled = False
            self.a.reset_cla_button.disabled = False
            self.a.vel_btn_cla.disabled = False


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
    """
    Pop up window with the plot of the different energies vs time.
    """
    def __init__(self, appcat):
        super(Energypopup, self).__init__()
        self.a = appcat
        self.surface = False
        self.ids.switch.text = "off"

        #Plot definitions:
        self.canvas_ene = FigureCanvasKivyAgg(fig_ene)
        self.ids.Panel4_id.add_widget(self.canvas_ene)
        aen.clear()
        aen.set_ylim([0, max(self.a.energynet[:, 3]) + 10])
        aen.set_xlim([0, self.a.tmax_cla])
        aen.set_xlabel("t (s)")
        aen.set_ylabel("E (J)")
        self.txt = aen.set_title("")
        self.txt.set_text("E" + r'$_{total}$' + "= " + str(self.a.energynet[-1, 3]).ljust(7)[:7])

        #Plots:
        aen.plot(self.a.timenet, self.a.energynet[:, 3], label = "Total", color = (1,0,0,1))
        aen.plot(self.a.timenet, self.a.energynet[:, 1], label = "Rotational", color = (0,1,0,1))
        aen.plot(self.a.timenet, self.a.energynet[:, 0], label = "Translational", color = (1,1,0,1))
        aen.plot(self.a.timenet, self.a.energynet[:, 2], label = "Potential", color = (0,0,1,1))
        aen.legend(loc=6)

        self.canvas_ene.draw()

    def changeplot(self):
        if self.surface:
            self.ids.switch.text = "off"
            self.surface = False
            aen.clear()

            #Plot definitions:
            aen.set_ylim([0, max(self.a.energynet[:, 3]) + 10])
            aen.set_xlim([0, self.a.tmax_cla])
            aen.set_xlabel("t (s)")
            aen.set_ylabel("E (J)")
            self.txt = aen.set_title("")
            self.txt.set_text("E" + r'$_{total}$' + "= " + str(self.a.energynet[-1, 3]).ljust(7)[:7])

            #Plots:
            aen.plot(self.a.timenet, self.a.energynet[:, 3], label = "Total", color = (1,0,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 1], label = "Rotational", color = (0,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 0], label = "Translational", color = (1,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 2], label = "Potential", color = (0,0,1,1))
            aen.legend(loc=6)

            self.canvas_ene.draw()

        else:
            self.ids.switch.text = "on"
            self.surface = True
            aen.clear()

            #Plot definitions:
            aen.set_ylim([0, max(self.a.energynet[:, 3]) + 10])
            aen.set_xlim([0, self.a.tmax_cla])
            aen.set_xlabel("t (s)")
            aen.set_ylabel("E (J)")
            self.txt = aen.set_title("")
            self.txt.set_text("E" + r'$_{total}$' + "= " + str(self.a.energynet[-1, 3]).ljust(7)[:7])

            #Plots:
            aen.plot(self.a.timenet, self.a.energynet[:, 1] + self.a.energynet[:, 0] + self.a.energynet[:, 2], label = "Rotational", color = (0,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 0] + self.a.energynet[:, 2], label = "Translational", color = (1,1,0,1))
            aen.plot(self.a.timenet, self.a.energynet[:, 2], label = "Potential", color = (0,0,1,1))

            aen.fill_between(self.a.timenet, 0, self.a.energynet[:, 2], color = (0,0,1,0.6))
            aen.fill_between(self.a.timenet, self.a.energynet[:, 2], self.a.energynet[:, 0] + self.a.energynet[:, 2], color = (1,1,0,0.6))
            aen.fill_between(self.a.timenet, self.a.energynet[:, 0] + self.a.energynet[:, 2], self.a.energynet[:, 1] + self.a.energynet[:, 0] + self.a.energynet[:, 2],
            color = (0,1,0,0.6))

            aen.legend(loc=6)

            self.canvas_ene.draw()



class appcatApp(App):
    def build(self):
        self.title = "Main App"
        a = Appcat()
        return a

if __name__ == "__main__":
    appcatApp().run()

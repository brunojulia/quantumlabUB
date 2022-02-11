from kivy.app import App
from kivy.core.window import Window
Window.clearcolor = (1, 1, 1, 1)
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.screenmanager import FadeTransition, SlideTransition
#from kivy.uix.popup import Popup
from functools import partial
from kivy.properties import ObjectProperty,ReferenceListProperty,\
    NumericProperty,StringProperty,ListProperty,BooleanProperty

import numpy as np
import matplotlib.pyplot as plt

class tun_v4App(App):
    def build(self):
        self.title='Spin Tunneling'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('Tunneling').stpseudo_init()
#        self.get_screen("Game").gpseudo_init()
    pass

class StartingScreen(Screen):
    def __init__(self,**kwargs):
        super(StartingScreen,self).__init__(**kwargs)

    def transition_ST(self):
        """Transicio del starting a l Spin Tunneling"""

        stscreen = self.manager.get_screen('Tunneling')
        self.manager.transition = FadeTransition()
        self.manager.current = 'Tunneling'

    def transition_Game(self):
        """Transicio del starting a l Spin Tunneling"""

        stscreen = self.manager.get_screen('Game')
        self.manager.transition = FadeTransition()
        self.manager.current = 'Game'

class TunnelingScreen(Screen):
    H_type= NumericProperty(1)
    s= NumericProperty(1)
    def __init__(self,**kwargs):
        super(TunnelingScreen,self).__init__(**kwargs)

    def stpseudo_init(self):
        def H(alpha, t):
            if H_type == 1:
                Ham = -Sz2 + a * t * Sz + alpha * (Sxy2)
            if H_type == 2:
                Ham = -Sz2 + a * t * Sz + alpha * (Sxy2)
            return Ham
            # Ham = -Sz2 + a * t * Sz + alpha * (Sxy2)
            # return Ham

        # Equacions diferencials de primer ordre associades a les diferents a_m
        def dam(t, am, i):
            sum_H = 0.
            m_i = np.array([m[i]])

            for k in range(mlt):
                m_k = np.array([m[k]])
                m_k = np.transpose(m_k)
                sum_H = sum_H + am[k] * np.dot(m_i, (np.dot(H(alpha, t), m_k)))

            dam_res = -1j * sum_H

            return dam_res

        # Funcio que utilitza el metode Runge Kutta4 per resoldre les equacions
        # diferencials de primer ordre
        def RK4(t, h, am, s):
            for i in range(mlt):
                k0 = h * dam(t, am, i)
                k1 = h * dam(t + 0.5 * h, am + 0.5 * k0, i)
                k2 = h * dam(t + 0.5 * h, am + 0.5 * k1, i)
                k3 = h * dam(t + 5 * h, am + k2, i)
                am[i] = am[i] + (k0 + 2 * k1 + 2 * k2 + k3) / 6.
            return a_m

        """ PROGRAMA """

        # Definició de l'Hamiltonià
        global H_type
        # H_type = int(input('Escriu 1'))

        # Creacio dels vectors propis de l'spin
        m = []
        s = 1.5
        mlt = int(2 * s + 1)
        dam_res = [0] * (mlt)
        alpha = 0.1
        a = 0.01
        t0 = -50.
        tf = 50.

        # Definicio de parametres generals per descriure l,Hamiltonia
        Sx = np.zeros(shape=(mlt, mlt), dtype=complex)
        Sy = np.zeros(shape=(mlt, mlt), dtype=complex)
        Sz = np.zeros(shape=(mlt, mlt), dtype=complex)

        for i in range(mlt):
            m_i = [0] * (int(2 * s) - i) + [1] + [0] * (i)
            m.append(m_i)

            for k in range(mlt):
                if k == (i + 1):
                    Sx[i, k] = 0.5 * np.sqrt(s * (s + 1) - (i - s) * (k - s))
                    Sy[i, k] = -0.5j * np.sqrt(s * (s + 1) - (i - s) * (k - s))

                if k == (i - 1):
                    Sx[i, k] = 0.5 * np.sqrt(s * (s + 1) - (i - s) * (k - s))
                    Sy[i, k] = 0.5j * np.sqrt(s * (s + 1) - (i - s) * (k - s))

                if k == i:
                    Sz[i, k] = i - s

        Sxy2 = np.dot(Sx, Sx) - np.dot(Sy, Sy)
        Sz2 = np.dot(Sz, Sz)

        # Condicions inicials
        a_m = np.zeros(shape=(mlt, 1), dtype=complex)
        a_m[0] = 1 + 0j

        # Execucio de les funcions

        nstep = 2000
        h = (tf - t0) / nstep
        t = t0

        ti = np.zeros(shape=(nstep + 1, 1))
        mod = np.zeros(shape=(nstep + 1, 1))
        prob = np.zeros(shape=(nstep + 1, mlt))
        ti[0] = t0


        # self.main_canvas = FigureCanvasKivyAgg(plt.plot([0],[0]))
        # self.box1.add_widget(self.main_canvas, 1)

class GameScreen(Screen):

    def __init__(self,**kwargs):
        super(GameScreen,self).__init__(**kwargs)

    def gamepseudo_init(self):
        pass

if __name__ == '__main__':
    tun_v4App().run()
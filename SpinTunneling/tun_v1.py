from kivy.app import App
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

class tun_v1App(App):
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
#        stscreen.st_schedule_fired()  # Aixo es per a fer la animacio
        self.manager.transition = FadeTransition()
        self.manager.current = 'Tunneling'

class TunnelingScreen(Screen):
    def __init__(self,**kwargs):
        super(TunnelingScreen,self).__init__(**kwargs)

    def H(self,alpha, t):
        Ham = -self.Sz2 + self.a * self.t * self.Sz + self.alpha * self.Sxy2
        return Ham

    def dam(self,t, am, i):
        self.sum_H = 0.
        self.m_i = np.array([self.m[i]])

        for k in range(2 * s + 1):
            self.m_k = np.array([self.m[k]])
            self.m_k = np.transpose(self.m_k)
            self.sum_H = self.sum_H + self.am[k] * np.dot(self.m_i, (np.dot(self.H(self.alpha, self.t), self.m_k)))

        dam_res = -1j * self.sum_H

        return dam_res

    def RK4(self,t, h, am, s):
        for i in range(int(2 * self.s + 1)):
            self.k0 = self.h * self.dam(self.t, self.am, i)
            self.k1 = self.h * self.dam(self.t + 0.5 * self.h, self.am + 0.5 * self.k0, i)
            self.k2 = self.h * self.dam(self.t + 0.5 * self.h, self.am + 0.5 * self.k1, i)
            self.k3 = self.h * self.dam(self.t + 0.5 * self.h, self.am + self.k2, i)
            self.am[i] = self.am[i] + (self.k0 + 2 * self.k1 + 2 * self.k2 + self.k3) / 6.
        return a_m



    def stpseudo_init(self):
        # Creacio dels vectors propis de l'spin
        m=[]
        s=1
        dam_res=[0]*(2*s+1)
        alpha=0.1
        a=0.1
        t0=-10.
        tf=10.

        for i in range(int(2*s+1)):
            m_i=[0]*(int(2*s)-i)+[1]+[0]*(i)
            m.append(m_i)


        #Condicions inicials
        a_m=np.zeros(shape=(int(2*s+1),1),dtype=complex)
        a_m[0]=1+0j

"""AQUI FALTA ESCRIURE SELF A LES VARIABLES DE LES FUNCIONS"""
        #Execucio de les funcions

        nstep=20000
        h=(tf-t0)/nstep
        t=t0
        a1=[0]*(nstep+1)
        a2=[0]*(nstep+1)
        a3=[0]*(nstep+1)
        ti=[0]*(nstep+1)
        mod=[0]*(nstep+1)

        ti[0]=t0
        a1[0]=abs(a_m[0])*abs(a_m[0])
        a2[0]=abs(a_m[1])*abs(a_m[1])
        a3[0]=abs(a_m[2])*abs(a_m[2])
        mod[0]=a1[0]+a2[0]+a3[0]

        for n in range(nstep):
            print(n)

        an=RK4(t,h,a_m,s)
        a1[n+1]=abs(an[0])*abs(an[0])
        a2[n+1]=abs(an[1])*abs(an[1])
        a3[n+1]=abs(an[2])*abs(an[2])
        ti[n+1]=t
        mod[n+1]=a1[n]+a2[n]+a3[n]
        t=t+h
        a_m=an

if __name__ == '__main__':
    tun_v1App().run()
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:01:12 2022

@author: Eric Vidal Marcos
"""
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.config import Config
Config.set('graphics', 'resizable', True)
from kivy.graphics import Color
from kivy.animation import Animation

from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.screenmanager import FadeTransition, SlideTransition
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner
from kivy.uix.image import Image
from kivy.uix.floatlayout import FloatLayout 
from kivy.uix.gridlayout import GridLayout

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.graphics.vertex_instructions import Line,Rectangle,Ellipse

from kivy.properties import ObjectProperty,ReferenceListProperty,\
    NumericProperty,StringProperty,ListProperty,BooleanProperty


import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from matplotlib import font_manager as fm, rcParams
plt.rcParams.update({'font.size': 13}) #matplotlib fontsize

#from functools import partial
#from scipy import integrate as inte
#import random 
#import time


#This is the way to name the kv file as we want
Builder.load_file("SpinTun_kivyfile.kv")


class Reso_P(FloatLayout):
    pass

def reso_popup():
    show = Reso_P()

    popupWindow = Popup(title="Resonance Information", content=show, size_hint=(.7,.7))

    popupWindow.open()

#SCREENS

class Starting_Screen(Screen):
    pass

class Menu_Screen(Screen):
    pass

class Resonance_Screen(Screen):
    
    def __init__(self,**kwargs):
        super(Resonance_Screen,self).__init__(**kwargs)
        #set empty graphic properties
        self.plot, self.axs =plt.subplots(1)
        self.axs.set(xlim=[-50,50],ylim=[0,1])

        #Add the graphic to the box which it corresponds
        self.plot = FigureCanvasKivyAgg(self.plot)
        self.graphic_box1.add_widget(self.plot)


    #predefined values, they are going to be set via buttons
    H_type= NumericProperty(1)
    s= NumericProperty(1)
    a=NumericProperty(0.01)
    alpha=NumericProperty(0.1)
    
    def popup_btn(self):
        reso_popup()
    
    def spinner_clicked(self, value):
        print("Spin is " + value)

    def send_but(self):

        plt.clf()
        self.mlt = int(2 * self.s + 1)

        # Definicio de parametres generals per descriure l,Hamiltonia per un spin s
        self.Sx = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.Sy = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.Sz = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.m = np.zeros(shape=(self.mlt, self.mlt), dtype=int)

        for i in range(self.mlt):
            self.Sz[i, i] = (i - self.s) * (-1)
            self.m[i, i] = 1
            for k in range(self.mlt):
                if k == (i + 1):
                    self.Sx[i, k] = 0.5 * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
                    self.Sy[i, k] = -0.5j * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))

                if k == (i - 1):
                    self.Sx[i, k] = 0.5 * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
                    self.Sy[i, k] = 0.5j * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
        self.Sxy2 = np.dot(self.Sx, self.Sx) - np.dot(self.Sy, self.Sy)
        self.Sz2 = np.dot(self.Sz, self.Sz)

        self.y0 = []
        for i in range(self.mlt):
            if i != 0:
                self.y0.append(0 + 0j)
            else:
                self.y0.append(1 + 0j)

        t0 = -50.
        tf = 50.
        # self.alpha=0.1
        # self.alpha = float(self.ids.alpha_text.text)
        # self.a=0.01
        self.ti = [t0, tf]

        self.sol = solve_ivp(self.dam, self.ti, self.y0,rtol=1e-4)

        am = self.sol.y
        mod2 = abs(am) * abs(am)

        for i in range(self.mlt - 1):
            if i == 0:
                norma = (list(map(sum, zip(mod2[i], mod2[i + 1]))))
            else:
                norma = (list(map(sum, zip(norma, mod2[i + 1]))))

        temps=self.sol.t
        plt.xlabel("t'")
        plt.ylabel('Probability')
        plt.plot(temps, norma, label='Norma')
        plt.axhline(y=1, xmin=t0, xmax=tf, ls='dashed')
        for i in range(self.mlt):
            # plt.plot(t,mod2[i],label='|'+str(int(i-s))+'>')
            plt.plot(temps, mod2[i], label='|' + str(int(i - self.s)*(-1)) + '>')
        plt.legend()

        #A VEURE SI PUC FER EL MALEÏT DIBUIX


        self.graphic_box1.remove_widget(self.plot)
        self.plot = FigureCanvasKivyAgg(plt.gcf())
        self.graphic_box1.add_widget(self.plot)

        # self.plot.draw()
        print('ACABAT')

    def H(self,t):
        if self.H_type==1:
            Ham = -self.Sz2 + self.a * t * self.Sz + self.alpha * (self.Sxy2)
        if self.H_type==2:
            Ham= -self.Sz2+self.a * t * self.Sz + self.alpha * self.Sx
        return Ham

    def dam(self,t, y):
        dadt = []
        for i in range(self.mlt):
            sum_H = 0.
            mi = self.m[i]

            for k in range(self.mlt):
                mk = self.m[k]
                mk = np.transpose(mk)
                sum_H = sum_H + y[k] * np.dot(mi, (np.dot(self.H(t), mk)))

            dam_res = -1j * sum_H
            dadt.append(dam_res)
            dam_res = 0

        return dadt


class SpinTunApp(App):
    def build(self):
        #sets window title
        self.title='Spin Tunneling'
        
        #changes default transition (right) to fade
        sm = ScreenManager(transition=FadeTransition())
        
        #Screens definition
        sm.add_widget(Starting_Screen(name='Starting'))
        sm.add_widget(Menu_Screen(name='Menu'))       
        sm.add_widget(Resonance_Screen(name='Resonance'))
        
        #Changes Starting screen to Menu
        def my_callback(self, *largs):
            sm.current = 'Menu'
        #Triggers the screen switch and controls the time that the Starting
        #Screen is showed
        Clock.schedule_once(my_callback, 0.5)
        #3.2 may be the most appropriate time
        return sm

if __name__ == '__main__':
    Window.maximize()   #opens in full screen
    SpinTunApp().run()  #run method inside App class
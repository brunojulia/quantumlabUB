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

#POPUPS
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
    
    def popup_btn(self):
        reso_popup()
    
    
    #BACKEND
    
    H_type= NumericProperty(1)
    s = NumericProperty(1)
    D = NumericProperty(0.01)
    alpha = NumericProperty(0.1)
    
    def spinner_clicked_s(self, value):
        self.s = int(value)
    
    def spinner_clicked_ham(self, value):
        #self.ham = int(value)
        pass

    def send(self):

        plt.clf()
        #1. Initial parameters
        #Arbitrary spin to study
        #s = 3     #total spin
        dim=round(2*self.s+1)    #in order to work when s is half-integer with int dim
        Nterm1=self.s*(self.s+1)      #1st term of N+-
        
        tf = 70
        #Hamiltonian Parameters
        #D = 7
        #alpha = 1.7
        H0 = (tf/2)*np.abs(self.alpha)
        B = 0.35
        
        #Time span
        At = [0,tf]
        
        #IC
        a_m0=[]
        
        for i in range(dim-1):
            a_m0.append(0+0j)
        a_m0.append(1+0j)
        
        
        #States energies if H_0
        energies=[]
        for i in range(dim):
            energies.append([])
        for i in range(dim):
            for j in range(2):
                energies[i].append(-self.D*(i-self.s)**2+(H0-self.alpha*At[j])*(i-self.s))
                
        #Transition times
        time_n=[]
        yintersec_n=[]
        for i in range(self.s):
            x1=At[1]
            energies_y=energies[2*i]
            energies_z=energies[2*i+2]
            y0=energies_y[0]
            y1=energies_y[1]
            z0=energies_z[0]
            z1=energies_z[1]
            xintersec=x1*((z0-y0)/((y1-y0)-(z1-z0)))
            yintersec=y0+xintersec*((y1-y0)/x1)
            time_n.append(xintersec)
            yintersec_n.append(yintersec)
            
        #2. Coupled differential equations
        #WE HAVE TO TAKE INTO ACCOUNT THAT M DOESNT GO FROM -S TO S
        #IT GOES FROM 0 TO DIM-1=2s
        
        #N+ and N- definition
        def Np(m):
            m=m-self.s   #cuz m goes from 0 to 2s
            Nplus=np.sqrt(Nterm1-m*(m+1))
            return Nplus
        def Nm(m):
            m=m-self.s   #cuz m goes from 0 to 2s
            Nminus=np.sqrt(Nterm1-m*(m-1))
            return Nminus
        
        #PER TESTEJAR SI EL METODE ES CORRECTE, COMPARAREM AMB RESULTATS LAURA
        #definition of ODE's
        def dak1(k, t, am):
            '''Inputs: s (int) total spin, k (int) quantum magnetic number,
            t (int or float) time, am (array (dim,1)) coefficients of each state.
            Outputs: dak (complex) time derivative of a_k.
            This function returns each differential equation for coefficients time
            evolution.'''
            #First we define k to the scale we work in QM
            kreal=k-self.s
            if (kreal>self.s):
                print('It makes no sense that k>s or k<s, sth went wrong.')
                exit()
                
            #eigenvalues term
            eigenterm=am[k]*(-self.D*kreal**2+(H0-self.alpha*t)*kreal)
            
            #summatory term
            sumterm=0
            for m in range(dim):
                #first we apply Kronicker deltas
                if (k==(m+2)):
                    sumtermpos=Np(m)*Np(m+1)
                else:
                    sumtermpos=0
                    
                if (k==(m-2)):
                    sumtermneg=Nm(m)*Nm(m-1)
                else:
                    sumtermneg=0
                    
                #and obtain summatory term along the for
                sumterm += am[m]*(B/2)*(sumtermpos+sumtermneg)
            
            #finally obtaining the result of one differential equation
            dak=-1j*(eigenterm+sumterm)
            return dak
    
        def odes(t, a_m):
            '''Input: t (int or float) time and a_m (1D list) coefficients. D, h, B
            (int or floats) are Hamiltonian parameters that could be omitted because
            they are global variables as s (spin).
            Ouput: system (1D list), this is the coupled differential equation
            system.'''
            system=[]
            for i in range(dim):
                system.append(dak1(i, t, a_m))
            
            return system
        
        #3. Resolution and plotting
        
        #solve
        a_m=solve_ivp(odes, At, a_m0)
        
        #Plotting parameters
        t=a_m.t[:]  #time
        
        aplot=[]
        totenergy_temp=[]
        for i in range(dim):
            totenergy_n=[]
            prob_i=np.abs(a_m.y[i,:])**2
            aplot.append(prob_i)     #Probabilities coeff^2
            for j in range(len(t)):
                totenergy_n.append(prob_i[j]*(-self.D*(i-self.s)**2+(H0-self.alpha*t[j])*(i-self.s)))
            totenergy_temp.append(totenergy_n)
        
        norm = np.sum(aplot, axis=0)    #Norm (sum of probs)
        totenergy = np.sum(totenergy_temp, axis=0)    #Total energy (sum of each spin energy*prob)
        
        #Plot
        plt.figure()
        plt.subplot(231)
        plt.title('Spin probabilties') #General spin method, solve_ivp
        plt.xlabel('t')
        plt.ylabel('$|a|^2$')
        plt.axhline(y=1.0,linestyle='--',color='grey')
        
        #Probabilities
        for i in range(dim):
            plt.plot(t, aplot[i],'-',label='m='+str(i-self.s))
        plt.plot(t, norm,'-',label='norma')
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        
        
        plt.subplot(233)
        plt.title('States energies if $\mathcal{H}_0$')
        plt.xlabel('t')
        plt.ylabel('$E$')
        for i in range(dim):
            plt.plot(At, energies[i],'-',label='$E_{'+str(i-self.s)+'}$')
        
        plt.plot(t, totenergy,'k--',label='$E_{tot}$',)
        
        plt.plot(time_n, yintersec_n, 'ro')     #Transition point
        
        plt.legend(bbox_to_anchor=(1,1), loc="upper left")
        
        
        #Magnetization
        magne=np.zeros(np.size(aplot[0]))
        for i in range(dim):
            magne=magne+aplot[i]*(i-self.s)
            
            
        plt.subplot(234)
        plt.title('Cicle amb t')
        plt.xlabel('t')
        plt.ylabel('M')
        plt.plot(t, magne,'-')
        
        plt.subplot(236)
        plt.title('Cicle Histèresis')
        plt.xlabel('H')
        plt.ylabel('M')
        plt.plot(H0-self.alpha*t, magne,'-')
        
        #plt.show()
        plt.tight_layout()
        
        #THIS IS IN CASE THAT WE FINALLY WANNA PLOT VERTICAL LINES TO SHOW WHERE THE
        #TRANSITIONS SHOULD OCCUR
        #for i in range(self.s):
        #    plt.axvline(x=time_n[i],linestyle='--',color='grey')
        
        self.graphic_box1.remove_widget(self.plot)
        self.plot = FigureCanvasKivyAgg(plt.gcf())
        self.graphic_box1.add_widget(self.plot)
        self.plot.draw()
        
        print('DONE')

#APP BUILDING
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
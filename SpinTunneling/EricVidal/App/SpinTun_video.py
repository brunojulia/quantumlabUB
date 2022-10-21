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
from kivy.uix.image import Image, AsyncImage
from kivy.uix.floatlayout import FloatLayout 
from kivy.uix.gridlayout import GridLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.video import Video

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

#to build GIF
import os
import imageio

#This is the way to name the kv file as we want
Builder.load_file("SpinTun_kivy_video.kv")

#POPUP
class exp_P(FloatLayout):
    pass

def exp_popup():
    show = exp_P()
    popupWindow = Popup(title="Information", content=show, size_hint=(.7,.7))
    popupWindow.open()

#SCREENS
class Starting_Screen(Screen):
    pass

class Menu_Screen(Screen):
    pass

class Video_Screen(Screen, FloatLayout): 
    def __init__(self, **kwargs):
        super(Video_Screen, self).__init__(**kwargs)

class Experiment_Screen(Screen):
    angle = NumericProperty(0)
    angle2 = NumericProperty(0)
    
    def __init__(self,**kwargs):
        super(Experiment_Screen,self).__init__(**kwargs)
        
        #set empty graphic properties
        self.plot1, self.axs =plt.subplots(1)
        self.axs.set(xlim=[-50,50],ylim=[0,1])

        #Add the graphic to the box which it corresponds
        self.plot1 = FigureCanvasKivyAgg(self.plot1)
        self.graphic_box1.add_widget(self.plot1)
        
        #set empty graphic properties
        self.plot2, self.axs =plt.subplots(1)
        self.axs.set(xlim=[-50,50],ylim=[0,1])

        #Add the graphic to the box which it corresponds
        self.plot2 = FigureCanvasKivyAgg(self.plot2)
        self.graphic_box2.add_widget(self.plot2)
        
        #Predet values
        self.s = 1
        self.t0=-35
        self.tf = 35
        self.D = 5
        self.hz = 1
        self.B = 1
        #predet Hamiltonian
        self.ham=2
        
        self.screen_decider=1
        self.vel_decider=1
        
        #initiate counter
        self.count=0
        self.final=0
    
    def back(self):
        
        #Removes screen graphics, and reinitiates
        
        self.graphic_box1.remove_widget(self.plot1)
        self.graphic_box2.remove_widget(self.plot2)
        
        #set empty graphic properties
        self.plot1, self.axs =plt.subplots(1)
        self.axs.set(xlim=[-50,50],ylim=[0,1])

        #Add the graphic to the box which it corresponds
        self.plot1 = FigureCanvasKivyAgg(self.plot1)
        self.graphic_box1.add_widget(self.plot1)
        
        #set empty graphic properties
        self.plot2, self.axs =plt.subplots(1)
        self.axs.set(xlim=[-50,50],ylim=[0,1])

        #Add the graphic to the box which it corresponds
        self.plot2 = FigureCanvasKivyAgg(self.plot2)
        self.graphic_box2.add_widget(self.plot2)

    
    #Loading anim
    #I couldnt manage to have an animation during the
    #matplotlib plot is being produced
    def loading_anim(self):
            
            anim = Animation(angle = 360, duration=2) 
            anim += Animation(angle = 360, duration=2)
            anim.repeat = True
            anim.start(self)
    
    #so i did a done animation instead
    def done_anim(self):
            
            anim = Animation(angle2 = 360, duration=0.1)
            anim += Animation(angle = 360, duration=1.2)
            anim.repeat = False
            anim.start(self)
    
    def on_angle(self, item, angle):
        if angle == 360:
            item.angle = 0
        
    def popup_btn(self):
        exp_popup()
    def spinner_clicked_s(self, value):
        self.s = int(value)
    
    
    #BACKEND
    def send(self):
        plt.clf()
        #1. Initial parameters
        #Arbitrary spin to study
        #s = 3     #total spin
        self.dim=round(2*self.s+1)    #in order to work when s is half-integer with int dim
        Nterm1=self.s*(self.s+1)      #1st term of N+-

        #Time span
        At = [self.t0,self.tf]
        
        #this parameter is only used for H1
        self.H0 = (self.tf/2)*np.abs(self.hz)
        
        #IC
        a_m0=[]
        
        #EXPERIMENT
        if (self.screen_decider==2):
            a_m0.append(1+0j)
        
        for i in range(self.dim-1):
            a_m0.append(0+0j)
            
        #RESONANCE
        if (self.screen_decider==1):
            a_m0.append(1+0j)
                
        #Transition times
        time_n=[]

            
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
            #if (self.ham==1):
            #    eigenterm=am[k]*(-self.D*kreal**2+(self.H0-self.hz*t)*kreal)
            #same eigenvalues for H2 and H3
            #else:
            #    eigenterm=am[k]*(-self.D*kreal**2-self.hz*t*kreal)
            eigenterm=am[k]*(-self.D*kreal**2-self.hz*t*kreal)
            #summatory term
            sumterm=0
            
            if (self.ham==3):
                for m in range(self.dim):
                    #first we apply Kronicker deltas
                    if (k==(m+1)):
                        sumtermpos=Np(m)
                    else:
                        sumtermpos=0
                        
                    if (k==(m-1)):
                        sumtermneg=Nm(m)
                    else:
                        sumtermneg=0
                        
                    #and obtain summatory term along the for
                    sumterm += am[m]*(-self.B/2)*(sumtermpos+sumtermneg)
            else:
                for m in range(self.dim):
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
                    sumterm += am[m]*(self.B/2)*(sumtermpos+sumtermneg)
                
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
            for i in range(self.dim):
                system.append(dak1(i, t, a_m))
            
            return system
        
        #3. Resolution and plotting
        
        #solve
        a_m=solve_ivp(odes, At, a_m0, rtol=10**(-6),atol=10**(-7))
        
        #Plotting parameters
        self.t=a_m.t[:]  #time
        
        self.aplot=[]
        totenergy_temp=[]
        self.ener=[]
        
        for i in range(self.dim):
            totenergy_n=[]
            prob_i=np.abs(a_m.y[i,:])**2
            self.aplot.append(prob_i)     #Probabilities coeff^2
            
            #Energies Analytical
            #if (self.ham==1):
            #    ener_temp=-self.D*(i-self.s)**2+(self.H0-self.hz*self.t)*(i-self.s)
            #as the eigenvalue is the same for H2 and H3, so as too the energy
            #else:
            #    ener_temp=-self.D*(i-self.s)**2-self.hz*self.t*(i-self.s)
            ener_temp=-self.D*(i-self.s)**2-self.hz*self.t*(i-self.s)
            self.ener.append(ener_temp) 
            
            for j in range(len(self.t)):
                totenergy_n.append(prob_i[j]*ener_temp[j])
            totenergy_temp.append(totenergy_n)
        
        self.norm = np.sum(self.aplot, axis=0)    #self.norm (sum of probs)
        self.totenergy = np.sum(totenergy_temp, axis=0)    #Total energy (sum of each spin energy*prob)
        
        
            
        #Plot1
        plt.close(1)
        plt.figure(1)
        plt.title('Spin probabilties') #General spin method, solve_ivp
        plt.xlabel('t')
        plt.ylabel('$|a|^2$')
        plt.axhline(y=1.0,linestyle='--',color='grey')  #to compare norm
        plt.xlim([self.t0, self.tf])
        for i in range(self.dim):
            plt.plot(self.t, self.aplot[i],'-',label='m='+str(i-self.s))
        
        plt.plot(self.t, self.norm,'-',label='norm')
        
        plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
        
        self.graphic_box1.remove_widget(self.plot1)
        self.plot1 = FigureCanvasKivyAgg(plt.gcf())
        self.graphic_box1.add_widget(self.plot1)
        self.plot1.draw()
        
        #Plot2
        plt.close(2)
        plt.figure(2)
        plt.title('States energies if $\mathcal{H}_0$')
        plt.xlabel('t')
        plt.ylabel('$E$')
        plt.xlim([self.t0, self.tf])
        for i in range(self.dim):
            plt.plot(self.t, self.ener[i],'-',label='$E_{'+str(i-self.s)+'}$')
        
        plt.plot(self.t, self.totenergy,'k--',label='$E_{tot}$',)
        
        plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
 
        #THIS IS IN CASE THAT WE FINALLY WANNA PLOT VERTICAL LINES TO SHOW WHERE THE
        #TRANSITIONS SHOULD OCCUR
        #for i in range(self.s):
        #    plt.axvline(x=time_n[i],linestyle='--',color='grey')
        
        self.graphic_box2.remove_widget(self.plot2)
        self.plot2 = FigureCanvasKivyAgg(plt.gcf())
        self.graphic_box2.add_widget(self.plot2)
        self.plot2.draw()

    def video(self):
        #reference to stop gif
        self.final=len(self.t)
        self.count=0
        
        #where to save images
        filenames=[]
        for self.count in np.arange(0,self.final,200):
            
            #Plot1
            plt.figure()
            plt.subplot(121)
            plt.title('Spin probabilties') #General spin method, solve_ivp
            plt.xlabel('t')
            plt.ylabel('$|a|^2$')
            plt.axhline(y=1.0,linestyle='--',color='grey')
            plt.xlim([self.t0, self.tf])
            #Probabilities
            for i in range(self.dim):
                temp_plot=self.aplot[i]
                plt.plot(self.t[:self.count], temp_plot[:self.count],'-',label='m='+str(i-self.s))
            
            #for the gif maybe it is better not to plot the norm and legend
            #plt.plot(self.t[:self.count], self.norm[:self.count],'-',label='self.norma')
            #plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
            
            
            #Plot2
            plt.subplot(122)
            plt.title('States energies if $\mathcal{H}_0$')
            plt.xlabel('t')
            plt.ylabel('$E$')
            plt.xlim([self.t0, self.tf])
            for i in range(self.dim):
                temp_ener=self.ener[i]
                plt.plot(self.t, temp_ener, '-', color='C'+str(i),  label='$E_{'+str(i-self.s)+'}$')
                
                #Prob balls
                temp_plot=self.aplot[i] #prob that determines each size
                plt.plot(self.t[self.count], temp_ener[self.count], 'o',color='C'+str(i), mew=10*temp_plot[self.count], ms=20*temp_plot[self.count])
                
            plt.plot(self.t[:self.count], self.totenergy[:self.count],'k--',label='$E_{tot}$',)
            
            #for the gif maybe it is better without legend
            #plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
    
            #THIS IS IN CASE THAT WE FINALLY WANNA PLOT VERTICAL LINES TO SHOW WHERE THE
            #TRANSITIONS SHOULD OCCUR
            #for i in range(self.s):
            #    plt.axvline(x=time_n[i],linestyle='--',color='grey')
            
            # create file name and append it to a list
            filename = 'im{:.0f}.png'.format(self.count)
            filenames.append(filename)
            # save frame
            plt.savefig(filename)
            plt.close()
            
            # build gif
        with imageio.get_writer('videos/prova.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)
                
        # Remove files
        for filename in set(filenames):
            os.remove(filename)

    def play(self):
        #(re)initialize the counter
        def dynamic_plots(dt):
            if (self.vel_decider==1):
                self.vel=75
            elif (self.vel_decider==2):
                self.vel=150
            elif (self.vel_decider==4):
                self.vel=300
            
            self.count += self.vel    #kind of step of time/frame plotted
            
            #reference to stop gif
            self.final=len(self.t)

            if self.count >= self.final:
                #reinitialize variable
                self.count=0
                #put the same graphics as before the gif
                #Plot1
                plt.close(1)
                plt.figure(1)
                plt.title('Spin probabilties') #General spin method, solve_ivp
                plt.xlabel('t')
                plt.ylabel('$|a|^2$')
                plt.axhline(y=1.0,linestyle='--',color='grey')
                plt.xlim([self.t0, self.tf])
                #Probabilities
                for i in range(self.dim):
                    plt.plot(self.t, self.aplot[i],'-',label='m='+str(i-self.s))
                plt.plot(self.t, self.norm,'-',label='norm')
                
                plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
                
                
                self.graphic_box1.remove_widget(self.plot1)
                self.plot1 = FigureCanvasKivyAgg(plt.gcf())
                self.graphic_box1.add_widget(self.plot1)
                self.plot1.draw()
                
                
                
                #Plot2
                plt.close(2)
                plt.figure(2)
                plt.title('States energies if $\mathcal{H}_0$')
                plt.xlabel('t')
                plt.ylabel('$E$')
                plt.xlim([self.t0, self.tf])
                for i in range(self.dim):
                    plt.plot(self.t, self.ener[i],'-',label='$E_{'+str(i-self.s)+'}$')
                
                plt.plot(self.t, self.totenergy,'k--',label='$E_{tot}$',)
                
                plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
         
                
                #THIS IS IN CASE THAT WE FINALLY WANNA PLOT VERTICAL LINES TO SHOW WHERE THE
                #TRANSITIONS SHOULD OCCUR
                #for i in range(self.s):
                #    plt.axvline(x=time_n[i],linestyle='--',color='grey')
                
                self.graphic_box2.remove_widget(self.plot2)
                self.plot2 = FigureCanvasKivyAgg(plt.gcf())
                self.graphic_box2.add_widget(self.plot2)
                self.plot2.draw()
                
                #to stop the Clock loop:
                return False
            
            #GIF
            
            
            #Plot1
            plt.close(1)
            plt.figure(1)
            plt.title('Spin probabilties') #General spin method, solve_ivp
            plt.xlabel('t')
            plt.ylabel('$|a|^2$')
            plt.axhline(y=1.0,linestyle='--',color='grey')
            plt.xlim([self.t0, self.tf])
            #Probabilities
            for i in range(self.dim):
                temp_plot=self.aplot[i]
                plt.plot(self.t[:self.count], temp_plot[:self.count],'-',label='m='+str(i-self.s))
            
            #for the gif maybe it is better not to plot the norm and legend
            #plt.plot(self.t[:self.count], self.norm[:self.count],'-',label='self.norma')
            #plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
        
        
            self.graphic_box1.remove_widget(self.plot1)
            self.plot1 = FigureCanvasKivyAgg(plt.gcf())
            self.graphic_box1.add_widget(self.plot1)
            self.plot1.draw()
            
            
            #Plot2
            plt.close(2)
            plt.figure(2)
            plt.title('States energies if $\mathcal{H}_0$')
            plt.xlabel('t')
            plt.ylabel('$E$')
            plt.xlim([self.t0, self.tf])
            for i in range(self.dim):
                temp_ener=self.ener[i]
                plt.plot(self.t, temp_ener, '-', color='C'+str(i),  label='$E_{'+str(i-self.s)+'}$')
                
                #Prob balls
                temp_plot=self.aplot[i] #prob that determines each size
                plt.plot(self.t[self.count], temp_ener[self.count], 'o',color='C'+str(i), mew=10*temp_plot[self.count], ms=20*temp_plot[self.count])
                
            plt.plot(self.t[:self.count], self.totenergy[:self.count],'k--',label='$E_{tot}$',)
            
            #for the gif maybe it is better without legend
            #plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
     
            #THIS IS IN CASE THAT WE FINALLY WANNA PLOT VERTICAL LINES TO SHOW WHERE THE
            #TRANSITIONS SHOULD OCCUR
            #for i in range(self.s):
            #    plt.axvline(x=time_n[i],linestyle='--',color='grey')
            
            self.graphic_box2.remove_widget(self.plot2)
            self.plot2 = FigureCanvasKivyAgg(plt.gcf())
            self.graphic_box2.add_widget(self.plot2)
            self.plot2.draw()
            
        
        
        #function that creates the loop until a False is returned
        self.event=Clock.schedule_interval(dynamic_plots, 10**(-9))
        self.event

    def pause(self):
        if (self.count != 0):    #in order to stop event while back, unless we do this, when it is not playing an error occurs because ther is no event ongoing
            self.event.cancel()
    
    def reset(self):
        self.event.cancel()
        self.count=0
        
        #put the same graphics as before the gif
        #Plot1
        plt.close(1)
        plt.figure(1)
        plt.title('Spin probabilties') #General spin method, solve_ivp
        plt.xlabel('t')
        plt.ylabel('$|a|^2$')
        plt.axhline(y=1.0,linestyle='--',color='grey')
        
        #Probabilities
        for i in range(self.dim):
            plt.plot(self.t, self.aplot[i],'-',label='m='+str(i-self.s))
        plt.plot(self.t, self.norm,'-',label='norm')
        
        plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
        
        
        self.graphic_box1.remove_widget(self.plot1)
        self.plot1 = FigureCanvasKivyAgg(plt.gcf())
        self.graphic_box1.add_widget(self.plot1)
        self.plot1.draw()
        
        
        
        #Plot2
        plt.close(2)
        plt.figure(2)
        plt.title('States energies if $\mathcal{H}_0$')
        plt.xlabel('t')
        plt.ylabel('$E$')
        for i in range(self.dim):
            plt.plot(self.t, self.ener[i],'-',label='$E_{'+str(i-self.s)+'}$')
        
        plt.plot(self.t, self.totenergy,'k--',label='$E_{tot}$',)
        
        plt.legend(bbox_to_anchor=(0.95,1), loc="upper left")
 
        
        #THIS IS IN CASE THAT WE FINALLY WANNA PLOT VERTICAL LINES TO SHOW WHERE THE
        #TRANSITIONS SHOULD OCCUR
        #for i in range(self.s):
        #    plt.axvline(x=time_n[i],linestyle='--',color='grey')
        
        self.graphic_box2.remove_widget(self.plot2)
        self.plot2 = FigureCanvasKivyAgg(plt.gcf())
        self.graphic_box2.add_widget(self.plot2)
        self.plot2.draw()


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
        sm.add_widget(Experiment_Screen(name='Experiment'))
        sm.add_widget(Video_Screen(name='Video'))
        
        #Changes Starting screen to Menu
        def intro(self, *largs):
            sm.current = 'Menu'

        #It triggers the screen switch and controls the time that the Starting Screen is showed
        Clock.schedule_once(intro, 3) #4 may be the most appropriate time
        
        return sm

if __name__ == '__main__':
    Window.maximize()   #opens in full screen
    SpinTunApp().run()  #run method inside App class
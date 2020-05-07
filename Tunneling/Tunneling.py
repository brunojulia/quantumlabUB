# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:29:04 2020

@author: usuari
"""
import kivy
import numpy as np
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.popup import Popup
from kivy.graphics import Color, Ellipse,Line,Rectangle
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout


class TunnelingApp(App):
    def build(self):
        self.title = 'Tunneling'
        return StartingScreen()
    

    
class StartingScreen(Screen):
    
    
    
    def  __init__(self, **kwargs):
    
        super(StartingScreen, self).__init__(**kwargs)

        
        
        self.posx_red = 60
        self.posy_red = 60
        self.posx_blue = 110
        self.posy_blue = 110
        self.act = False
        
        self.draw_redcircle()
        self.draw_bluecircle()
        self.tauler()
        
        #Ara creem els cercles:
        
        
    def draw_redcircle(self):
        
        with self.box1.canvas:
            Color(1,0,0,mode = 'rgb')
            Ellipse(pos=(self.posx_red,self.posy_red),size = (30,30))
            
            
        
        self.red_particle = FloatLayout()
        self.box1.add_widget(self.red_particle)
        
        return self.red_particle
        
        
    def draw_bluecircle(self):
    
        with self.box1.canvas:
            Color(0,0,1,mode = 'rgb')
            self.bluecircle = Ellipse(pos=(self.posx_blue,self.posy_blue),size=(30,30))
            
        self.blue_particle = FloatLayout()
        self.box1.add_widget(self.blue_particle)
        
        
            
        return self.blue_particle

    #Ara dibuixem el tauler
    def tauler(self):
        
        with self.box1.canvas:
            Color(0,1,0,mode='rgb')
            for i in range(0,10):
                Line(rectangle=(50 + 50*i,50,50,500))
                Line(rectangle=(50, 50 + 50*i,500,50))
            Color(0,1,1,mode='rgb')
            Line(rectangle = (50,50,500,500))
        
        self.rectangle1 = FloatLayout()
        self.box1.add_widget(self.rectangle1)
        
        return self.rectangle1
    
    def down_red(self):
        if self.posy_red == 60:
            self.posy_red = self.posy_red
        
        else:
            self.posy_red = self.posy_red - 50
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        
     
    def down_blue(self):
        if self.posy_blue == 60:
            self.posy_blue = self.posy_blue
        
        else:
            self.posy_blue = self.posy_blue - 50
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
    
    def up_blue(self):
        if self.posy_blue == 510:
            self.posy_blue = self.posy_blue
        
        else:
            self.posy_blue = self.posy_blue + 50
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        
    def up_red(self):
        if self.posy_red == 510:
            self.posy_red = self.posy_red
        
        else:
            self.posy_red = self.posy_red + 50
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
    
    def right_red(self):
        if self.posx_red == 510:
            self.posx_red = self.posx_red
        
        else:
            self.posx_red = self.posx_red + 50
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
     
    def right_blue(self):
        if self.posx_blue == 510:
            self.posx_blue = self.posx_blue
        
        else:
            self.posx_blue = self.posx_blue + 50
    
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        
    def left_red(self):
        if self.posx_red == 60:
            self.posx_red = self.posx_red
        
        else:
            self.posx_red = self.posx_red - 50
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
    
    def left_blue(self):
        if self.posx_blue == 60:
            self.posx_blue = self.posx_blue
        
        else:
            self.posx_blue = self.posx_blue - 50
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        
        
        
        
        
        
class TunnelPopup(Popup):
    
    def __init__(self):

        super(Popup, self).__init__()  
        
        self.gpseudo_init()

    def gpseudo_init(self): 
        
        self.hbar = 1.0
        self.massa = 1.0
        self.lx_max = 17.0
        self.lx_min = -17.0
        self.nx = 500
        self.dx = (self.lx_max - self.lx_min)/float(self.nx)
        self.xx = np.arange(self.lx_min,self.lx_max + self.dx,self.dx)
        self.dt = 0.01
        self.tmax = 1.5
        self.temps = 0.
        self.nt = (self.tmax/self.dt)+1
        self.r = 1j*self.dt/(2.0*(self.dx**2))
        
        self.p0 = -200.0/self.lx_max
        self.bwidth = 0.2
        self.bheight_quadrat = 76.0
        self.bheight_gauss = 0.1
        self.potencial()
        self.control_pot = 'quadrat'
        self.max_pot = 90.0
        self.min_pot = 0.0
        self.phi0()
        self.phi02 = abs(self.phi0)**2
    
        
        self.colpot = 'blue'
        self.cole = 'green'
        self.colphi = 'red'
  
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('white')
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas)
            
        self.pot_graf = plt.subplot()
        self.pot_graf.set_facecolor('black')
        self.pot_graf.axis([self.lx_min, self.lx_max, self.min_pot, self.max_pot])
        self.pot_data, = self.pot_graf.plot(self.xx, self.pot,color = self.colpot)
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.cole)
        
        self.ona = self.pot_graf.twinx()
        self.visu_ona, = self.ona.plot(self.xx,self.phi02, color = self.colphi)
        self.ona.axis([self.lx_min,self.lx_max,0.0,1.0])
        
        #"Pantalla" on es veurè el coeficient de transmissió analític
        self.pant1 = plt.figure()
        self.pant1.patch.set_facecolor('white')
        self.pant1_canvas = FigureCanvasKivyAgg(self.pant1)
        self.box2.add_widget(self.pant1_canvas)
        
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        
        #"Pantalla" on es veurà la probabilotat de l'ona d'estar a l'esquerra de la barrera
        self.pant2 = plt.figure()
        self.pant2.patch.set_facecolor('white')
        self.pant2_canvas = FigureCanvasKivyAgg(self.pant2)
        self.box3.add_widget(self.pant2_canvas)
        
        self.prob_left_txt = self.pant2.text(0.1,0.5,'0.0')
        
        self.pant3 = plt.figure()
        self.pant3.patch.set_facecolor('white')
        self.pant3_canvas = FigureCanvasKivyAgg(self.pant3)
        self.box4.add_widget(self.pant3_canvas)
        
        self.norma_txt = self.pant3.text(0.1,0.5,'1.0')
        
                                    
        self.main_fig.tight_layout() 
        self.main_canvas.draw()

        
    def potencial(self):
        
        #self.pot = 42.55*(np.exp(-(self.xx/self.bwidth)**2))/np.sqrt(self.bheight_gauss*np.pi)
        self.pot = np.zeros(len(self.xx))
        for i in range(0,len(self.xx)):
            if -self.bwidth/2.0 < self.xx[i] < self.bwidth/2.0:
                self.pot[i] = self.bheight_quadrat
                
    def canvi_potencial_gauss(self):
        
        self.pot = 42.55*(np.exp(-(self.xx/self.bwidth)**2))/np.sqrt(self.bheight_gauss*np.pi)
        self.control_pot  = 'gauss'
        
        self.pot_data.remove()
        self.pot_data, = self.pot_graf.plot(self.xx, self.pot,color=self.colpot)
        
        self.T_analitic()
        self.T_txt.remove()
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        
        self.main_canvas.draw()
        self.pant1_canvas.draw()
        
    def canvi_potencial_quadrat(self):
        
        self.pot = np.zeros(len(self.xx))
        for i in range(0,len(self.xx)):
            if -self.bwidth/2.0 < self.xx[i] < self.bwidth/2.0:
                self.pot[i] = self.bheight_quadrat
            
        self.pot_data.remove()
        self.pot_data, = self.pot_graf.plot(self.xx, self.pot,color=self.colpot)
        
        self.T_analitic()
        self.T_txt.remove()
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        
        self.main_canvas.draw()
        self.pant1_canvas.draw()
        
            

    def phi0(self):
        
        self.phi0 = 1j*np.zeros(len(self.xx))
        self.phi0 = (1.0/(2.0*np.pi)**(1.0/4.0))*np.exp((-(self.xx - 7.0)**2)/4.0)*np.exp(1j*self.xx*self.p0)
        self.energia()
        self.T_analitic()
    
    def energia(self):
        
        self.matriu_H = np.zeros((self.nx+1,self.nx+1))
        k = 1.0/(self.dx**2)
        for i in range(0,self.nx+1):
            self.matriu_H[i,i] = self.pot[i] + k
            if i != self.nx:
                self.matriu_H[i,i+1] = -(k/2.0)
                self.matriu_H[i+1,i] = -(k/2.0)
    
        
        self.phihphi = (np.conjugate(self.phi0)*(np.dot(self.matriu_H,self.phi0)))
        
        self.E = self.simpson2(self.dx,self.phihphi)
        
        self.E_vect = np.zeros((self.nx+1))
        
        for i in range(0,self.nx+1):
            self.E_vect[i] = self.E
        

    def simpson2(self,h,vect):
        
        #add the extrems
        add=(vect[0]+vect[len(vect)-1])

        #add each parity its factor
        for i in np.arange(2,len(vect)-1,2):
            add+=2*vect[i]

        for i in np.arange(1,len(vect)-1,2):
            add+=4*vect[i]

        #add the global factor
        add*=(h/np.float(3))

        return add 
        
    def add_width(self):
        
        self.pot_data.remove()
        self.T_txt.remove()
        self.bwidth = self.bwidth + 0.05
        self.potencial()
        self.T_analitic()
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        self.pot_data, = self.pot_graf.plot(self.xx, self.pot,color=self.colpot)
        
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()

        
    def subtract_width(self):
        
        self.pot_data.remove()
        self.T_txt.remove()
        self.bwidth = self.bwidth - 0.05
        self.potencial()
        self.T_analitic()
        self.pot_data, = self.pot_graf.plot(self.xx, self.pot,color=self.colpot)
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()
        
    def add_bheight(self):
        
        self.pot_data.remove()
        self.T_txt.remove()
        self.bheight_quadrat = self.bheight_quadrat + 0.5
        self.potencial()
        self.T_analitic()
        self.pot_data, = self.pot_graf.plot(self.xx, self.pot,color=self.colpot)
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()
        
    def subtract_bheight(self):
        
        self.pot_data.remove()
        self.T_txt.remove()
        self.bheight_quadrat = self.bheight_quadrat - 0.5
        self.potencial()
        self.T_analitic()
        self.pot_data, = self.pot_graf.plot(self.xx, self.pot,color=self.colpot)
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()

        
    def T_analitic(self):
        
        self.arrel = []
        
        for i in range(0,self.nx+1):
            if self.pot[i] > self.E:
                self.arrel = np.append(self.arrel,np.sqrt(2.0*(self.pot[i]-self.E)))
        
        
        integral = self.simpson2(self.dx,self.arrel)
        #integral = (self.bwidth*np.sqrt(76.0-self.E))
        
        self.T = np.abs((np.exp((-2.0)*integral)))
        
        
        
        
    def inici_evolucio(self):
        
        self.ev = Clock.schedule_interval(self.evolucio,self.dt)

        #Es crearan els vectors i matriu necessaris per la evolucio
        self.abc()
        self.matriuB()
        self.phii = 1j*np.zeros(self.nx-1)
        for i in range(1,self.nx-1):
            self.phii[i] = self.phi0[i]
        self.phi = 1j*np.zeros(self.nx+1)
        self.phi2 = np.zeros(self.nx + 1)
        
        self.t = 0
        
        
    def abc(self):
        
        self.b = 1j*np.zeros(self.nx-1)
        self.a = 1j*np.zeros(self.nx-1)
        self.c = 1j*np.zeros(self.nx-1)
        
        for i in range(0,self.nx-1):
            self.b[i] = 2.0*(1.0 + self.r) + 1.0j*self.pot[i]*self.dt
            if i != 0:
                self.a[i] = -self.r
            if i != self.nx-2:
                self.c[i] = -self.r
        
        
    def stop_evolucio(self):
        
        self.ev.cancel()
    
    def matriuB(self):
        
        self.matriu_B = 1j*np.zeros((self.nx -1,self.nx-1))
        for i in range(0,self.nx-1):
            self.matriu_B[i,i] = 2.0*(1.0 -self.r) - 1.0j*self.pot[i]*self.dt
            if i != (self.nx-2):
                self.matriu_B[i,i+1] = self.r
                self.matriu_B[i+1,i] = self.r
                
    def prob_left(self):
        
        n = int(self.nx/2)
        
        self.probleft = self.simpson2(self.dx,self.phi2[0:n])/self.simpson2(self.dx,self.phi2)
        
    def calc_norma(self):
        
        self.norma = self.simpson2(self.dx,self.phi2)
        
    def evolucio(self,dt):
        
        self.abc()
        self.matriuB()
        
        self.t += 100*self.dt
    
        self.d = np.dot(self.matriu_B,self.phii)
        
        # Apliquem condicions de contorn
        self.d[0] = self.d[0] + 2.0*self.r*self.phi0[0]
        self.d[self.nx-2] = self.d[self.nx-2] + 2.0*self.r*self.phi0[self.nx]
        
        
        # Cridem la subrutina tridiag que ens calculi phi pel temsp següent
        self.tridiag()
        
        self.phi[1:self.nx] = self.phii
        self.phi[0] = self.phi0[0]
        self.phi[self.nx] = self.phi0[self.nx]
        
        for i in range(0,self.nx+1):
            self.phi2[i] = (np.abs(self.phi[i]))**2
        
        #Ara per a cada temps ha de esborrar la figura anterior i 
        #dibuixar la nova
        
        self.visu_ona.remove()
        self.visu_ona, = self.ona.plot(self.xx,self.phi2,color = self.colphi)
        
        self.main_canvas.draw()
        
        if np.mod(self.t,5.0) == 0.0:
            #Cridem la funció que integra la funcio d'ona a l'esquerra
            self.prob_left()
        
            self.prob_left_txt.remove()
            self.prob_left_txt = self.pant2.text(0.1,0.5,str(self.probleft))
        
            #Cridem la funció que ens calcula la norma
            self.calc_norma()
        
            self.norma_txt.remove()
            self.norma_txt = self.pant3.text(0.1,0.5,str(self.norma))
        
            self.pant2_canvas.draw()
            self.pant3_canvas.draw()
            
        
        
        
    def tridiag(self):
    
        n = np.size(self.a)

        cp = np.zeros(n, dtype = np.complex)
        dp = np.zeros(n, dtype = np.complex)
        cp[0] = self.c[0]/self.b[0]
        dp[0] = self.d[0]/self.b[0]

        for i in range(1,n):
            m = (self.b[i]-self.a[i]*cp[i-1])
            cp[i] = self.c[i]/m
            dp[i] = (self.d[i] - self.a[i]*dp[i-1])/m

        self.phii[n-1] = dp[n-1]

        for j in range(1,n):
            i = (n-1)-j
            self.phii[i] = dp[i]-cp[i]*self.phii[i+1]

    
    def reset(self):
        
        self.stop_evolucio()
        self.visu_ona.remove()
        self.visu_ona, = self.ona.plot(self.xx,self.phi02, color = self.colphi)
        
        self.norma_txt.remove()
        self.prob_left_txt.remove()        
        self.prob_left_txt = self.pant2.text(0.1,0.5,'0.0')
        self.norma_txt = self.pant3.text(0.1,0.5,'1.0')
        
        self.pant2.canvas.draw()
        self.pant3.canvas.draw()
        self.main_canvas.draw()
        
      
    
            
        
TunnelingApp().run()
        
    

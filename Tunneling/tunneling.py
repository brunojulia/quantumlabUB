# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:29:04 2020

@author: usuari
"""
import kivy
import numpy as np
import matplotlib.pyplot as plt
import random
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.popup import Popup
from kivy.graphics import Color, Ellipse,Line,Rectangle
from kivy.clock import Clock
from kivy.uix.floatlayout import FloatLayout
from kivy.properties import NumericProperty,StringProperty
from kivy.core.window import Window
import GlobalShared
from kivy.core.window import WindowBase
from kivy.uix.label import Label


class TunnelingApp(App):
    def build(self):
        self.title = 'Tunneling'
        return StartingScreen()
    

    
class StartingScreen(Screen):
       
    def  __init__(self):
    
        super(StartingScreen, self).__init__()
        self.inicial()
    
    def inicial(self):

        
        self.lx_max = 17.0
        self.lx_min = -17.0
        self.nx = 500
        self.dx = (self.lx_max - self.lx_min)/float(self.nx)
        self.xx = np.arange(self.lx_min,self.lx_max + self.dx,self.dx)
        self.p0 = -200.0/self.lx_max
        
        #Creem una llista de possibles amplades inicials
        self.list_amp = np.zeros(10)
        for i in range(0,10):
            self.list_amp[i] = 0.2 + i*0.1
        
        self.botons_amp = np.zeros(4)
        self.botons_T = np.zeros(4)
        for i in range(0,4):
            self.botons_amp[i] = random.choice(self.list_amp)
            self.amplada = self.botons_amp[i]
            self.potencial()
            self.ona()
            self.botons_T[i] = self.T
            
        self.bonus = [1.2,1.5,2.0,5.0,10.0]
        self.posx_rand = [25,75,125,175,225,275,325,375,425,475]
        self.posy_rand = [25,75,125,175,225,275,325,375,425,475]
        self.nbonus = 4
        self.matriu_bonus = np.zeros((self.nbonus,3))
        
        for i in range(0,self.nbonus):
            
            self.matriu_bonus[i,0] = random.choice(self.posx_rand)
            self.matriu_bonus[i,1] = random.choice(self.posy_rand)
            self.matriu_bonus[i,2] = random.choice(self.bonus)
        
        
        self.amplada = 0.1
        self.posx_red = 60
        self.posy_red = 60
        self.posx_blue = 510
        self.posy_blue = 60
        self.posx_end = 300
        self.posy_end = 500
        self.angle_start_red = 0
        self.angle_end_red = 360
        self.angle_start_blue = 0
        self.angle_end_blue = 360
        
        self.red_prob = 1.0
        self.blue_prob = 1.0
        
        self.torn = 'red'
        self.control_botons = False
        
        self.draw_redcircle()
        self.draw_bluecircle()
        self.draw_endpoint()
        self.tauler()
        self.pantalles()
        self.write_bonus()
        
        
    def pantalles(self):
        
        
        self.pantup = plt.figure()
        self.pantup.patch.set_facecolor('black')
        self.pantup_canvas =  FigureCanvasKivyAgg(self.pantup)
        self.box3.add_widget(self.pantup_canvas)
        
        self.up_txt = self.pantup.text(0.1,0.1,'%.6f' % (self.botons_T[1]),color='white')
        
        self.pantdown = plt.figure()
        self.pantdown.patch.set_facecolor('black')
        self.pantdown_canvas =  FigureCanvasKivyAgg(self.pantdown)
        self.box6.add_widget(self.pantdown_canvas)
        
        self.down_txt = self.pantdown.text(0.1,0.1,'%.6f' % (self.botons_T[0]),color='white')
        
        self.pantright = plt.figure()
        self.pantright.patch.set_facecolor('black')
        self.pantright_canvas =  FigureCanvasKivyAgg(self.pantright)
        self.box7.add_widget(self.pantright_canvas)
        
        self.right_txt = self.pantright.text(0.1,0.1,'%.6f' % (self.botons_T[2]),color='white')
        
        self.pantleft = plt.figure()
        self.pantleft.patch.set_facecolor('black')
        self.pantleft_canvas =  FigureCanvasKivyAgg(self.pantleft)
        self.box8.add_widget(self.pantleft_canvas)
        
        self.left_txt = self.pantleft.text(0.1,0.1, '%.6f' % (self.botons_T[3]),color='white')
        
        self.pantprob_red = plt.figure()
        self.pantprob_red.patch.set_facecolor('red')
        self.pantprob_red_canvas =  FigureCanvasKivyAgg(self.pantprob_red)
        self.box4.add_widget(self.pantprob_red_canvas)
        
        self.prob_txt = self.pantprob_red.text(0.1,0.5,str(self.red_prob))
        
        self.pantprob_blue = plt.figure()
        self.pantprob_blue.patch.set_facecolor('blue')
        self.pantprob_blue_canvas =  FigureCanvasKivyAgg(self.pantprob_blue)
        self.box2.add_widget(self.pantprob_blue_canvas)
        
        self.probb_txt = self.pantprob_blue.text(0.1,0.5,str(self.blue_prob))
        
        self.pantorn = plt.figure()
        self.pantorn.patch.set_facecolor('black')
        self.pantorn_canvas = FigureCanvasKivyAgg(self.pantorn)
        self.box5.add_widget(self.pantorn_canvas)
        
        self.torn_txt = self.pantorn.text(0.1,0.5,"RED PARTICLE'S TURN!",color=self.torn)
        
        
    def draw_redcircle(self):
        
        with self.box1.canvas:
            Color(1,0,0,mode = 'rgb')
            Ellipse(pos=(self.posx_red,self.posy_red),size = (30,30),
                    angle_start = self.angle_start_red, angle_end = self.angle_end_red)
                 
        
        self.red_particle = FloatLayout()
        self.box1.add_widget(self.red_particle)
        
        return self.red_particle
        
        
    def draw_bluecircle(self):
    
        with self.box1.canvas:
            Color(0,0,1,mode = 'rgb')
            self.bluecircle = Ellipse(pos=(self.posx_blue,self.posy_blue),size=(30,30),
                                      angle_start = self.angle_start_blue, angle_end = self.angle_end_blue)
            
        self.blue_particle = FloatLayout()
        self.box1.add_widget(self.blue_particle)
        
        
            
        return self.blue_particle
    
    def draw_endpoint(self):
        
        with self.box1.canvas:
            Color(0,1,0,mode='rgb')
            self.endpoint = Rectangle(pos= (self.posx_end,self.posy_end),size=(50,50))
        
        self.endpoint = FloatLayout()
        self.box1.add_widget(self.endpoint)
        
        return self.endpoint
    
    def write_bonus(self):
        self.write_bonus0()
        self.write_bonus1()
        self.write_bonus2()
        self.write_bonus3()
        
    
    def write_bonus0(self):
        
        with self.box1.canvas:
            self.bonus0 = Label(text= 'x' + (str(self.matriu_bonus[0,2])), 
                                  pos=(int(self.matriu_bonus[0,0]),int(self.matriu_bonus[0,1])))
        
        self.bonus0 = FloatLayout()
        self.box1.add_widget(self.bonus0)
        
        return self.bonus0
    
    def write_bonus1(self):
        
        with self.box1.canvas:
            self.bonus1 = Label(text= 'x' + (str(self.matriu_bonus[1,2])), 
                                  pos=(int(self.matriu_bonus[1,0]),int(self.matriu_bonus[1,1])))
        
        self.bonus1 = FloatLayout()
        self.box1.add_widget(self.bonus1)
        
        return self.bonus1
    
    def write_bonus2(self):
        
        with self.box1.canvas:
            self.bonus2 = Label(text= 'x' + (str(self.matriu_bonus[2,2])), 
                                  pos=(int(self.matriu_bonus[2,0]),int(self.matriu_bonus[2,1])))
        
        self.bonus2 = FloatLayout()
        self.box1.add_widget(self.bonus2)
        
        return self.bonus2
    
    def write_bonus3(self):
        
        with self.box1.canvas:
            self.bonus3 = Label(text= 'x' + (str(self.matriu_bonus[3,2])), 
                                  pos=(int(self.matriu_bonus[3,0]),int(self.matriu_bonus[3,1])))
        
        self.bonus3 = FloatLayout()
        self.box1.add_widget(self.bonus3)
        
        return self.bonus3


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
        
        if self.torn == 'red' and self.control_botons == False:
            self.amplada = self.botons_amp[0]
        
            if self.posy_red == 60:
                self.posy_red = self.posy_red
        
            else:
                self.posy_red = self.posy_red - 50
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.control_botons = True



    def down_blue(self):
        
        if self.torn == 'blue' and self.control_botons == False:
            self.amplada = self.botons_amp[0]
        
            if self.posy_blue == 60:
                self.posy_blue = self.posy_blue
        
            else:
                self.posy_blue = self.posy_blue - 50
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.control_botons = True

            
    def up_blue(self):
        
        if self.torn == 'blue' and self.control_botons == False:
            self.amplada = self.botons_amp[1]
        
            if self.posy_blue == 510:
                self.posy_blue = self.posy_blue
        
            else:
                self.posy_blue = self.posy_blue + 50
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.control_botons = True
        
        
    def up_red(self):
        
        if self.torn == 'red' and self.control_botons == False:
            self.amplada = self.botons_amp[1]
        
            if self.posy_red == 510:
                self.posy_red = self.posy_red
        
            else:
                self.posy_red = self.posy_red + 50
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.control_botons = True
            
    
    def right_red(self):
        
        if self.torn == 'red' and self.control_botons == False:
            self.amplada = self.botons_amp[2]
        
            if self.posx_red == 510:
                self.posx_red = self.posx_red
        
            else:
                self.posx_red = self.posx_red + 50
            
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.control_botons = True
            
    def right_blue(self):
        
        if self.torn == 'blue' and self.control_botons == False:
            self.amplada = self.botons_amp[2]
        
            if self.posx_blue == 510:
                self.posx_blue = self.posx_blue
        
            else:
                self.posx_blue = self.posx_blue + 50
    
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
        
            self.control_botons = True
            
    def left_red(self):
        
        if self.torn == 'red' and self.control_botons == False:
            self.amplada = self.botons_amp[3]
        
            if self.posx_red == 60:
                self.posx_red = self.posx_red
        
            else:
                self.posx_red = self.posx_red - 50
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.control_botons = True

            
    def left_blue(self):
        
        if self.torn =='blue' and self.control_botons == False:
            self.amplada = self.botons_amp[3]
        
            if self.posx_blue == 60:
                self.posx_blue = self.posx_blue
        
            else:
                self.posx_blue = self.posx_blue - 50
                
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.control_botons = True
            
        
    def canvi_torn(self):
        
        if self.torn == 'red':
            self.torn = 'blue'
            
            self.torn_txt.remove()
            self.pantorn.patch.set_facecolor('black')
            self.torn_txt = self.pantorn.text(0.1,0.5,"BLUE PARTICLE'S TURN!",color = self.torn)
            self.pantorn_canvas.draw()
            
        elif self.torn == 'blue':
            self.torn = 'red'
            
            self.torn_txt.remove()
            self.pantorn.patch.set_facecolor('black')
            self.torn_txt = self.pantorn.text(0.1,0.5,"RED PARTICLE'S TURN!",color = self.torn)
            self.pantorn_canvas.draw()
        
        if self.control_botons == True:
            self.control_botons = False
    
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
    
    def potencial(self):
        
        #self.pot = 42.55*(np.exp(-(self.xx/self.bwidth)**2))/np.sqrt(self.bheight_gauss*np.pi)
        self.pot = np.zeros(len(self.xx))
        for i in range(0,len(self.xx)):
            if -self.amplada/2.0 < self.xx[i] < self.amplada/2.0:
                self.pot[i] = 76.0
    
    def ona(self):
        
        self.phi0 = 1j*np.zeros(len(self.xx))
        self.phi0 = (1.0/(2.0*np.pi)**(1.0/4.0))*np.exp((-(self.xx - 7.0)**2)/4.0)*np.exp(1j*self.xx*self.p0)
        self.energia()
        self.T_analitic()
        
    def T_analitic(self):
        
        self.arrel = []
        
        for i in range(0,self.nx+1):
            if self.pot[i] > self.E:
                self.arrel = np.append(self.arrel,np.sqrt(2.0*(self.pot[i]-self.E)))
        
        
        integral = self.simpson2(self.dx,self.arrel)
        #integral = (self.bwidth*np.sqrt(76.0-self.E))
        
        self.T = np.abs((np.exp((-2.0)*integral)))
    
                
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
            
    def actualitzacio(self,instance):
        
        #En l'actualitzacio també hi ha d'haver nous valors per a T
        for i in range(0,4):
            self.botons_amp[i] = random.choice(self.list_amp)
            self.amplada = self.botons_amp[i]
            self.potencial()
            self.ona()
            self.botons_T[i] = self.T
        
        #for i in range(0,self.nbonus):
            #if self.posx_red == self.matriu_bonus[i,0] and self.posy_red == self.matriu_bonus[i,1]:
                #GlobalShared.bonus = self.matriu_bonus[i,2]
                
        #for i in range(0,self.nbonus):
            #if self.posx_blue == self.matriu_bonus[i,0] and self.posy_blue == self.matriu_bonus[i,1]:
                #GlobalShared.bonus = self.matriu_bonus[i,2]
        
        #Ara ho ha de dibuixar en les pantalles assignades
        self.up_txt.remove()
        self.up_txt = self.pantup.text(0.1,0.1,'%.6f' % (self.botons_T[1]),color='white')
        self.pantup_canvas.draw()
        
        self.down_txt.remove()
        self.down_txt = self.pantdown.text(0.1,0.1,'%.6f' % (self.botons_T[0]),color='white')
        self.pantdown_canvas.draw()
        
        self.right_txt.remove()
        self.right_txt = self.pantright.text(0.1,0.1,'%.6f' % (self.botons_T[2]),color='white')
        self.pantright_canvas.draw()
        
        self.left_txt.remove()
        self.left_txt = self.pantleft.text(0.1,0.1,'%.6f' % (self.botons_T[3]),color='white')
        self.pantleft_canvas.draw()

        if self.torn == 'red':
            self.prob_txt.remove()
            self.red_prob = self.red_prob*GlobalShared.prob*GlobalShared.bonus
            self.prob_txt = self.pantprob_red.text(0.1,0.5,str(self.red_prob))
        
            self.pantprob_red_canvas.draw()
            
            self.angle_end_red = self.angle_end_red - 50*(1 - GlobalShared.prob)
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.canvi_torn()
            
        elif self.torn == 'blue':
            self.probb_txt.remove()
            self.blue_prob = self.blue_prob*GlobalShared.prob*GlobalShared.bonus
            self.probb_txt = self.pantprob_blue.text(0.1,0.5,str(self.blue_prob))
        
            self.pantprob_blue_canvas.draw()
            
            self.angle_end_blue = self.angle_end_blue - 50*(1 - GlobalShared.prob)
            self.box1.canvas.clear()
            self.tauler()
            self.draw_redcircle()
            self.draw_bluecircle()
            self.write_bonus()
            self.draw_endpoint()
            
            self.canvi_torn()
        #GlobalShared.bonus = 1.0
        
        
        
        
        
        

        
    def openpop(self):
        
        amplada = self.amplada
        torn = self.torn
        popup = TunnelPopup(amplada,torn)
        popup.open() 
        
        popup.bind(on_dismiss = self.actualitzacio)      
        
        
        
        
        
class TunnelPopup(Popup):
    
    amplada = NumericProperty()
    torn = StringProperty()
    
    
    def __init__(self,amplada,torn):

        super(Popup, self).__init__()  
        
        self.bwidth = amplada
        self.color = torn
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
        self.bheight_quadrat = 76.0
        self.bheight_gauss = 0.1
        self.potencial()
        self.control_pot = 'quadrat'
        self.max_pot = 90.0
        self.min_pot = 0.0
        self.phi0()
        self.phi02 = abs(self.phi0)**2
        
        self.colpot = 'yellow'
        self.cole = 'green'
        self.colphi = 'red'
  
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('white')
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        self.main_canvas.bind(on_touch_up = self.control_teclat)
        self.box1.add_widget(self.main_canvas)
            
        self.pot_graf = plt.subplot()
        self.pot_graf.set_facecolor('black')
        self.pot_graf.axis([self.lx_min, self.lx_max, self.min_pot, self.max_pot])
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color = self.colpot)
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.color)
        
        self.ona = self.pot_graf.twinx()
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.color)
        self.ona.axis([self.lx_min,self.lx_max,0.0,1.0])
        
        #"Pantalla" on es veurè el coeficient de transmissió analític
        self.pant1 = plt.figure()
        self.pant1.patch.set_facecolor('white')
        self.pant1_canvas = FigureCanvasKivyAgg(self.pant1)
        self.box2.add_widget(self.pant1_canvas)
        
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        
        #"Pantalla" on es veurà la probabilotat de l'ona d'estar a l'esquerra de la barrera
        self.pant2 = plt.figure()
        self.pant2.patch.set_facecolor('white')
        self.pant2_canvas = FigureCanvasKivyAgg(self.pant2)
        self.box3.add_widget(self.pant2_canvas)
        
        self.prob_left_txt = self.pant2.text(0.1,0.5,'0.00000000')
        
        self.pant3 = plt.figure()
        self.pant3.patch.set_facecolor('white')
        self.pant3_canvas = FigureCanvasKivyAgg(self.pant3)
        self.box4.add_widget(self.pant3_canvas)
        
        self.norma_txt = self.pant3.text(0.1,0.5,'1.00000000')
        
                                    
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
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        
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
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        
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
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()

        
    def subtract_width(self):
        
        self.pot_data.remove()
        self.T_txt.remove()
        self.bwidth = self.bwidth - 0.05
        self.potencial()
        self.T_analitic()
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()
        
    def add_bheight(self):
        
        self.pot_data.remove()
        self.T_txt.remove()
        self.bheight_quadrat = self.bheight_quadrat + 0.5
        self.potencial()
        self.T_analitic()
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()
        
    def subtract_bheight(self):
        
        self.pot_data.remove()
        self.T_txt.remove()
        self.bheight_quadrat = self.bheight_quadrat - 0.5
        self.potencial()
        self.T_analitic()
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        
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
        GlobalShared.prob = self.probleft
        print (GlobalShared.prob)
    
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
        self.visu_ona, = self.ona.fill(self.xx,self.phi2,color = self.color)
        
        
        self.main_canvas.draw()
        
        if np.mod(self.t,5.0) == 0.0:
            #Cridem la funció que integra la funcio d'ona a l'esquerra
            self.prob_left()
        
            self.prob_left_txt.remove()
            self.prob_left_txt = self.pant2.text(0.1,0.5,'%.8f' % (self.probleft))
        
            #Cridem la funció que ens calcula la norma
            self.calc_norma()
        
            self.norma_txt.remove()
            self.norma_txt = self.pant3.text(0.1,0.5,'%.8f' % (self.norma))
        
            self.pant2_canvas.draw()
            self.pant3_canvas.draw()
            
        if self.t == 150:
            
            self.stop_evolucio()
        
        
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
        self.prob_left_txt = self.pant2.text(0.1,0.5,'0.00000000')
        self.norma_txt = self.pant3.text(0.1,0.5,'1.00000000')
        
        self.pant2.canvas.draw()
        self.pant3.canvas.draw()
        self.main_canvas.draw()
    
    def _teclat_activat(self,keyboard, keycode, text, modifiers):
        if keycode[1] == 'spacebar':
            self.inici_evolucio()
            self._teclat.unbind(on_hey_down = self._teclat_activat)
        return
    
    def control_teclat(self, *args):
        
        
        self._teclat = Window.request_keyboard(self._teclat_tancat,self)
        self._teclat.bind(on_key_down=self._teclat_activat) 
                
      
    def _teclat_tancat(self):
        
        self._teclat.unbind(on_key_down=self._teclat_activat) 

        
TunnelingApp().run()
    













    
    

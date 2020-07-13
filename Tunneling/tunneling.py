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
            
        #self.bonus = [1.2,1.5,2.0,5.0,10.0]
        #self.posx_rand = [25,75,125,175,225,275,325,375,425,475]
        #self.posy_rand = [25,75,125,175,225,275,325,375,425,475]
        #self.nbonus = 4
        #self.matriu_bonus = np.zeros((self.nbonus,3))
        
        #for i in range(0,self.nbonus):
            
            #self.matriu_bonus[i,0] = random.choice(self.posx_rand)
            #self.matriu_bonus[i,1] = random.choice(self.posy_rand)
            #self.matriu_bonus[i,2] = random.choice(self.bonus)
            
        #definim on comença i on acaba el tauler
        self.inrect = 50.0
        self.finrect = 410.0
        
        self.amplada = 0.1
        
        #definim la maxima posicio de les partícules en x i en y
        self.maxpos = self.finrect + 20.0
        self.minpos = self.inrect + 20.0
        #definim posició inicial de les partícules
        self.posx_red = self.minpos
        self.posy_red = self.minpos
        self.posx_blue = self.maxpos
        self.posy_blue = self.minpos
        
        self.salt = 90.0 # salt cada cop que movem la partícula
        self.angle_start_red = 0
        self.angle_end_red = 360
        self.angle_start_blue = 0
        self.angle_end_blue = 360
        
        self.red_prob = 1.0
        self.blue_prob = 1.0
        
        self.torn = 'red'
        #control_botons controla que només es pugui prémer un dels botons
        self.control_botons = False
    
        
        self.interval = 90.0
        self.marge = 5
        self.dt_barr = 0.5
        self.bar_pos_x = np.arange(self.inrect,self.finrect + 2.0*self.interval,self.interval)
        self.bar_pos_y = np.arange(self.inrect,self.finrect + 2.0*self.interval,self.interval)
        
        #també necessitem tenir la posició actualitzada de cada partícula
        self.actualposx_blue = 9
        self.actualposy_blue = 9
        self.actualposx_red = 0
        self.actualposy_red = 9
        
        self.draw_redcircle()
        self.draw_bluecircle()
        self.tauler()
        self.pantalles()
        #Sself.write_bonus()
        self.rectangle()
        
        #que comenci el canvi d'amplada de les barreres automàticament
        self.canvi_barreres()
        
        
    def pantalles(self):
        
        
        #self.pantup = plt.figure()
        #self.pantup.patch.set_facecolor('black')
        #self.pantup_canvas =  FigureCanvasKivyAgg(self.pantup)
        #self.box3.add_widget(self.pantup_canvas)
        
        #self.up_txt = self.pantup.text(0.1,0.1,'%.6f' % (self.botons_T[1]),color='white')
        
        #self.pantdown = plt.figure()
        #self.pantdown.patch.set_facecolor('black')
        #self.pantdown_canvas =  FigureCanvasKivyAgg(self.pantdown)
        #self.box6.add_widget(self.pantdown_canvas)
        
        #self.down_txt = self.pantdown.text(0.1,0.1,'%.6f' % (self.botons_T[0]),color='white')
        
        #self.pantright = plt.figure()
        #self.pantright.patch.set_facecolor('black')
        #self.pantright_canvas =  FigureCanvasKivyAgg(self.pantright)
        #self.box7.add_widget(self.pantright_canvas)
        
        #self.right_txt = self.pantright.text(0.1,0.1,'%.6f' % (self.botons_T[2]),color='white')
        
        #self.pantleft = plt.figure()
        #self.pantleft.patch.set_facecolor('black')
        #self.pantleft_canvas =  FigureCanvasKivyAgg(self.pantleft)
        #self.box8.add_widget(self.pantleft_canvas)
        
        #self.left_txt = self.pantleft.text(0.1,0.1, '%.6f' % (self.botons_T[3]),color='white')
        
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
            Ellipse(pos=(self.posx_red,self.posy_red),size = (50,50),
                    angle_start = self.angle_start_red, angle_end = self.angle_end_red)
                 
        
        self.red_particle = FloatLayout()
        self.box1.add_widget(self.red_particle)
        
        return self.red_particle
        
        
    def draw_bluecircle(self):
    
        with self.box1.canvas:
            Color(0,0,1,mode = 'rgb')
            self.bluecircle = Ellipse(pos=(self.posx_blue,self.posy_blue),size=(50,50),
                                      angle_start = self.angle_start_blue, angle_end = self.angle_end_blue)
            
        self.blue_particle = FloatLayout()
        self.box1.add_widget(self.blue_particle)
        
        
            
        return self.blue_particle
    
    
    #def write_bonus(self):
        #self.write_bonus0()
        #self.write_bonus1()
        #self.write_bonus2()
        #self.write_bonus3()
        
    
    #def write_bonus0(self):
        
        #with self.box1.canvas:
            #self.bonus0 = Label(text= 'x' + (str(self.matriu_bonus[0,2])), 
                                  #pos=(int(self.matriu_bonus[0,0]),int(self.matriu_bonus[0,1])))
        
        #self.bonus0 = FloatLayout()
        #self.box1.add_widget(self.bonus0)
        
        #return self.bonus0


    def tauler(self):
        
        self.index_barr = 0
        
        for i in range(1,len(self.bar_pos_x)-1):
            for j in range(0,len(self.bar_pos_y)-1):
                x = self.bar_pos_x[i]
                y_1 = self.bar_pos_y[j] + self.marge
                y_2 = self.bar_pos_y[j+1] - self.marge
                amp_bar = GlobalShared.barr_vert[j,i-1,0]
                
                with self.box1.canvas:
                    Color(0,1,0,mode='rgb')
                    Line(bezier=(x,y_1,x,y_2),width = amp_bar*5.0)
                    
        for i in range(1,len(self.bar_pos_y)-1):
            for j in range(0,len(self.bar_pos_x)-1):
                y = self.bar_pos_y[i]
                x_1 = self.bar_pos_x[j] + self.marge
                x_2 = self.bar_pos_x[j+1] - self.marge
                amp_bar = GlobalShared.barr_hor[j-1,i,0]
                
                with self.box1.canvas:
                    Color(0,1,0,mode='rgb')
                    Line(bezier=(x_1,y,x_2,y),width = amp_bar*5.0)
                    
    
    def rectangle(self):
        
        with self.box1.canvas:
            Color(0,1,0,mode='rgb')
            Line(rectangle = (self.inrect,self.inrect,self.finrect + self.inrect,self.finrect + self.inrect),width = 3.0)
        
        self.limits = FloatLayout()
        self.box1.add_widget(self.limits)
        
        return self.limits
    
    
    
    def canvi_barreres(self):
        
        self.ev_barr = Clock.schedule_interval(self.evolucio_barr,self.dt_barr)
        
    def evolucio_barr(self,dt):
        
        self.index_barr = self.index_barr + 1
        
        if self.index_barr == 3:
            self.index_barr = 0
            
        self.box1.canvas.clear()
        
        for i in range(1,len(self.bar_pos_x)-1):
            for j in range(0,len(self.bar_pos_y)-1):
                x = self.bar_pos_x[i]
                y_1 = self.bar_pos_y[j] + self.marge
                y_2 = self.bar_pos_y[j+1] - self.marge
                amp_bar = GlobalShared.barr_vert[j,i-1,self.index_barr]
                
                with self.box1.canvas:
                    Color(0,1,0,mode='rgb')
                    Line(bezier=(x,y_1,x,y_2),width = amp_bar*5.0)
                    
        for i in range(1,len(self.bar_pos_y)-1):
            for j in range(0,len(self.bar_pos_x)-1):
                y = self.bar_pos_y[i]
                x_1 = self.bar_pos_x[j] + self.marge
                x_2 = self.bar_pos_x[j+1] - self.marge
                amp_bar = GlobalShared.barr_hor[j-1,i,self.index_barr]
                
                with self.box1.canvas:
                    Color(0,1,0,mode='rgb')
                    Line(bezier=(x_1,y,x_2,y),width = amp_bar*5.0)  
                    
        self.rectangle()
        self.draw_bluecircle()
        self.draw_redcircle()
        
    def stop_evolucio_barr(self):
        
        self.ev_barr.cancel()
        
    
    def down_red(self):
        
        if self.torn == 'red' and self.control_botons == False:
            
            if self.posy_red == self.minpos:
                self.posy_red = self.posy_red
        
            else:
                self.stop_evolucio_barr() #parem primer el canvi d'amplada de les barreres
                self.posy_red = self.posy_red - self.salt
                self.actualposy_red = self.actualposy_red + 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_red,self.actualposx_red,self.index_barr]
                
                self.post_botons_posicio()



    def down_blue(self):
        
        if self.torn == 'blue' and self.control_botons == False:
            self.amplada = self.botons_amp[0]
        
            if self.posy_blue == self.minpos:
                self.posy_blue = self.posy_blue
        
            else:
                self.stop_evolucio_barr()
                self.posy_blue = self.posy_blue - self.salt
                self.actualposy_blue = self.actualposy_blue + 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_blue,self.actualposx_blue,self.index_barr]
                
                self.post_botons_posicio()

            
    def up_blue(self):
        
        if self.torn == 'blue' and self.control_botons == False:
        
            if self.posy_blue == self.maxpos:
                self.posy_blue = self.posy_blue
        
            else:
                self.stop_evolucio_barr()
                self.posy_blue = self.posy_blue + self.salt
                self.actualposy_blue = self.actualposy_blue - 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_blue - 1,self.actualposx_blue,self.index_barr]
                
                self.post_botons_posicio()

        
        
    def up_red(self):
        
        if self.torn == 'red' and self.control_botons == False:
            
            
        
            if self.posy_red == self.maxpos:
                self.posy_red = self.posy_red
        
            else:
                self.stop_evolucio_barr()
                self.posy_red = self.posy_red + self.salt
                self.actualposy_red = self.actualposy_red - 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_red-1,self.actualposx_red,self.index_barr]
                
                self.post_botons_posicio()

            
    
    def right_red(self):
        
        if self.torn == 'red' and self.control_botons == False:
        
            if self.posx_red == self.maxpos:
                self.posx_red = self.posx_red
        
            else:
                self.stop_evolucio_barr()
                self.posx_red = self.posx_red + self.salt
                self.actualposx_red = self.actualposx_red + 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_red,self.actualposx_red,self.index_barr]
                
            
                self.post_botons_posicio()

                
    def right_blue(self):
        
        if self.torn == 'blue' and self.control_botons == False:
        
            if self.posx_blue == self.maxpos:
                self.posx_blue = self.posx_blue
        
            else:
                self.stop_evolucio_barr()
                self.posx_blue = self.posx_blue + self.salt
                self.actualposy_blue = self.actualposy_blue + 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_blue,self.actualposx_blue,self.index_barr]
                
    
                self.post_botons_posicio()
            
    def left_red(self):
        
        if self.torn == 'red' and self.control_botons == False:
        
            if self.posx_red == self.minpos:
                self.posx_red = self.posx_red
        
            else:
                self.stop_evolucio_barr()
                self.posx_red = self.posx_red - self.salt
                self.actualposx_red = self.actualposx_red - 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_red,self.actualposx_red-1,self.index_barr]
            
                self.post_botons_posicio()


            
    def left_blue(self):
        
        if self.torn =='blue' and self.control_botons == False:
        
            if self.posx_blue == self.minpos:
                self.posx_blue = self.posx_blue
        
            else:
                self.stop_evolucio_barr()
                self.posx_blue = self.posx_blue - self.salt
                self.actualposx_blue = self.actualposx_blue - 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_blue,self.actualposx_blue-1,self.index_barr]
                
                self.post_botons_posicio()
                
                
    def post_botons_posicio(self):
        
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        #self.write_bonus()
        self.rectangle()
            
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
        
        
        #for i in range(0,self.nbonus):
            #if self.posx_red == self.matriu_bonus[i,0] and self.posy_red == self.matriu_bonus[i,1]:
                #GlobalShared.bonus = self.matriu_bonus[i,2]
                
        #for i in range(0,self.nbonus):
            #if self.posx_blue == self.matriu_bonus[i,0] and self.posy_blue == self.matriu_bonus[i,1]:
                #GlobalShared.bonus = self.matriu_bonus[i,2]
        
        #Ara ho ha de dibuixar enles  pantalles assignades
        #self.up_txt.remove()
        #self.up_txt = self.pantup.text(0.1,0.1,'%.6f' % (self.botons_T[1]),color='white')
        #self.pantup_canvas.draw()
        
        #self.down_txt.remove()
        #self.down_txt = self.pantdown.text(0.1,0.1,'%.6f' % (self.botons_T[0]),color='white')
        #self.pantdown_canvas.draw()
        
        #self.right_txt.remove()
        #self.right_txt = self.pantright.text(0.1,0.1,'%.6f' % (self.botons_T[2]),color='white')
        #self.pantright_canvas.draw()
        
        #self.left_txt.remove()
        #self.left_txt = self.pantleft.text(0.1,0.1,'%.6f' % (self.botons_T[3]),color='white')
        #self.pantleft_canvas.draw()

        if self.torn == 'red':
            self.prob_txt.remove()
            self.red_prob = self.red_prob*GlobalShared.prob*GlobalShared.bonus
            self.prob_txt = self.pantprob_red.text(0.1,0.5,'%.3f' % (self.red_prob))
        
            self.pantprob_red_canvas.draw()
            
            self.angle_end_red = self.angle_end_red - 50*(1 - GlobalShared.prob)
           
            
        elif self.torn == 'blue':
            self.probb_txt.remove()
            self.blue_prob = self.blue_prob*GlobalShared.prob*GlobalShared.bonus
            self.probb_txt = self.pantprob_blue.text(0.1,0.5,'%.3f' % (self.blue_prob))
        
            self.pantprob_blue_canvas.draw()
            
            self.angle_end_blue = self.angle_end_blue - 50*(1 - GlobalShared.prob)
           
        self.box1.canvas.clear()
        self.tauler()
        self.rectangle()
        self.draw_redcircle()
        self.draw_bluecircle()
        #self.write_bonus()
        self.canvi_barreres()
            
        self.canvi_torn()
        
        
        
        
        
        

        
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
        self.control_pot = 'quadrat'
        self.potencial()
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
        
        self.pot = np.zeros(len(self.xx))
        
        if self.control_pot == 'quadrat':
            for i in range(0,len(self.xx)):
                if -self.bwidth/2.0 < self.xx[i] < self.bwidth/2.0:
                    self.pot[i] = self.bheight_quadrat
        elif self.control_pot == 'gauss':
            self.pot = 42.55*(np.exp(-(self.xx/self.bwidth)**2))/np.sqrt(self.bheight_gauss*np.pi)
                
    def canvi_potencial_gauss(self):
        
        self.pot = 42.55*(np.exp(-(self.xx/self.bwidth)**2))/np.sqrt(self.bheight_gauss*np.pi)
        
        self.pot_data.remove()
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        
        self.T_analitic()
        self.T_txt.remove()
        self.T_txt = self.pant1.text(0.1, 0.5,str(self.T))
        
        self.main_canvas.draw()
        self.pant1_canvas.draw()
        
        self.control_pot = 'gauss'
        
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
        
        self.control_pot = 'quadrat'
        
            

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
        
        if self.control_pot == 'quadrat': 
            self.bwidth = self.bwidth + 0.05 
        elif self.control_pot == 'gauss':
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
        
        if self.control_pot == 'quadrat':
            self.bwidth = self.bwidth - 0.05
        if self.control_pot == 'gauss':
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
        
        if self.control_pot == 'quadrat':
            self.bheight_quadrat = self.bheight_quadrat + 0.5
        elif self.control_pot == 'gauss':
            self.bheight_gauss = self.bheight_gauss - 0.01
            
        self.potencial()
        self.T_analitic()
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        
        self.pant1_canvas.draw()
        self.main_canvas.draw()
        
    def subtract_bheight(self):
        
        
        
        if self.control_pot == 'quadrat':
            if self.bheight_quadrat > 66.0:
                self.bheight_quadrat = self.bheight_quadrat - 0.5
                
                self.pot_data.remove()
                self.T_txt.remove()
        
                self.potencial()
                self.T_analitic()
                self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
                self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
                
                self.pant1_canvas.draw()
                self.main_canvas.draw()
                
            else:
                self.bheight_quadrat = self.bheight_quadrat
                
        elif self.control_pot == 'gauss':
            if self.bheight_gauss < 0.13:
                self.bheight_gauss = self.bheight_gauss + 0.01
                
                self.pot_data.remove()
                self.T_txt.remove()
                
                self.potencial()
                self.T_analitic()
                self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
                self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
                
                self.pant1_canvas.draw()
                self.main_canvas.draw()
            else:
                self.bheight_gauss = self.bheight_gauss

            
        

        
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
    













    
    

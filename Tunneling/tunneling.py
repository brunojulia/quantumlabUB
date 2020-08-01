# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 20:29:04 2020

@author: usuari
"""
import kivy
import numpy as np
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.uix.screenmanager import FadeTransition, SlideTransition
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
from kivy.uix.label import Label 
from kivy.uix.slider import Slider
from kivy.uix.button import Button
from kivy.uix.image import Image

import random






class TunnelingApp(App):
    
    def build(self):
        
        self.title = 'Tunneling'
        return MyScreenManager() 
    
class MyScreenManager(ScreenManager):
    
    def __init__(self,**kwargs):
        
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('gaming').gpseudo_init()
        self.get_screen('Simulation').spseudo_init()
        
class StartingScreen(Screen):
    
    def __init__(self,**kwargs):
        
        super(StartingScreen,self).__init__(**kwargs)
        #Window.fullscreen = 'auto'
        
    """ Les següents funcoins són cridades pels botons de la pantalla d'inici
    i canvien de pantalla segons el botó premut """
        
    def transition_SG(self):
        
        self.manager.current = 'gaming'
        self.manager.transition = FadeTransition()
        
    def transition_SS(self):
        
        self.manager.current = 'Simulation'
        self.manager.transition = FadeTransition()
    
class GamingScreen(Screen):
    
       
    def  __init__(self,**kwargs):
    
        super(GamingScreen, self).__init__(**kwargs)
        Window.fullscreen = 'auto'
        
    
        
    def transition_GS(self):
        
        """ Aquesta funció és cridada quan el botó "back" és premut per poder 
        tornar a la patnalla inicial """
        
        self.manager.current = 'starting'
        self.manager.transition = FadeTransition()
    
    def gpseudo_init(self):

        
        self.lx_max = 17.0
        self.lx_min = -17.0
        self.nx = 500
        self.dx = (self.lx_max - self.lx_min)/float(self.nx)
        self.xx = np.arange(self.lx_min,self.lx_max + self.dx,self.dx)
        self.p0 = -200.0/self.lx_max
            
        #definim on comença i on acaba el tauler
        self.inrect = 50.0
        self.finrect = 410.0*1.5
        
        self.amplada = 0.1
        
        self.angle_start_red = 0
        self.angle_end_red = 360
        self.angle_start_blue = 0
        self.angle_end_blue = 360
        
        self.red_prob = 1.0
        self.blue_prob = 1.0
        
        self.torn = 'red'
        #control_botons controla que només es pugui prémer un dels botons
        self.control_botons = False
    
        
        self.interval = 90.0*1.5
        self.marge = 5
        self.dt_barr = 0.5
        self.bar_pos_x = np.arange(self.inrect,self.finrect + 1.0*self.interval,self.interval)
        self.bar_pos_y = np.arange(self.inrect,self.finrect + 1.0*self.interval,self.interval)
        
        #també necessitem tenir la posició actualitzada de cada partícula
        self.actualposx_blue = 4
        self.actualposy_blue = 4
        self.actualposx_red = 0
        self.actualposy_red = 4
        
        #definim la maxima posicio de les partícules en x i en y
        self.maxpos = self.inrect + 45.0 + 4*(self.interval)
        self.minpos = self.inrect + 45.0 
        #definim posició inicial de les partícules
        self.posx_red = self.minpos
        self.posy_red = self.minpos
        self.posx_blue = self.maxpos
        self.posy_blue = self.minpos
        self.salt = self.interval # salt cada cop que movem la partícula
        
        #definim variables necessàries per a controlar el final del joc
        self.end_red = False
        self.end_blue = False
        
        self.imatge_cursor = 'cursor_amplada.png'
        
        """ Definim tots els paràmetres per calcular l'evolucio de la ona
        al travessar la barrera"""
        
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
        self.color = self.torn
        
        self.colpot = 'yellow'
        self.cole = 'green'
        self.colphi = 'red'
        
        self.control_evolucio = False
        
        """ Un cop tot definit cridem les funcions que ho dibuixen """
        
        self.draw_redcircle()
        self.draw_bluecircle()
        self.tauler()
        self.imatge_tauler()
        self.rectangle()
        self.pantalles()
        
        #que comenci el canvi d'amplada de les barreres automàticament
        self.canvi_barreres()
        
        
        self.create_first_slider()
        
        #definim el mode de joc
        self.control_mode = 'wkb'
        self.box3.size_hint_y = 0.1
        self.label_mode = Label(text = 'WKB')
        self.box10.add_widget(self.label_mode)
        self.box8.size_hint_y = 0.5
        self.box9.size_hint_y = 0.5
        self.botons_mode()

        
  
    def pantalles(self):
        
        """ Dibuixa les pantalles amb les probabilitats """
       
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
        
        #self.pantorn = plt.figure()
        #self.pantorn.patch.set_facecolor('black')
        #self.pantorn_canvas = FigureCanvasKivyAgg(self.pantorn)
        #self.box5.add_widget(self.pantorn_canvas)
        
        #self.torn_txt = self.pantorn.text(0.1,0.5,"RED PARTICLE'S TURN!",color=self.torn)
       
        """ Dibuixa la pantalla amb el valor del WKB """
        self.pant_wkb = plt.figure()
        self.pant_wkb.patch.set_facecolor('white')
        self.pant_wkb_canvas = FigureCanvasKivyAgg(self.pant_wkb)
        self.box5.add_widget(self.pant_wkb_canvas)
        
        self.wkb_txt = self.pant_wkb.text(0.1,0.5,'0.00000000')
        
        
    def pantalla_evolucio(self):
        
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('white')
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        #self.main_canvas.bind(on_touch_up = self.control_teclat)
        self.box3.add_widget(self.main_canvas)
            
        self.pot_graf = plt.subplot()
        self.pot_graf.set_facecolor('black')
        self.pot_graf.axis([self.lx_min, self.lx_max, self.min_pot, self.max_pot])
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color = self.colpot)
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.torn)
        
        self.ona = self.pot_graf.twinx()
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.torn)
        self.ona.axis([self.lx_min,self.lx_max,0.0,1.0])
        
        self.main_fig.tight_layout() 
        self.main_canvas.draw()
        
        
        
    """_______ CONTROL DEL MODE DEL JOC ____________________________________"""
    
    def botons_mode(self):
        
        #creem els botons per triar el mode
        self.boto_wkb = Button(text = 'WKB mode')
        self.box8.add_widget(self.boto_wkb)
        self.boto_wkb.bind(on_press = self.wkb_mode)
        self.boto_wkb.background_color = (0,0,0,0)
        
        self.boto_schrodinger = Button(text = 'Schrodinger mode')
        self.box9.add_widget(self.boto_schrodinger)
        self.boto_schrodinger.bind(on_press = self.schrodinger_mode)
        
        self.boto_nexturn = Button(text= "Next turn")
        self.box11.add_widget(self.boto_nexturn)
        self.boto_nexturn.bind(on_press = self.boto_nexturn_premut)
        
    def boto_nexturn_premut(self,instance):
        
        if self.control_mode == 'wkb':
        
            self.actualitzacio()
        
        
    def wkb_mode(self,instance):
        
        if self.control_mode == 'schrodinger':
    
            #fem que canvii de color el botó WKB
            self.boto_wkb.background_color = (0,0,0,0)
            self.boto_schrodinger.background_color = (1,1,1,1)
        
            self.box10.remove_widget(self.label_mode)
            self.label_mode = Label(text = 'WKB')
            self.box10.add_widget(self.label_mode)
            
            self.box8.size_hint_y = 0.5
            self.box9.size_hint_y = 0.5
            
            self.box11.size_hint_y = 1
            self.box11.remove_widget(self.boto_nexturn)
            self.boto_nexturn = Button(text= "Next turn")
            self.box11.add_widget(self.boto_nexturn)
            self.boto_nexturn.bind(on_press = self.boto_nexturn_premut)
        

            self.box3.remove_widget(self.main_canvas)
            self.box3.size_hint_y = 0.1
            
            self.control_mode = 'wkb'
        
            return self.label_mode

        
    def schrodinger_mode(self,instance):
        
        if self.control_mode == 'wkb':
        
            self.control_mode = 'schrodinger'
        
            #fem que canvii el color del boto Schrodinger
            self.boto_schrodinger.background_color = (0,0,0,0)
            self.boto_wkb.background_color = (1,1,1,1)
        
            self.box10.remove_widget(self.label_mode)
            self.label_mode = Label(text = 'Probability left')
            self.box10.add_widget(self.label_mode)
            
            self.box8.size_hint_y = 1
            self.box9.size_hint_y = 1
            
            self.box11.remove_widget(self.boto_nexturn)
            self.box11.size_hint_y = 0.1

        
            self.box3.size_hint_y = 5
            self.pantalla_evolucio()
        
            return self.label_mode
        
        
        
            
        
            
        
            

    """________FUNCIONS QUE DIBUIXEN ELS ELEMENTS DEL TAULER________________ """
    
    """Aquestes funcions dibuixen les barreres del tauler cada cop que les funcions 
    que controlen l'evolució temporal de les barreres les criden. També dibuixen les 
    partícules en la seva posicio corresponent i el rectangle extern del tauler. Són
    cridades totes en conjunt en la funcio actualitzacio_tauler """


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
    
    


    def tauler(self):
        
        """ Dibuixa el tauler cada cop que és cridada per les funcions de la 
        evolucio temporal del tauler self.evolucio_barr """
        
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
        
        #self.imatge_tauler()
                    
    
    def rectangle(self):
        
        with self.box1.canvas:
            if self.torn == 'red':
                Color(1,0,0,mode='rgb')
            elif self.torn == 'blue':
                Color(0,0,1,mode='rgb')
            Line(rectangle = (self.inrect,self.inrect,self.finrect + self.inrect + 10.0,self.finrect + self.inrect + 10.0),width = 3.0)
        
        self.limits = FloatLayout()
        self.box1.add_widget(self.limits)
        
        return self.limits
    
    def imatge_tauler(self):
        
        for i in range(0,5):
            for j in range(0,5):
                
                #primer passem del númemro assignat al tipus de casella
                if GlobalShared.caselles_tauler[i,j] == 1:
                    
                    imatge1 = Image(source = 'cursor_alçada.png')
                    imatge1.allow_stretch = True
                    imatge1.keep_ratio = False
                    imatge1.size_hint_x = 0.1
                    imatge1.size_hint_y = 0.1
                    imatge1.pos = ((self.inrect  + 22.5 + i*self.interval),(self.inrect + 22.5 + j*self.interval))
                    self.box1.add_widget(imatge1)

                
                if GlobalShared.caselles_tauler[i,j] == 2:
                    
                    imatge2 = Image(source = 'cursor_amplada.png')
                    imatge2.allow_stretch = True
                    imatge2.keep_ratio = False
                    imatge2.size_hint_x = 0.1
                    imatge2.size_hint_y = 0.1
                    imatge2.pos = ((self.inrect  + 22.5 + i*self.interval),(self.inrect + 22.5 + j*self.interval))
                    self.box1.add_widget(imatge2)
                    
                    
        

                    
                    


        

    
    
    
    
    
    """ CONTROL DE L'EVOLUCIÓ TEMPORAL DEL TAULER """

    """ La funció que inicia l'evolució de les barreres és cridada automàticament
    quan s'obre la pantalla de joc. També és cridada per la funció de l'actualització
    del tauler (que és cridada just despés de moure una partícula) """    
    
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
        #self.imatge_tauler()
        self.draw_bluecircle()
        self.draw_redcircle()
        
        
        
    def stop_evolucio_barr(self):
        
        self.ev_barr.cancel()
        
    
    """ BOTONS PER MOURE LES PARTÍCULES """ 
    
    """Comproven de qui es el torn i mouen la partícula corresponent.
    Paren levolució temporal de les barreres del tauler i agafen lamplada
    de la barrera del moment del pas. Després es criden les funcions 
    dactualització sactualiza la posicio de la partícula, es calcula el WKB
    en funció de lamplada registrada i es dibuixa la barrera de potencial 
    amb lamplada corresponent. També es crida la funció que controla 
    la variació de paràmetres, i aquesta crea l'slider corresponent etc"""
        
    
    def down(self):
        
        if self.torn == 'red' and self.control_botons == False:
            
            if self.posy_red == self.minpos:
                self.posy_red = self.posy_red
        
            else:
                self.stop_evolucio_barr() #parem primer el canvi d'amplada de les barreres
                self.posy_red = self.posy_red - self.salt
                self.actualposy_red = self.actualposy_red + 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_red - 1,self.actualposx_red,self.index_barr]
                print(self.amplada)
                
        elif self.torn == 'blue' and self.control_botons == False:
            self.amplada = self.botons_amp[0]
        
            if self.posy_blue == self.minpos:
                self.posy_blue = self.posy_blue
        
            else:
                self.stop_evolucio_barr()
                self.posy_blue = self.posy_blue - self.salt
                self.actualposy_blue = self.actualposy_blue + 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_blue - 1,self.actualposx_blue,self.index_barr]
                print(self.amplada)
            
        self.control_botons = True            
                
        self.actualitzacio_tauler()
        self.actualitzacio_wkb()
    
        self.var_parametres()
        
        if self.control_mode == 'schrodinger':
            self.actualitzacio_pantallaev()
            self.inici_evolucio()


                
                
   

            
    def up(self):
        
        if self.torn == 'blue' and self.control_botons == False:
        
            if self.posy_blue == self.maxpos:
                self.posy_blue = self.posy_blue
        
            else:
                self.stop_evolucio_barr()
                self.posy_blue = self.posy_blue + self.salt
                self.actualposy_blue = self.actualposy_blue - 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_blue,self.actualposx_blue,self.index_barr]
                print(self.amplada)

            
        elif self.torn == 'red' and self.control_botons == False:    
        
            if self.posy_red == self.maxpos:
                self.posy_red = self.posy_red
        
            else:
                self.stop_evolucio_barr()
                print(self.index_barr)
                self.posy_red = self.posy_red + self.salt
                self.actualposy_red = self.actualposy_red - 1
                self.amplada = GlobalShared.barr_hor[self.actualposy_red,self.actualposx_red,self.index_barr]
                print(self.amplada)
                print(GlobalShared.barr_hor[self.actualposy_red,self.actualposx_red,:])
                
        self.control_botons = True
                
        self.actualitzacio_tauler()
        self.actualitzacio_wkb()
        
        
        self.var_parametres()
        
        if self.control_mode == 'schrodinger':
            self.actualitzacio_pantallaev()
            self.inici_evolucio()



            
    
    def right(self):
        
        if self.torn == 'red' and self.control_botons == False:
        
            if self.posx_red == self.maxpos:
                self.posx_red = self.posx_red
        
            else:
                self.stop_evolucio_barr()
                self.posx_red = self.posx_red + self.salt
                self.actualposx_red = self.actualposx_red + 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_red,self.actualposx_red -1,self.index_barr]
                print(self.amplada)
            

                
        elif self.torn == 'blue' and self.control_botons == False:
        
            if self.posx_blue == self.maxpos:
                self.posx_blue = self.posx_blue
        
            else:
                self.stop_evolucio_barr()
                self.posx_blue = self.posx_blue + self.salt
                self.actualposx_blue = self.actualposx_blue + 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_blue,self.actualposx_blue -1,self.index_barr]
                print(self.amplada)
                
        self.control_botons = True
    
        self.actualitzacio_tauler()
        self.actualitzacio_wkb()
        
        
        self.var_parametres()
        
        if self.control_mode == 'schrodinger':
            self.actualitzacio_pantallaev()
            self.inici_evolucio()


            
    def left(self):
        
        if self.torn == 'red' and self.control_botons == False:
        
            if self.posx_red == self.minpos:
                self.posx_red = self.posx_red
        
            else:
                self.stop_evolucio_barr()
                self.posx_red = self.posx_red - self.salt
                self.actualposx_red = self.actualposx_red - 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_red,self.actualposx_red,self.index_barr]
                print(self.amplada)
                
                
                
        elif self.torn =='blue' and self.control_botons == False:
        
            if self.posx_blue == self.minpos:
                self.posx_blue = self.posx_blue
        
            else:
                self.stop_evolucio_barr()
                self.posx_blue = self.posx_blue - self.salt
                self.actualposx_blue = self.actualposx_blue - 1
                self.amplada = GlobalShared.barr_vert[self.actualposy_blue,self.actualposx_blue,self.index_barr]
                print(self.amplada)
        
        self.control_botons = True
                
        self.actualitzacio_tauler()
        self.actualitzacio_wkb()
        
        
        self.var_parametres()
        
        if self.control_mode == 'schrodinger':
            self.actualitzacio_pantallaev()
            self.inici_evolucio()

        
        

    """ FUNCIONS UTILITZADES DESPRÉS DE MOURE LA PARTÍCULA PER DIBUIXAR """
               
                
    def actualitzacio_tauler(self):
        
        """ Aquesta funció dibuixa el nou estat del tauler  """

        #dibuixem el nou estat del tauler
        self.box1.canvas.clear()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        self.rectangle()
        #self.imatge_tauler()
        
        self.canvi_barreres()
        
    def actualitzacio_wkb(self):
        
        if self.control_mode == 'wkb':
            
            self.potencial()
            self.T_analitic()
            self.wkb_txt.remove()
            self.wkb_txt = self.pant_wkb.text(0.1,0.5,'%.6f' % self.T)
        
            self.pant_wkb_canvas.draw()
            
        
    def actualitzacio_potencial(self):
        
        self.pot_data.remove()
        self.potencial()
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.main_canvas.draw()
        
        
    def actualitzacio_pantallaev(self):
        
        """ Fa un reset de l'ona i la dibuixa amb el color corresponent al torn """
        
        self.pot_data.remove()
        self.visu_ona.remove()
        self.potencial()
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.torn)
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.torn)
        self.main_canvas.draw()
        
  
        

    """___________________ CONTROL DEL JOC ____________________________ """
               
    def actualitzacio(self):
        
        """ Controla que s' la partícula ja s'ha mogut, es pugui fer el canvi de torn.
        Actualitza doncs la pantalla on surt la probabilitat d'haver passat de cada
        partícula, resta una part a la partícula (proporcional al que no ha passat),
        actualitza el tauler i la pantalla on es veu l'evolució. A més a més canvia el torn.
        També comprova que el joc no s'hagi acabat"""
        
        if self.control_botons == True: 
            
            if self.control_mode == 'wkb':
                GlobalShared.prob = self.T

            if self.torn == 'red':
                self.prob_txt.remove()
                self.red_prob = self.red_prob*GlobalShared.prob*GlobalShared.bonus
                self.prob_txt = self.pantprob_red.text(0.1,0.5,'%.7f' % self.red_prob)
                
                self.angle_end_red = self.angle_end_red - 50*(1 - GlobalShared.prob)
        
                self.pantprob_red_canvas.draw()
            
                
           
            
            elif self.torn == 'blue':
                self.probb_txt.remove()
                self.blue_prob = self.blue_prob*GlobalShared.prob*GlobalShared.bonus
                self.probb_txt = self.pantprob_blue.text(0.1,0.5,'%.7f' % self.blue_prob)
                
                self.angle_end_blue = self.angle_end_blue - 50*(1 - GlobalShared.prob)
        
                self.pantprob_blue_canvas.draw()
            

            self.canvi_torn()
        
            #hem de redibuixar el tauler ja que quan canvia 
            #el torn canvia el color del contorn
            self.actualitzacio_tauler()
            
            #Si s'ha activat la evolució, demanem que s'aturi
            if self.control_evolucio == True:
                self.control_evolucio = False
                self.stop_evolucio()
        
            #dibuixem la barrera de nou amb el color corresponent
            if self.control_mode == 'schrodinger':
                self.actualitzacio_pantallaev()
            
            #aquesta funció també comprovarà que el joc no s'hagi acabat
            self.final_joc()
            
    def canvi_torn(self):
        
        if self.torn == 'red' and self.end_red == False:
            self.torn = 'blue'
            
        elif self.torn == 'blue' and self.end_blue == False:
            self.torn = 'red'
        

        self.control_botons = False
        
    def final_joc(self):
        
        """ Aquesta funció controla primerament quan les partícules arriben a la
        casella final. Si hi arriben els hi treu el torn i l'altra partícula
        continua jugant fins que arriba també. Un cop arribades les dues comprova
        quina de les dues té més probabilitats d'haver arribat """
        
        if self.actualposy_red == 0 and self.actualposx_red == 2:
            print('LA PARTICULA VERMELLA HA ACABAT')
            self.end_red = True
        
        if self.actualposy_blue == 0 and self.actualposx_blue == 2:
            print('LA PARTÍCULA BLAVA HA ACABAT')
            self.end_blue = True
        
        if self.end_blue == True and self.end_red == True:
            
            print('EL JOC HA ACABAT')
            #ara comrpovem qui ha guanyat
            if self.blue_prob > self.red_prob:
                print('HA GUANYAT LA PARTÍCULA BLAVA')
                self.torn_txt.remove()
                self.pantorn.patch.set_facecolor('blue')
                self.torn_txt = self.pantorn.text(0.1,0.5,"BLUE PARTICLE WINNER!",color = 'black')
                self.pantorn_canvas.draw()
                
            elif self.red_prob > self.blue_prob:
                print('HA GUANYAT LA PARTÍCULA VERMELLA')
                self.torn_txt.remove()
                self.pantorn.patch.set_facecolor('red')
                self.torn_txt = self.pantorn.text(0.1,0.5,"RED PARTICLE WINNER!",color = 'black')
                self.pantorn_canvas.draw()
                
    """______________________ PARÀMETRES VARIABLES EN L'EVOLUCIÓ ________ """
    
    """ Les següents funcions seran cridades quan una partícula passi una barrera.
    Depenent del paràmetre que es pugui variar en aquella casella es cridarà una
    funció o una altra. Aquestes funcions crearan nous botons o sliders que 
    permetran variar el paràmetre en qüestio """
    
    def var_parametres(self):
        
        if self.torn == 'red':
        
            self.parametre = GlobalShared.caselles_tauler[self.actualposx_red,self.actualposy_red]
            
        elif self.torn == 'blue':
            
            self.parametre = GlobalShared.caselles_tauler[self.actualposx_blue,self.actualposy_blue]
            
        self.create_slider()
        
    def create_first_slider(self):
        
        self.slider = Slider(min = 67.0,max = 80.0,value = 76.0,orientation = 'horizontal',
                        value_track = True, cursor_image = 'atom.jpg',
                        cursor_size = (60,60))
        
        self.box7.add_widget(self.slider)
        
        self.label = Label(text =str(self.slider.value))
        self.box6.add_widget(self.label)
        
        return self.slider,self.label
        
    
    def create_slider(self):
        
    
        if self.parametre == 1:
            self.box7.remove_widget(self.slider)
            imatge = 'cursor_amplada.png'
            self.slider = Slider(min = 0.2,max = 1.2,value =0.5,orientation = 'horizontal',
                        value_track = True, cursor_image = imatge,
                        cursor_size = (60,60))
            
            
        elif self.parametre == 2:
            self.box7.remove_widget(self.slider)
            imatge = 'cursor_alçada.png'
            self.slider = Slider(min = 67.0,max = 80.0,value = 76.0,orientation = 'horizontal',
                        value_track = True, cursor_image = imatge,
                        cursor_size = (60,60))
        
            
            
        self.box7.add_widget(self.slider)
        
        return self.slider
    
    
    def label_slider(self):
        
        self.box6.remove_widget(self.label)
        
        self.label = Label(text =str(self.slider.value))
        
        self.box6.add_widget(self.label)
        
        
        return self.label
        
        
        
        
    
    def on_touch_Slider(self):
        
     
        if self.control_mode == 'schrodinger' and self.control_evolucio == True:
            
            if self.parametre == 1:
            
                self.amplada = self.slider.value
                self.actualitzacio_potencial()
                self.actualitzacio_wkb()
                self.label_slider()
            
                
            elif self.parametre == 2:
                    
                self.bheight_quadrat = self.slider.value
                self.actualitzacio_potencial()
                self.actualitzacio_wkb()
                self.label_slider()
                
        elif self.control_mode == 'wkb':
            
            if self.parametre == 1:
            
                self.amplada = self.slider.value
                self.actualitzacio_wkb()
                self.label_slider()
            
                
            elif self.parametre == 2:
                    
                self.bheight_quadrat = self.slider.value
                self.actualitzacio_wkb()
                self.label_slider()
            
            
            

                
            
            
    
                    
                
    """ ______________________ CÀLCULS ONA I BARRERA ___________________ """                

    def potencial(self):
        
        self.pot = np.zeros(len(self.xx))
        
        if self.control_pot == 'quadrat':
            for i in range(0,len(self.xx)):
                if -self.amplada/2.0 < self.xx[i] < self.amplada/2.0:
                    self.pot[i] = self.bheight_quadrat
        elif self.control_pot == 'gauss':
            self.pot = 42.55*(np.exp(-(self.xx/self.amplada)**2))/np.sqrt(self.bheight_gauss*np.pi)

    def phi0(self):
        
        self.phi0 = 1j*np.zeros(len(self.xx))
        self.phi0 = (1.0/(2.0*np.pi)**(1.0/4.0))*np.exp((-(self.xx - 7.0)**2)/4.0)*np.exp(1j*self.xx*self.p0)
        self.energia()
        self.T_analitic()
        
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
        
    def inici_evolucio(self):
        
        if self.control_mode == 'schrodinger':
        
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
        
            #Controlem si s'ha activat la evolucio
            self.control_evolucio = True
        
        
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
        self.actualitzacio()
        self.control_evolucio = False
    
    def matriuB(self):
        
        self.matriu_B = 1j*np.zeros((self.nx -1,self.nx-1))
        for i in range(0,self.nx-1):
            self.matriu_B[i,i] = 2.0*(1.0 -self.r) - 1.0j*self.pot[i]*self.dt
            if i != (self.nx-2):
                self.matriu_B[i,i+1] = self.r
                self.matriu_B[i+1,i] = self.r
                
    
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
        self.visu_ona, = self.ona.fill(self.xx,self.phi2,color = self.torn)
        
        
        self.main_canvas.draw()
        
        if np.mod(self.t,5.0) == 0.0:
            
            #Cridem la funció que integra la funcio d'ona a l'esquerra
            self.prob_left()
        
            self.wkb_txt.remove()
            self.wkb_txt = self.pant_wkb.text(0.1,0.5,'%.6f' % self.probleft)
        
            self.pant_wkb_canvas.draw()
            
            
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
    
    def prob_left(self):
        
        n = int(self.nx/2)
        
        self.probleft = self.simpson2(self.dx,self.phi2[0:n])/self.simpson2(self.dx,self.phi2)
          
        

    """_________________POPUP INFORMATIU________________________________"""
    
    def openpop(self):
        
        popup = InfoPopup()
        popup.open() 
        




class InfoPopup (Popup):
    
    def __init__(self):
        
        super(Popup, self).__init__() 
        
    def ipseudo_init(self):
        
        label = Label(text = "En aquest Popup s'explicarà el joc i la diferència entre els dos modes")
        self.box1.add_widget(label)
        
        return label











"""__________________________SIMULATION SCREEN___________________________"""
           
        
class SimulationScreen(Screen):
    
    
    def __init__(self,**kwargs):
        
        super(SimulationScreen, self).__init__(**kwargs)
        
    def transition_SS(self):
        
        self.manager.current = 'starting'
        self.manager.transition = FadeTransition()
    

    def spseudo_init(self): 
        
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
        self.bwidth = 0.5
        self.control_pot = 'quadrat'
        self.potencial()
        self.max_pot = 90.0
        self.min_pot = 0.0
        self.phi0()
        self.phi02 = abs(self.phi0)**2
        
        self.colpot = 'yellow'
        self.cole = 'green'
        self.colphi = 'red'
        self.color = 'red'
  
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
    













    
    

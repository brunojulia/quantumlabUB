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


"""IDEA DEL JOC: a cada casella hi haurà una target probability que el jugador
haurà de conseguir. Primer, amb la predicció wkb establirà unes condicions inicials
i després començarà l'evolució, on es podran modificar els paràmetres de barrera
per poder ajustar-se més a la target probability"""



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

            
        #definim on comença i on acaba el tauler
        self.inrect = 50.0
        self.finrect = 410.0*1.5
        
        self.amplada = 0.1
        
        self.red_prob = 0.0
        self.blue_prob = 0.0
        
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
        self.control_pot = 'gauss'
        self.potencial()
        self.max_pot = 90.0
        self.min_pot = 0.0
        self.sigma = 1.0
        self.calc_phi0()
        self.color = self.torn
        
        self.colpot = 'yellow'
        self.cole = 'green'
        self.colphi = 'red'
        
        self.control_evolucio = False
        
        self.targetprob = 1.0
        
        """ Un cop tot definit cridem les funcions que ho dibuixen """
        
        self.imatge_tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        self.tauler()
        self.rectangle()
        
        self.pantalles()
        self.pantalla_evolucio()

        self.botons_parametres()
    

        
  
    def pantalles(self):
        
        """ Dibuixa les pantalles amb les probabilitats """
       
        self.pantprob_red = plt.figure()
        self.pantprob_red.patch.set_facecolor('black')
        self.pantprob_red_canvas =  FigureCanvasKivyAgg(self.pantprob_red)
        self.box4.add_widget(self.pantprob_red_canvas)
       
        self.prob_txt = self.pantprob_red.text(0.1,0.5,'%.0f' % (self.red_prob),color = 'red')
       
        self.pantprob_blue = plt.figure()
        self.pantprob_blue.patch.set_facecolor('black')
        self.pantprob_blue_canvas =  FigureCanvasKivyAgg(self.pantprob_blue)
        self.box2.add_widget(self.pantprob_blue_canvas)
       
        self.probb_txt = self.pantprob_blue.text(0.1,0.5,'%.0f' % (self.blue_prob),color = 'blue')
    
       
        """ Dibuixa la pantalla amb el valor del WKB """
        self.pant_wkb = plt.figure()
        self.pant_wkb.patch.set_facecolor('white')
        self.pant_wkb_canvas = FigureCanvasKivyAgg(self.pant_wkb)
        self.box5.add_widget(self.pant_wkb_canvas)
        
        self.wkb_txt = self.pant_wkb.text(0.1,0.5,'0.00')
        
        """Dibuixa la pantalla amb el valor de la probabilitat a l'esquerra """
        self.pant_probleft = plt.figure()
        self.pant_probleft.patch.set_facecolor('white')
        self.pant_probleft_canvas = FigureCanvasKivyAgg(self.pant_probleft)
        self.box6.add_widget(self.pant_probleft_canvas)
        
        self.probleft_txt = self.pant_probleft.text(0.1,0.5,'0,00000')
        
        
        """ Dibuixa la pantalla on hi haurà la target probability """
        self.pant_target = plt.figure()
        self.pant_target.patch.set_facecolor('white')
        self.pant_target_canvas = FigureCanvasKivyAgg(self.pant_target)
        self.box17.add_widget(self.pant_target_canvas)
        
        self.target_txt = self.pant_target.text(0.1,0.5,'-')
        
        """ Dibuixa la pantalla amb l'error comès """
        #self.pant_error = plt.figure()
        #self.pant_error.patch.set_facecolor('white')
        #self.pant_error_canvas = FigureCanvasKivyAgg(self.pant_error)
        #self.box22.add_widget(self.pant_error_canvas)
        
        #self.error_txt = self.pant_error.text(0.1,0.5,'-')
        
        
        
    def pantalla_evolucio(self):
        
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black')
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        #self.main_canvas.bind(on_touch_up = self.control_teclat)
        self.box3.add_widget(self.main_canvas)
            
        self.pot_graf = plt.subplot()
        self.pot_graf.set_facecolor('black')
        self.pot_graf.axis('off')
        self.pot_graf.axis([self.lx_min, self.lx_max, self.min_pot, self.max_pot])
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color = self.colpot)
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.torn)
        
        self.ona = self.pot_graf.twinx()
        self.ona.axis('off')
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.torn)
        self.ona.axis([self.lx_min,self.lx_max,0.0,1.0])
        
        self.main_fig.tight_layout() 
        self.main_canvas.draw()
        self.eixos()
        
    def eixos(self):
        
        fweight = 600
        midp = (self.max_pot - self.min_pot) / 2.0
        fakeax_x = self.lx_min
        fakeax_text = 0.5
        
        self.pot_graf.text(fakeax_x - fakeax_text, midp,             #POTENTIAL 
                           'Potential/Energy',
                           weight = fweight, color = 'grey', 
                           rotation = 90, va = 'center', ha = 'center')
        self.pot_graf.text(fakeax_x - fakeax_text, self.max_pot,      #Bot lim
                           str(self.max_pot), 
                           weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')
        self.pot_graf.text(fakeax_x - fakeax_text, self.min_pot,      #Top lim
                           str(self.min_pot), 
                           weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')
        
        mido= 0.5

        self.ona.text(-fakeax_x + fakeax_text, mido,                 #PROB
                           'Probability' ,
                           weight = fweight, color = 'grey', 
                           rotation = 90, va = 'center', ha = 'center')
        self.ona.text(-fakeax_x + fakeax_text, 1.0,    
                           '1.0', weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')
        self.ona.text(-fakeax_x + fakeax_text, 0.0,     #Top lim
                           '0.0', weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')


        
            

    """________FUNCIONS QUE DIBUIXEN ELS ELEMENTS DEL TAULER________________ """
    
    """Aquestes funcions dibuixen les barreres del tauler cada cop que les funcions 
    que controlen l'evolució temporal de les barreres les criden. També dibuixen les 
    partícules en la seva posicio corresponent i el rectangle extern del tauler. Són
    cridades totes en conjunt en la funcio actualitzacio_tauler """


    def draw_redcircle(self):
        
        with self.box1.canvas:
            
            Color(1,0,0,GlobalShared.red_life)
            Ellipse(pos=(self.posx_red,self.posy_red),size = (50,50))
                 
        
        self.red_particle = FloatLayout()
        self.box1.add_widget(self.red_particle)
        
        return self.red_particle
        
        
    def draw_bluecircle(self):
        
           
        with self.box1.canvas:
                
            Color(0,0,1,GlobalShared.blue_life)
            Ellipse(pos=(self.posx_blue,self.posy_blue),size=(50,50))
            
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
                #amp_bar = GlobalShared.barr_vert[j,i-1,0]
                
                with self.box1.canvas:
                    Color(1,1,0,mode='rgb')
                    Line(bezier=(x,y_1,x,y_2),width = 3)
                    
        for i in range(1,len(self.bar_pos_y)-1):
            for j in range(0,len(self.bar_pos_x)-1):
                y = self.bar_pos_y[i]
                x_1 = self.bar_pos_x[j] + self.marge
                x_2 = self.bar_pos_x[j+1] - self.marge
                #amp_bar = GlobalShared.barr_hor[j-1,i,0]
                
                with self.box1.canvas:
                    Color(1,1,0,mode='rgb')
                    Line(bezier=(x_1,y,x_2,y),width = 3)
        
                    
    
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
        
        self.image_end = Image(source = 'Images/end.png')
        self.image_end.pos = (315.,585.)
        self.image_end.allow_stretch = True
        self.image_end.size_hint_x = 0.2
        self.image_end.size_hint_y = 0.2

        self.box1.add_widget(self.image_end)
        
        
        for i in range(0,5):
            for j in range(0,5):
                
                if i == 0 and j == 2:
                    pass
                    
                
                else:
                    targetprob = GlobalShared.nivell1[i,j]
                    posx = self.minpos + (j*self.salt)
                    posy = self.maxpos - (i*self.salt)
                
                    with self.box1.canvas:
                        Color(1,1,1,(1*targetprob))
                        Ellipse(pos=(posx,posy),size = (50,50))

        
                    
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
                
                self.posy_red = self.posy_red - self.salt
                self.actualposy_red = self.actualposy_red + 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_red,self.actualposx_red]
                GlobalShared.red_life = GlobalShared.red_life - ((1-self.targetprob)*0.2)
                
        elif self.torn == 'blue' and self.control_botons == False:
            self.amplada = self.botons_amp[0]
        
            if self.posy_blue == self.minpos:
                self.posy_blue = self.posy_blue
        
            else:
                self.posy_blue = self.posy_blue - self.salt
                self.actualposy_blue = self.actualposy_blue + 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_blue,self.actualposx_blue]
                GlobalShared.blue_life = GlobalShared.blue_life - ((1 - self.targetprob)*0.2)
            
        self.control_botons = True            
                
        self.actualitzacio_tauler()
        self.actualitzacio_target()
        self.actualitzacio_wkb()
        

        self.actualitzacio_pantallaev()
        self.control_flux_botons()
        
        #hem de vigilar també que no haguem arribat a la ultima casella
        self.final_joc()



                
                
   

            
    def up(self):
        
        if self.torn == 'blue' and self.control_botons == False:
        
            if self.posy_blue == self.maxpos:
                self.posy_blue = self.posy_blue
        
            else:
                self.posy_blue = self.posy_blue + self.salt
                self.actualposy_blue = self.actualposy_blue - 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_blue,self.actualposx_blue]
                GlobalShared.blue_life = GlobalShared.blue_life - ((1 - self.targetprob)*0.2)

            
        elif self.torn == 'red' and self.control_botons == False:    
        
            if self.posy_red == self.maxpos:
                self.posy_red = self.posy_red
        
            else:
                self.posy_red = self.posy_red + self.salt
                self.actualposy_red = self.actualposy_red - 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_red,self.actualposx_red]
                GlobalShared.red_life = GlobalShared.red_life - ((1-self.targetprob)*0.2)
    
        print(self.actualposy_red)
        print(self.actualposx_red)
        
        self.control_botons = True
                
        self.actualitzacio_tauler()
        self.actualitzacio_target()
        self.actualitzacio_wkb()

        self.actualitzacio_pantallaev()
        self.control_flux_botons()

        #hem de vigilar també que no haguem arribat a la ultima casella
        self.final_joc()


            
    
    def right(self):
        
        if self.torn == 'red' and self.control_botons == False:
        
            if self.posx_red == self.maxpos:
                self.posx_red = self.posx_red
        
            else:
                self.posx_red = self.posx_red + self.salt
                self.actualposx_red = self.actualposx_red + 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_red,self.actualposx_red]
                GlobalShared.red_life = GlobalShared.red_life - ((1-self.targetprob)*0.2)
            

                
        elif self.torn == 'blue' and self.control_botons == False:
        
            if self.posx_blue == self.maxpos:
                self.posx_blue = self.posx_blue
        
            else:
                self.posx_blue = self.posx_blue + self.salt
                self.actualposx_blue = self.actualposx_blue + 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_blue,self.actualposx_blue]
                GlobalShared.blue_life = GlobalShared.blue_life - ((1 - self.targetprob)*0.2)

                
        self.control_botons = True
    
        self.actualitzacio_tauler()
        self.actualitzacio_target()
        self.actualitzacio_wkb()

        self.actualitzacio_pantallaev()
        self.control_flux_botons()

        #hem de vigilar també que no haguem arribat a la ultima casella
        self.final_joc()

            
    def left(self):
        
        if self.torn == 'red' and self.control_botons == False:
        
            if self.posx_red == self.minpos:
                self.posx_red = self.posx_red
        
            else:
                self.posx_red = self.posx_red - self.salt
                self.actualposx_red = self.actualposx_red - 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_red,self.actualposx_red]
                GlobalShared.red_life = GlobalShared.red_life - ((1-self.targetprob)*0.2)
                
                
                
        elif self.torn =='blue' and self.control_botons == False:
        
            if self.posx_blue == self.minpos:
                self.posx_blue = self.posx_blue
        
            else:
                self.posx_blue = self.posx_blue - self.salt
                self.actualposx_blue = self.actualposx_blue - 1
                self.targetprob = GlobalShared.nivell1[self.actualposy_blue,self.actualposx_blue]
                GlobalShared.blue_life = GlobalShared.blue_life - ((1 - self.targetprob)*0.2)
        
        self.control_botons = True
                
        self.actualitzacio_tauler()
        self.actualitzacio_target()
        self.actualitzacio_wkb()

        self.actualitzacio_pantallaev()
        self.control_flux_botons()

        #hem de vigilar també que no haguem arribat a la ultima casella
        self.final_joc()
        

    """ FUNCIONS UTILITZADES DESPRÉS DE MOURE LA PARTÍCULA PER DIBUIXAR """
               
                
    def actualitzacio_tauler(self):
        
        """ Aquesta funció dibuixa el nou estat del tauler  """

        #dibuixem el nou estat del tauler
        self.box1.canvas.clear()
        self.imatge_tauler()
        self.tauler()
        self.draw_redcircle()
        self.draw_bluecircle()
        self.rectangle()
        
        
    def actualitzacio_wkb(self):
        
        self.potencial()
        self.T_analitic()

        self.wkb_txt.remove()
        self.wkb_txt = self.pant_wkb.text(0.1,0.5,'%.2f' % self.T)
        
        self.pant_wkb_canvas.draw()
        
        #que s'actualitzi l'error cada cop que canvia el wkb
        #self.actualitzacio_error()
        
            
        
    def actualitzacio_potencial(self):
        
        self.pot_data.remove()
        self.potencial()
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.main_canvas.draw()
        
        
    def actualitzacio_pantallaev(self):
        
        """ Fa un reset de l'ona i la dibuixa amb el color corresponent al torn """
        
        self.pot_data.remove()
        self.visu_ona.remove()
        self.e_data.remove()
        self.potencial()
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.torn)
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color=self.colpot)
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.torn)
        self.main_canvas.draw()
    
    def actualitzacio_target(self):
        
        if self.torn == 'red':
            self.targetprob = GlobalShared.nivell1[self.actualposy_red,self.actualposx_red]
            self.target_txt.remove()
            self.target_txt = self.pant_target.text(0.1,0.5,str(self.targetprob))
            
            self.pant_target_canvas.draw()
            
        elif self.torn == 'blue':
            self.targetprob =  GlobalShared.nivell1[self.actualposy_blue,self.actualposx_blue]
            self.target_txt.remove()
            self.target_txt = self.pant_target.text(0.1,0.5,str(self.targetprob))
            
            self.pant_target_canvas.draw()
            
    def actualitzacio_error(self):
        
        
        if self.control_evolucio == False:
            self.error = abs(self.targetprob - self.T)
        elif self.control_evolucio == True:
            self.error = abs(self.targetprob - self.probleft)
            
        self.error_txt.remove()
        self.error_txt = self.pant_error.text(0.1,0.5,self.error)
        self.pant_error_canvas.draw()
        
        
  
        

    """___________________ CONTROL DEL JOC ____________________________ """
               
    def actualitzacio(self):
        
        """ Controla que s' la partícula ja s'ha mogut, es pugui fer el canvi de torn.
        Actualitza doncs la pantalla on surt la probabilitat d'haver passat de cada
        partícula, resta una part a la partícula (proporcional al que no ha passat),
        actualitza el tauler i la pantalla on es veu l'evolució. A més a més canvia el torn.
        També comprova que el joc no s'hagi acabat"""
        
            
        #primer s'ha d'aturar l'evolucio
        self.stop_evolucio()
        self.control_evolucio = False
        
        self.control_botons = False
        
        self.control_flux_botons()
        
        
        
        self.error = abs(self.probleft - self.targetprob)
        
        
            
        if self.torn == 'red':
            #cridem la funcio de ountuacio
            self.puntuacio()
            
            self.prob_txt.remove()
            self.prob_txt = self.pantprob_red.text(0.1,0.5,'%.0f' % self.red_prob,color = 'red')
        
            self.pantprob_red_canvas.draw()
            
                
        elif self.torn == 'blue':
            #cridem la funcio de puntuacio
            self.puntuacio()
            
            self.probb_txt.remove()
            self.probb_txt = self.pantprob_blue.text(0.1,0.5,'%.0f' % self.blue_prob,color = 'blue')
    
        
            self.pantprob_blue_canvas.draw()
            
        #tornem a definir els paràmetres perquè no quedin com l'anterior jugador
        #els ha deixat
        self.p0 = -200.0/self.lx_max
        self.bheight_quadrat = 76.0
        self.bheight_gauss = 0.1
        self.sigma = 1.0
         

        self.canvi_torn()
        
        #hem de redibuixar el tauler ja que quan canvia 
        #el torn canvia el color del contorn
        self.actualitzacio_tauler()
        self.actualitzacio_pantallaev()
        
        #Posem a 0 els valors de wkb i left probability de les pantalles
        self.wkb_txt.remove()
        self.wkb_txt = self.pant_wkb.text(0.1,0.5,'0.00')
        self.pant_wkb_canvas.draw()
        
        self.probleft_txt.remove()
        self.probleft_txt = self.pant_probleft.text(0.1,0.5,'0.00')
        
        self.pant_probleft_canvas.draw()
        
        self.target_txt.remove()
        self.target_txt = self.pant_target.text(0.1,0.5,'-')
            
        self.pant_target_canvas.draw()
        
            
        
           
    def puntuacio(self):
        
        if self.torn == 'red':
            if self.error <= 0.1:
                self.red_prob = self.red_prob + 3
            elif 0.1 < self.error <= 0.3:
                self.red_prob = self.red_prob + 2    
            elif 0.3 < self.error <= 0.5:
                self.red_prob = self.red_prob + 1
            elif 0.5 < self.error:
                self.red_prob = self.red_prob + 0
                
        elif self.torn == 'blue':
            if self.error <= 0.1:
                self.blue_prob = self.blue_prob + 3
            elif 0.1 < self.error <= 0.3:
                self.blue_prob = self.blue_prob + 2    
            elif 0.3 < self.error <= 0.5:
                self.blue_prob = self.blue_prob + 1
            elif 0.5 < self.error:
                self.blue_prob = self.blue_prob + 0
            
    def canvi_torn(self):
        
        if self.torn == 'red' and self.end_red == False:
            self.torn = 'blue'
            
        elif self.torn == 'blue' and self.end_blue == False:
            self.torn = 'red'
            
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
    

        
    def botons_parametres(self):
        

        """ Botons de moment inicial i sigma del paquet d'ones:
            Aquests botons només poden funcionar abans de l'evolució
            i just després que el jugadoir hagi migut la fitxa """
            
        self.boto_px_up = Button(text = '+')
        self.boto_px_up.bind(on_press = self.px_up)
        self.box7.add_widget(self.boto_px_up)
        
        self.boto_px_down = Button(text = '-')
        self.boto_px_down.bind(on_press = self.px_down)
        self.box8.add_widget(self.boto_px_down)
        
        self.boto_amp_up = Button(text = '+')
        self.boto_amp_up.bind(on_press = self.amp_up)
        self.box9.add_widget(self.boto_amp_up)
        
        self.boto_amp_down = Button(text = '-')
        self.boto_amp_down.bind(on_press = self.amp_down)
        self.box10.add_widget(self.boto_amp_down)
        
        
        """ Botons d'amplada i alçada de la barrera """
        self.boto_add_width = Button(text = '+')
        self.boto_add_width.bind(on_press = self.add_width)
        self.box11.add_widget(self.boto_add_width)
        
        self.boto_subtract_width = Button(text = '-')
        self.boto_subtract_width.bind(on_press = self.subtract_width)
        self.box12.add_widget(self.boto_subtract_width)
        
        self.boto_add_height = Button(text='+')
        self.boto_add_height.bind(on_press = self.add_bheight)
        self.box13.add_widget(self.boto_add_height)
        
        self.boto_subtract_height = Button(text='-')
        self.boto_subtract_height.bind(on_press = self.subtract_bheight)
        self.box14.add_widget(self.boto_subtract_height)
        
        
    def control_flux_botons(self):
        
        #si ja s'ha tocat un dels botons de posicio, han de tornar al color gris
        if self.control_botons == True and self.control_evolucio == False and self.torn == 'red':
            
            self.boto_px_up.background_color = (1,0,0,1)
            self.boto_px_down.background_color = (1,0,0,1)
            self.boto_amp_up.background_color = (1,0,0,1)
            self.boto_amp_down.background_color = (1,0,0,1)
            
        elif self.control_botons == True and self.control_evolucio == True and self.torn == 'red':
            self.boto_px_up.background_color = (1,1,1,1)
            self.boto_px_down.background_color = (1,1,1,1)
            self.boto_amp_up.background_color = (1,1,1,1)
            self.boto_amp_down.background_color = (1,1,1,1)
            
            self.boto_add_width.background_color = (1,0,0,1)
            self.boto_add_height.background_color = (1,0,0,1)
            self.boto_subtract_height.background_color = (1,0,0,1)
            self.boto_subtract_width.background_color = (1,0,0,1)
            
            
        elif self.control_botons == False and self.control_evolucio == False and self.torn == 'red':
            
            self.boto_add_width.background_color = (1,1,1,1)
            self.boto_add_height.background_color = (1,1,1,1)
            self.boto_subtract_height.background_color = (1,1,1,1)
            self.boto_subtract_width.background_color = (1,1,1,1)
            
        elif self.control_botons == True and self.control_evolucio == False and self.torn == 'blue':
            
            self.boto_px_up.background_color = (0,0,1,1)
            self.boto_px_down.background_color = (0,0,1,1)
            self.boto_amp_up.background_color = (0,0,1,1)
            self.boto_amp_down.background_color = (0,0,1,1)
            
        elif self.control_botons == True and self.control_evolucio == True and self.torn == 'blue':
            self.boto_px_up.background_color = (1,1,1,1)
            self.boto_px_down.background_color = (1,1,1,1)
            self.boto_amp_up.background_color = (1,1,1,1)
            self.boto_amp_down.background_color = (1,1,1,1)
            
            self.boto_add_width.background_color = (0,0,1,1)
            self.boto_add_height.background_color = (0,0,1,1)
            self.boto_subtract_height.background_color = (0,0,1,1)
            self.boto_subtract_width.background_color = (0,0,1,1)
            
        elif self.control_botons == False and self.control_evolucio == False and self.torn == 'blue':
            
            self.boto_add_width.background_color = (1,1,1,1)
            self.boto_add_height.background_color = (1,1,1,1)
            self.boto_subtract_height.background_color = (1,1,1,1)
            self.boto_subtract_width.background_color = (1,1,1,1)
            
            

            
        
            
        
                
        
    def px_up(self,instance):
        
        """ Aquest botó només funcionarà quan el jugador ja hagi avançat en el
        tauler i quan l'evolució encara no hagi començat. També controlarem que
        al donar més moment inicial, i per tant més energia, aquesta no sobrepassi
        el màxim potencial de la barrera, ja que llavors el càlcul del WKB 
        dóna errors """
        if self.control_botons == True and self.control_evolucio == False:
        
            if (self.control_pot == 'gauss' and  abs(self.E - max(self.pot)) > 0.6) or (self.control_pot == 'quadrat' and  abs(self.E - self.bheight_quadrat) > 0.6):
            
                self.p0 = self.p0 - 0.1 #el signe és negatiu per la direcció
        
                self.calc_phi0()
                self.actualitzacio_wkb()
                self.actualitzacio_pantallaev()
    
        
    def px_down(self,instance):
        
        if self.control_botons == True and self.control_evolucio == False:
            
            self.p0 = self.p0 + 0.1 
        
            self.calc_phi0()
            self.actualitzacio_wkb()
            self.actualitzacio_pantallaev()
        
    def amp_up(self,instance):
        
        if self.control_botons == True and self.control_evolucio == False:
        
            if self.sigma > 0.5:
                self.sigma = self.sigma - 0.5
        
            self.calc_phi0()
            self.actualitzacio_wkb()
            self.actualitzacio_pantallaev()
            #self.calc_norma()

        
    def amp_down(self,instance):
        
        if self.control_botons == True and self.control_evolucio == False:
            self.sigma = self.sigma + 0.5
        
            self.calc_phi0()
            self.actualitzacio_wkb()
            self.actualitzacio_pantallaev()
            #self.calc_norma()

        
    def add_width(self,instance):
    
        if self.control_botons == True and self.control_evolucio == True:
        
            if self.control_pot == 'quadrat': 
                self.amplada = self.amplada + 0.05 
            elif self.control_pot == 'gauss':
                self.amplada = self.amplada + 0.05
            
            self.actualitzacio_potencial()
    
    def subtract_width(self,instance):
        
        if self.control_botons == True and self.control_evolucio == True:
        
            if self.control_pot == 'quadrat':
                self.amplada = self.amplada - 0.05
            if self.control_pot == 'gauss':
                self.amplada = self.amplada - 0.05
        
            self.actualitzacio_potencial()
        
    def add_bheight(self,instance):
        
        if self.control_botons == True and self.control_evolucio == True:
        
            if self.control_pot == 'quadrat':
                self.bheight_quadrat = self.bheight_quadrat + 0.5
            elif self.control_pot == 'gauss':
                self.bheight_gauss = self.bheight_gauss - 0.005
            
            self.actualitzacio_pantallaev()
        
    def subtract_bheight(self,instance):
        
        if self.control_botons == True and self.control_evolucio == True:
        
            if (self.control_pot == 'quadrat' and abs(self.E - max(self.pot)) > 0.6):
                self.bheight_quadrat = self.bheight_quadrat - 0.5
                
            elif (self.control_pot == 'gauss' and abs(self.E - max(self.pot)) > 0.6):

                self.bheight_gauss = self.bheight_gauss + 0.005
 

            self.actualitzacio_pantallaev()
        
    def canvi_potencial_gauss(self):
        
        self.control_pot = 'gauss'
        
        self.actualitzacio_pantallaev()
        self.actualitzacio_wkb()
        
    def canvi_potencial_quadrat(self):
        
        self.control_pot = 'quadrat'
        
        self.actualitzacio_pantallaev()
        self.actualitzacio_wkb()
        
    def calc_norma(self):
        
        self.norma = self.simpson2(self.dx,self.phi02)


                
    """ ______________________ CÀLCULS ONA I BARRERA ___________________ """                

    def potencial(self):
        
        self.pot = np.zeros(len(self.xx))
        
        if self.control_pot == 'quadrat':
            for i in range(0,len(self.xx)):
                if -self.amplada/2.0 < self.xx[i] < self.amplada/2.0:
                    self.pot[i] = self.bheight_quadrat
        elif self.control_pot == 'gauss':
            self.pot = 42.55*(np.exp(-(self.xx/self.amplada)**2))/np.sqrt(self.bheight_gauss*np.pi)

    def calc_phi0(self):
        
        self.phi0 = 1j*np.zeros(len(self.xx))
        self.phi0 = (1.0/((self.sigma**(0.5))*((2.0*np.pi)**(1.0/4.0))))*np.exp((-(self.xx - 7.0)**2)/(4.0*(self.sigma**2)))*np.exp(1j*self.xx*self.p0)
        self.energia()
        self.phi02 = abs(self.phi0)**2
        #self.T_analitic()
        
#    def ona(self):
#        
#        self.phi0 = 1j*np.zeros(len(self.xx))
#        self.phi0 = (1.0/(2.0*np.pi)**(1.0/4.0))*np.exp((-(self.xx - 7.0)**2)/4.0)*np.exp(1j*self.xx*self.p0)
#        self.energia()
#        self.T_analitic()
        
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
        
        #Només es pot iniciar l'evolució si ja s'ha clicat algun dels botons direccio
        
        if self.control_botons == True:
        
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
            
            self.control_flux_botons()
        
        
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
        
            self.probleft_txt.remove()
            self.probleft_txt = self.pant_probleft.text(0.1,0.5,'%.2f' % self.probleft)
        
            self.pant_probleft_canvas.draw()
  #          self.actualitzacio_error()

            
#        if self.t == 150:
            
#            self.stop_evolucio()

            
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
    
    with open ('instructions.txt','r') as t:
        
        lines = t.readlines()
        
        intro_text = lines[0]  + lines[1]  + lines[2] +  lines[3] + lines[4] +  lines[5]  + lines[6] +  lines[7] + lines[8]
        
        how_to_play_text =  lines[9] + lines[10] + lines[11] + lines[12] + lines[13] + lines[14] + lines[15] + lines[16]
                    
    
    def __init__(self):
        
        super(Popup, self).__init__() 
        self.intro.text = self.intro_text
        self.howtoplay.text = self.how_to_play_text
        
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
        self.sigma = 1.0
        self.calc_phi0()
        self.phi02 = abs(self.phi0)**2
        
        self.colpot = 'yellow'
        self.cole = 'green'
        self.colphi = 'red'
        self.color = 'red'
  
        self.main_fig = plt.figure()
        self.main_fig.patch.set_facecolor('black')
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        self.main_canvas.bind(on_touch_up = self.control_teclat)
        self.box1.add_widget(self.main_canvas)
            
        self.pot_graf = plt.subplot()
        self.pot_graf.set_facecolor('black')
        self.pot_graf.axis('off')
        self.pot_graf.axis([self.lx_min, self.lx_max, self.min_pot, self.max_pot])
        self.pot_data, = self.pot_graf.fill(self.xx, self.pot,color = self.colpot)
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.color)
        
        self.ona = self.pot_graf.twinx()
        self.ona.axis('off')
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
        
        
        self.eixos()
        
    def eixos(self):
        
        fweight = 600
        midp = (self.max_pot - self.min_pot) / 2.0
        fakeax_x = self.lx_min
        fakeax_text = 0.5
        
        self.pot_graf.text(fakeax_x - fakeax_text, midp,             #POTENTIAL 
                           'Potential/Energy',
                           weight = fweight, color = 'grey', 
                           rotation = 90, va = 'center', ha = 'center')
        self.pot_graf.text(fakeax_x - fakeax_text, self.max_pot,      #Bot lim
                           str(self.max_pot), 
                           weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')
        self.pot_graf.text(fakeax_x - fakeax_text, self.min_pot,      #Top lim
                           str(self.min_pot), 
                           weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')
        
        mido= 0.5

        self.ona.text(-fakeax_x + fakeax_text, mido,                 #PROB
                           'Probability' ,
                           weight = fweight, color = 'grey', 
                           rotation = 90, va = 'center', ha = 'center')
        self.ona.text(-fakeax_x + fakeax_text, 1.0,    
                           '1.0', weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')
        self.ona.text(-fakeax_x + fakeax_text, 0.0,     #Top lim
                           '0.0', weight = fweight, color = 'grey',
                           va = 'center', ha = 'center')
        
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
        
            

    def calc_phi0(self):
        
        self.phi0 = 1j*np.zeros(len(self.xx))
        self.phi0 = (1.0/((self.sigma**(0.5))*((2.0*np.pi)**(1.0/4.0))))*np.exp((-(self.xx - 7.0)**2)/(4.0*(self.sigma**2)))*np.exp(1j*self.xx*self.p0)
        self.energia()
        self.T_analitic()
        self.phi02 = abs(self.phi0)**2
    
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


    def amp_up(self):
        
        if self.sigma > 0.5:
            self.sigma = self.sigma - 0.5
        
        self.calc_phi0()
        self.energia()
        self.T_analitic()

        self.norma = self.simpson2(self.dx,self.phi02)
            
        #haurà canviat el paquet inicial, l'energia i el wkb
        self.e_data.remove()
        self.T_txt.remove()
        self.visu_ona.remove()
        self.norma_txt.remove()
        
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.color)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.color)
        self.norma_txt = self.pant3.text(0.1,0.5,'%6f' % (self.norma))
                
        self.pant1_canvas.draw()
        self.main_canvas.draw()
        self.pant3_canvas.draw()

        
    def amp_down(self):
        

        self.sigma = self.sigma + 0.5
        
        self.calc_phi0()
        self.energia()
        self.T_analitic()

        self.norma = self.simpson2(self.dx,self.phi02)
            
        #haurà canviat el paquet inicial, l'energia i el wkb
        self.e_data.remove()
        self.T_txt.remove()
        self.visu_ona.remove()
        
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.color)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.color)
        self.norma_txt = self.pant3.text(0.1,0.5,'%6f' % (self.norma))

                
        self.pant1_canvas.draw()
        self.main_canvas.draw()
        self.pant3_canvas.draw()
       
        
    def px_up(self):
        
 
        
        if (self.control_pot == 'gauss' and  abs(self.E - max(self.pot)) > 0.6) or (self.control_pot == 'quadrat' and  abs(self.E - self.bheight_quadrat) > 0.6):
            
            self.p0 = self.p0 - 0.1 #el signe és negatiu per la direcció
        
        self.calc_phi0()
        self.energia()
        self.T_analitic()
            
        #haurà canviat el paquet inicial, l'energia i el wkb
        self.e_data.remove()
        self.T_txt.remove()
        self.visu_ona.remove()
        
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.color)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.color)
                
        self.pant1_canvas.draw()
        self.main_canvas.draw()
                
               
    
        
    def px_down(self):
        
        self.p0 = self.p0 + 0.1 
        
        self.calc_phi0()
        self.energia()
        self.T_analitic()
            
        #haurà canviat el paquet inicial, l'energia i el wkb
        self.e_data.remove()
        self.T_txt.remove()
        self.visu_ona.remove()
        
        self.e_data, = self.pot_graf.plot(self.xx,self.E_vect,color = self.color)
        self.T_txt = self.pant1.text(0.1, 0.5,'%.8f' % (self.T))
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.color)
                
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
        self.reset()
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
        
        self.visu_ona.remove()
        self.visu_ona, = self.ona.fill(self.xx,self.phi02, color = self.color)
        
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
    













    
    

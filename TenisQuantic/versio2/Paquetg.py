# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(Adrià Bravo Vidal)s
"""
import matplotlib
import cranknicolsonbe2 as ck
from kivy.app import App 
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.uix.screenmanager import FadeTransition
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
import numpy as np
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import matplotlib.pyplot as plt 
import matplotlib.colorbar as clb
from numba import jit
from functools import partial
from kivy.properties import ObjectProperty,ReferenceListProperty,NumericProperty,StringProperty,ListProperty,BooleanProperty
from kivy.graphics import Color, Ellipse,Line,Rectangle
from kivy.core.window import Window
from matplotlib.patches import Circle
############

class PaquetgApp(App):
    """Inicialitzem la aplicació"""
    
    title='PaquetApp'
    def build(self):
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    "Activarà la pantalla d'inici.Manetja les screens, claro..."
    
    def __init__(self,**kwargs):
        super(MyScreenManager,self).__init__(**kwargs)
        self.get_screen('paquet').ppseudo_init()
        self.get_screen('game').gpseudo_init()
        #self.get_screen('starting')
        #self.get_screen('paquet').g_schedule_fired()
        #self.get_screen('starting')
        
class StartingScreen(Screen):
    
    def __init__(self,**kwargs):
        super(StartingScreen,self).__init__(**kwargs)
        
    def transition_SP(self):
        """Transició des de la pantalla inicial a la pantalla de edició"""
        
        paquetscreen=self.manager.get_screen('paquet')

        paquetscreen.p_schedule_fired()
        self.manager.transition=FadeTransition()
        self.manager.current='paquet'

    def transition_SG(self):
        """Transició des de la pantalla inicial al propi joc."""
        gamescreen=self.manager.get_screen('game')
        gamescreen.g_schedule_fired()
        self.manager.transition=FadeTransition()
        self.manager.current='game'

class PaquetScreen(Screen):
    "Això és la part més important de la app. Introduïrem la pantalla inicial"
    
    #Les propietats s'han de definir a nivell de classe (millor). Per tant, 
    #les definim aquí.
    quadrat=ObjectProperty(None)
    position=ListProperty(None)
    arrow=ObjectProperty(None)
    pxslider=ObjectProperty(None)
    pyslider=ObjectProperty(None)
    fish=ObjectProperty(None)
    

    def __init__(self,**kwargs):
        "Unim la clase amb la paraula self"
        super(PaquetScreen,self).__init__(**kwargs)
        #Aquesta linia uneix keyboard amb request keyboard
        #self._keyboard = Window.request_keyboard(self._keyboard_closed, self)

        #self._keyboard.bind(on_key_down=self._on_keyboard_down)
        

        
    def ppseudo_init(self):

        "Iniciem el primer dibuix de l'aplicació"
        
        #Variables globals...
        global Vvec,avec,cvec,psi0,normavec,mode,Vvecstandard
        global setcordpress,setcordrelease,avec,cvec,value,s2
        
        ################### PARAMETRES DE LA CAIXA,DISCRETITZAT,PARTICULA########
        self.L=3.0
        self.xa=-self.L
        self.xb=self.L
   
        #Discretitzat
        self.tb=0.1
        self.ta=0
        self.dx=0.03
        self.dt=0.02
        self.Nx=np.int((2*self.L)/self.dx)

        
        #particula i constants físiques
        self.m=1
        self.hbar=1
        self.t_total=0.00
        self.x0=0.
        self.y0=0.
        self.px=0
        self.py=0
        
        #Comptador de moments
        self.i=0
        #parametre del discretitzat
        self.r=(self.dt/(4*(self.dx**2)))*(self.hbar/self.m)
        
        #Parametres del potencial infinit
        self.value=10000
        self.s2=self.dx/2.

        #Canvia el text del interior de la caixa 'parameters' amb els parametr
        #es inicials escollits
        #self.box3.dxchange.text="{}".format(self.dx)
        self.box3.longitudchange.text="{}".format(self.L)
        #self.box3.dtchange.text="{}".format(self.dt)
        
        print(self.size)

        #################### PARAMETRES NECESSARIS PER EFECTUAR ELS CALCULS######
        
        
        #################### MODES
        #mode: 0, estandard
        #      1,slit
        #      2,disparo(encara no està fet)
        
        #Posició slit

        self.x0slit=(6.4*self.L/10.)
        self.y0slit=0.

        
        ################### CONFIGURACIÓ INICIAL
            
        self.Vvec=self.Vvecstand()
        self.psi0=self.psistand(0.,0.)
        #Ens indica en quin mode estem
        self.mode=0
        
        #Per últim, construïm les matrius molt importants per efectuar els càlculs.
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        
        self.pause_state=True
        
        
        
        
        ##########################PRIMER CÀLCUL ENERGIA#####
                       
        #Calculem la matriu densitat inicial
        self.normavec=ck.norma(self.psi0,self.Nx)
    
        #Calculem moments i energia inicials i els posem al lloc corresponent
        self.norma0=ck.trapezis(self.xa,self.xb,self.dx,np.copy(self.normavec))
        self.energy0=ck.Ham(self.xa,self.xb,self.dx,np.copy(self.psi0))
        self.box3.normachange.text='{0:.3f}'.format(self.norma0)
        self.box3.energychange.text='{0:.3f}'.format(self.energy0)
        
        
        ##########################PRIMER DIBUIX###################
        
        #creem la figura recipient del plot imshow
        self.main_fig=plt.figure()
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas,1)
        #Plot principal
        self.visu=plt.axes()
        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                      extent=(-self.L,self.L,-self.L,self.L),
                                      cmap='inferno')
        self.main_fig.set_facecolor(color='xkcd:grey')
       
                
        #Posició de la figura
        self.visu.set_position([0,0.15,0.8,0.8])

        #Dibuixem tot lo dit
         
        self.main_canvas.draw()
        
        ##################################MATPLOTLIB EVENTS/RESIZING
        #self.main_canvas.mpl_connect('motion_notify_event', motionnotify)
        
        #Conectem l'event resize de matplotlib amb la funció resize_kivy.
        #Aquesta útlima ens permet saber on estan els cantons de la caixa en
        #coordenades de pixels de la finestra en un primer instant (ja que quan
        # kivy obre matplotlib per primer cop fa ja un resize), i tambe posteriorment,
        # cada cop que fem el resize. Això ens podria ser útil en algún moment.
        
        #self.main_canvas.mpl_connect('resize_event',resize)
        self.main_canvas.mpl_connect('resize_event',partial(self.resize_kivy))
        

        self.arrowcolor=np.array([231/255.,30/255.,30/255.])
        self.arrowwidth=2.
        self.arrowdist=15.
        self.arrowdist2=8.5
        ################################POTENCIAL MODE/Widget de dibuix
        self.ara=0
        self.setcordpress=np.array([])
        self.setcordrelease=np.array([])
        self.paint=MyPaintWidget()
        self.arrow=Arrow()
        self.slitcanvas=Slit()
        self.fish=Fish()
        #No pinta a no ser que estigui activat.
        self.paint.pause_paint=True
        self.box1.add_widget(self.paint,0)
        self.box1.add_widget(self.arrow,0)
        self.box1.add_widget(self.slitcanvas,0)


        ###############################PROPIETATS DEL QUADRAT AMB EL QUE JUGUEM
        self.quadrat.pos=[800,800]
        self.play_game=False
        ####Propietats de la fletxa del moment
        #self.arrow.pos=[800,800]
        #self.arrow.points=[0,0,1,1]
        ###################################BUTONS DE LA CAIXA 2###################
        #Aquests butons varian el menu principal del joc
        
                
        #Aquest activa la funció que activa,atura, o reseteja el calcul
        #self.box2.add_widget(Button(text='Play',on_press=partial(self.compute)))
        #self.box2.add_widget(Button(text='Pause',on_press=partial(self.pause)))
        #self.box2.add_widget(Button(text='Reset',on_press=partial(self.reset)))
        #self.box2.add_widget(Button(text='Standard',on_press=partial(self.standard)))
        #self.box2.add_widget(Button(text='Slit',on_press=partial(self.slit)))
        #self.box2.add_widget(Button(text='Game',on_press=partial(self.gameon)))
        #self.box2.add_widget(Button(text='Back',on_press=partial(self.transition_PS)))
        
        

        #####################################PAUSESTATE


    ##Ara venen tot el seguït de funcions que utilitzarem conjuntament amb els 
    #parametres definits a g_init
        self.slitwidth0=1.5
        self.remove_widget(self.box5)
        self.activatepaint(1)
        self.activatepaint(0)
    ##############################RESIZING############################
    #1.Resize_kivy
    
    #Funcions que s'ocupen del tema resize:
        
    def resize_kivy(self,*largs):
        """Aquesta funció és trucada cada cop que la finestra canvia de tamany.
        Cada cop que això passa, les coordenades (en pixels) dels cantons del plot 
        imshow canvien de lloc. Aquests són molt importants sobretot pel joc, per
        tant, cal tenir-nos ben localitzats."""
        #Aquesta funció de matplotlib ens transforma els cantons de coordenades Data
        # a display (pixels)
        self.cantonsnew=self.visu.transData.transform([(-self.L,-self.L),
                                                       (-self.L,self.L),
                                                       (self.L,self.L),
                                                       (self.L,-self.L)])
      
        self.ara=0
        
                

        
    ################################FUNCIONS DE CALCUL/FLUX##################
    #1.g_schedule_fired
    #2.plotpsiev
    #3.compute(play)
    #4.pause
    #5.reset
    #6.transition_PS

        
        print(self.cantonsnew)
    def p_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotpsiev,1/60.)
    
    def p_schedule_cancel(self):
        self.schedule.cancel()

           
    
    def plotpsiev(self,dt):
        "Update function that updates psi plot"
        if self.ara<1:
            self.ara=self.ara+1
            self.cantonsnew=self.visu.transData.transform([(-self.L,-self.L),
                                                       (-self.L,self.L),
                                                       (self.L,self.L),
                                                       (self.L,-self.L)])
            if self.setcordpress.size>0:
                           
                self.paint.canvas.clear()
                vecpress=self.setcordpress
                vecrelease=self.setcordrelease
                vecsize=np.int(vecpress.size)
    
                #Coordenades dels nous cantons
                for i in range(0,vecsize,2):
                    x0=vecpress[i]
                    y0=vecpress[i+1]
                    
                    x1=vecrelease[i]
                    y1=vecrelease[i+1]
                    
                    newcoord=self.visu.transData.transform([(x0,y0),(x1,y1)])
               
                    with self.paint.canvas:
                        Color(1, 1, 1)
                        d=5.
                        Ellipse(pos=(newcoord[0,0] - d / 2, newcoord[0,1] - d / 2), 
                                size=(d,d))
                        
                        Ellipse(pos=(newcoord[1,0] - d / 2, newcoord[1,1] - d / 2), 
                                 size=(d,d))
                        
                        Line(points=(newcoord[0,0],newcoord[0,1],newcoord[1,0],
                                     newcoord[1,1]))
            
            if (np.abs(self.px)>0. or np.abs(self.py)>0.) and self.t_total<self.dt:
                
                self.arrow.canvas.clear()
                if self.mode==1:
                    self.cent=self.visu.transData.transform([(self.x0slit,self.y0slit)])
                else:
                    self.cent=self.visu.transData.transform([(self.x0,self.y0)])
                
                with self.arrow.canvas:
                    Color(self.arrowcolor[0],self.arrowcolor[1],self.arrowcolor[2])
                    x0=self.cent[0,0]
                    y0=self.cent[0,1]
                    x1=x0+self.px*7.5
                    y1=y0+self.py*7.5
                    Line(points=(x0,y0,x1,y1),width=self.arrowwidth)
                    p_max=np.sqrt(10**2 + 10**2)
                    p=np.sqrt(self.px**2+self.py**2)
                    c=p/p_max
                    if self.px==0.0:
                        p1=np.array([x0-self.arrowdist*c,y1])
                        p2=np.array([x0+self.arrowdist*c,y1])
                        p3=np.array([x1,y0+self.py*self.arrowdist2])                    
                    elif self.py==0.0:
                        p1=np.array([x1,y0+self.arrowdist*c])
                        p2=np.array([x1,y0-self.arrowdist*c])
                        p3=np.array([x0+self.px*self.arrowdist2,y1])
                    else:
                        m=(y1-y0)/(x1-x0)
                        phi=np.pi/2.-np.arctan(m)
                        xr=x1-np.cos(phi)*self.arrowdist*c
                        xm=x1+np.cos(phi)*self.arrowdist*c
                        p1=np.array([xr,-np.tan(phi)*(xr-x1)+y1])
                        p2=np.array([xm,-np.tan(phi)*(xm-x1)+y1])
                        p3=np.array([x0 + self.px*self.arrowdist2,y0 + self.py*self.arrowdist2])
                    
                    
                    Line(points=(p1[0],p1[1],p2[0],p2[1]),width=self.arrowwidth)
                    Line(points=(p2[0],p2[1],p3[0],p3[1]),width=self.arrowwidth)
                    Line(points=(p3[0],p3[1],p1[0],p1[1]),width=self.arrowwidth)
            
                        
            if self.mode==1:
                self.slitcanvas.canvas.clear()
                self.xslit=((self.L)*2)*(1/3.)-self.L
                self.yslitinf=-self.slitwidth/2.
                self.yslitsup=self.slitwidth/2.
                self.inf1=self.visu.transData.transform([(self.xslit,-self.L)])
                self.inf2=self.visu.transData.transform([(self.xslit,self.yslitinf)])
                
                self.sup1=self.visu.transData.transform([(self.xslit,self.yslitsup)])
                self.sup2=self.visu.transData.transform([(self.xslit,self.L)])
                
                with self.slitcanvas.canvas:
                    Color(1,1,1)
                    Line(points=(self.inf1[0,0],self.inf1[0,1],self.inf2[0,0],self.inf2[0,1]))
                    Line(points=(self.sup1[0,0],self.sup1[0,1],self.sup2[0,0],self.sup2[0,1]))   
                        
                            
                    
        if self.pause_state==False:
            #Primer de tot, representem el primer frame
            
           # if (self.i)==0:
            #    self.compute_parameters()    
            #Animation
            #Lanimació consisteix en elimanr primer el que teniem abans
            self.normavec=ck.norma(self.psi0,self.Nx)
            self.visu_im.remove()
            #Posar el nou
            self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                          extent=(-self.L,self.L,-self.L,self.L),
                                          cmap='inferno') 
            #I utilitzar-lo
            self.main_canvas.draw()
            #CALCUL
            #Tot següit fem el càlcul del següent step
            self.psi0=ck.Crannk_stepm(self.psi0, self.avec, self.cvec, self.r, 
                                      self.Vvec, self.dt, self.Nx, self.i)
            
            #Canviem el temps
            self.t_total=self.t_total+(self.dt)/2.
            self.box3.tempschange.text='{0:.3f}'.format(self.t_total)
            
            #Videojoc, coses del  videojoc
            if self.play_game==True:
                self.maxvalue=np.amax(self.normavec)
                self.llindar=self.maxvalue/20.

                if self.normavec[self.pos_discret[0],self.pos_discret[1]]>self.llindar:
                    print('Touched')
     
            
            
            #Cada 31 pasos de temps calculem el temps:
            if self.i>2:
                self.box4.statechange.text='Simulating'
                
            self.i+=1


        

    
    def compute(self,*largs):    
        """Quan es clica el butó play, s'activa aquesta funció, que activa
        el calcul"""     
        
        self.pause_state=False
        
        self.arrow.canvas.clear()
        self.box4.boxpx.pxslider.disabled=True
        self.box4.boxpy.pyslider.disabled=True
        self.box4.box_edition.okay_but.disabled=True
        self.box2.reset_but.disabled=True
        self.box4.box_edition.draw_but.disabled=True
        self.box4.box_edition.select_but.disabled=True
        #self.box4.box_sel.disabled=True
        self.box4.statechange.text='Computing...'
        if self.mode==1:   
            self.box5.slit_slider.disabled=True
        
    def pause(self,*largs):
        """Aquesta funció para el càlcul del joc"""
        self.pause_state=True
        self.box4.box_edition.okay_but.disabled=False
        self.box2.reset_but.disabled=False
        self.box4.box_edition.draw_but.disabled=False
        self.box4.statechange.text='Pause'
        if self.mode==1:
            self.box5.slit_slider.disabled=False


        
       
    def reset(self,*largs):
        """Aquesta funció reseteja el paquet i el potencial inicials. Segons el
        mode inicial en el que ens trobem, es resetejara un o l altre."""
        if self.pause_state==True:
            #Segons el mode, reseteja un potencial o un paquet diferent
            if self.mode==0:
                self.standard()

            elif self.mode==1:
                self.slit()
                self.box4.box_edition.select_but.disabled=True
        
            #Parametres que s'han de tornar a resetejar
            self.i=0
            self.px=0.
            self.py=0.
            self.t_total=0.00

            
            if self.setcordpress.size>0:
                self.editorstop()
            else:
                pass

            self.paint.canvas.clear()
            self.arrow.canvas.clear()
            self.setcordpress=self.emptylist()
            self.setcordrelease=self.emptylist()          
            #self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
            print('Clear!')
            #Rseteja la imatge
            
            
            
            self.box3.tempschange.text='0.00'
            self.box3.pxchange.text='0.0'
            self.box3.pychange.text='0.0'
            self.box4.boxpx.pxslider.value=0.
            self.box4.boxpy.pyslider.value=0.
            self.x0=self.zero()
            self.y0=self.zero()
            self.selectmode_pause=False
            self.box4.boxpx.pxslider.disabled=False
            self.box4.boxpy.pyslider.disabled=False
            self.box4.box_edition.okay_but.disabled=False
            self.box2.reset_but.disabled=False
            self.box4.box_edition.draw_but.disabled=False
    
            self.box4.statechange.text='Modify initial parameters'
            
            
            
    def transition_PS(self,*largs):
            self.p_schedule_cancel()
            self.manager.transition=FadeTransition()
            self.manager.current='starting'
            
            
    def zero(self,*largs):
        a=0.
        return a
    
    
            
    ############################## MODES####################################
    #Aquestes funcions s'activen quan clikem self o slit i s'encarreguen de que
    #resetejar tot el que hi ha en pantalla (si estan en un mode difent al seu
    #propi) i posar el mode escollit. També escrivim aquí els potencials i
    #paquets propis de cada mode.
    
    #1.standard(mode 0)
    #2.slit(mode slit)
    #3.modechange (canvia entre un mode o l'altre)   
    #4.Vvecstand
    #5.Vvecslit
    #6.psistand
    #7.psislit
    def standard(self,*largs):
        """Funció que retorna el mode estandard en tots els aspectes: de càlcul,
        en pantalla, etf..."""
        
        if self.mode==1:
            self.box4.box_edition.select_but.disabled=False
            self.remove_widget(self.box5)
            self.slitcanvas.canvas.clear()
        self.box4.box_edition.select_but.disabled=False
        #self.mode=0
        self.L=3.0
        self.dx=0.03
        self.box3.longitudchange.text='{}'.format(self.L)
        self.Nx=np.int((2*self.L)/self.dx)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        self.Vvec=self.Vvecstand()
        self.psi0=self.psistand(0.,0.)
        #Llevem les líneas que hi pogui haver
        
            
        #Això es un reset, pensar fer-ho amb el reset.
        #Canviem la pantalla i el mode:
        self.i=0
        #Rseteja la imatge
            
        self.normavec=ck.norma(self.psi0,self.Nx)
        self.visu_im.remove()
        #Posar el nou
        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                         extent=(-self.L,self.L,-self.L,self.L),
                                         cmap='inferno') 
        

        #I utilitzar-lo
        self.main_canvas.draw()            
  
            
                    
 
    def slit(self,*largs):
        """Quan activem aquest buto, canviem el paquet de referencia i el 
        potencial de referència i ho deixem en el mode slit. Aquest mode
        correspon amb el número 1."""
        #self.mode=1
        self.L=3.0
        self.dx=0.03
        self.box3.longitudchange.text='{}'.format(self.L)
        self.Nx=np.int((2*self.L)/self.dx)
        self.slitwidth=self.slitwidth0
        self.xslit=((self.L)*2)*(1/3.)-self.L
        self.yslitinf=-self.slitwidth/2.
        self.yslitsup=self.slitwidth/2.
        self.value=10000
        self.s2=self.dx/2.
        self.Vvec=self.Vvecstand()

        self.Vvec+=self.potencial_final(self.xslit,-self.L,self.xslit,self.yslitinf,
                                        self.value,self.s2)
        self.Vvec+=self.potencial_final(self.xslit,self.yslitsup,self.xslit,self.L,
                                        self.value,self.s2)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        self.psi0=self.psistand(self.x0slit,self.y0slit)
        #self.box4.box_sel.select_but.disabled=True
        
        #Parametres del forat del slit

        
        """Pintem tot lo dit i afegim una líniea que es correspón amb el potencial,
        fix en aquest mode."""
        
    #Rseteja la imatge
            
        self.normavec=ck.norma(self.psi0,self.Nx)
        self.visu_im.remove()
        #Posar el nou
        self.visu_im=self.visu.imshow(self.normavec,origin={'upper'},
                                         extent=(-self.L,self.L,-self.L,self.L),
                                         cmap='inferno') 
 
        
        
        self.inf1=self.visu.transData.transform([(self.xslit,-self.L)])
        self.inf2=self.visu.transData.transform([(self.xslit,self.yslitinf)])
        
        self.sup1=self.visu.transData.transform([(self.xslit,self.yslitsup)])
        self.sup2=self.visu.transData.transform([(self.xslit,self.L)])
        
        self.slitcanvas.canvas.clear()
        with self.slitcanvas.canvas:
            Color(1,1,1)
            Line(points=(self.inf1[0,0],self.inf1[0,1],self.inf2[0,0],self.inf2[0,1]))
            Line(points=(self.sup1[0,0],self.sup1[0,1],self.sup2[0,0],self.sup2[0,1]))               
        #Afegim un slider
        
        if self.mode==0:
            self.add_widget(self.box5)
            self.box4.box_edition.select_but.disabled=True
            
            
        self.box5.slit_slider.max=self.L*2
        self.box5.slit_slider.min=0
        self.box5.slit_slider.value=self.slitwidth0
        

        #I utilitzar-lo
        self.main_canvas.draw()    
        
    def changedrawslit(self,value):
        
        
        self.slitwidth=value
        self.xslit=((self.L)*2)*(1/3.)-self.L
        self.yslitinf=-self.slitwidth/2.
        self.yslitsup=self.slitwidth/2.
        self.slitcanvas.canvas.clear()
        
        self.inf1=self.visu.transData.transform([(self.xslit,-self.L)])
        self.inf2=self.visu.transData.transform([(self.xslit,self.yslitinf)])
        
        self.sup1=self.visu.transData.transform([(self.xslit,self.yslitsup)])
        self.sup2=self.visu.transData.transform([(self.xslit,self.L)])
        
        self.slitcanvas.canvas.clear()
        with self.slitcanvas.canvas:
            Color(1,1,1)
            Line(points=(self.inf1[0,0],self.inf1[0,1],self.inf2[0,0],self.inf2[0,1]))
            Line(points=(self.sup1[0,0],self.sup1[0,1],self.sup2[0,0],self.sup2[0,1])) 
        
    def changeslit(self,value):
        
        
        self.slitwidth=value
        self.xslit=((self.L)*2)*(1/3.)-self.L
        self.yslitinf=-self.slitwidth/2.
        self.yslitsup=self.slitwidth/2.
        
        self.inf1=self.visu.transData.transform([(self.xslit,-self.L)])
        self.inf2=self.visu.transData.transform([(self.xslit,self.yslitinf)])
        
        self.sup1=self.visu.transData.transform([(self.xslit,self.yslitsup)])
        self.sup2=self.visu.transData.transform([(self.xslit,self.L)])
        
        self.slitcanvas.canvas.clear()
        with self.slitcanvas.canvas:
            Color(1,1,1)
            Line(points=(self.inf1[0,0],self.inf1[0,1],self.inf2[0,0],self.inf2[0,1]))
            Line(points=(self.sup1[0,0],self.sup1[0,1],self.sup2[0,0],self.sup2[0,1])) 
        
        self.Vvec=self.Vvecstand()

        self.Vvec+=self.potencial_final(self.xslit,-self.L,self.xslit,self.yslitinf,
                                        self.value,self.s2)
        self.Vvec+=self.potencial_final(self.xslit,self.yslitsup,self.xslit,self.L,
                                        self.value,self.s2)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        
        self.box4.statechange.text='Potential changed' + '\n'+ 'Do not forget to apply changes'
    
        
    def modechange(self,*largs):
        """Funció que s'encarrega de canviar de mode. De moment només en tenim
        2. Possible futura ampliació."""
        #primer de tot apliquem un reset
        
        self.reset()
        #Canviem el mode posteriorment.
        if self.mode==1:
            self.standard()
            self.mode=0
            self.box4.changemode.text='[color=F027E8]Standard mode[/color]'
            
        else:
            self.slit()
            self.box4.changemode.text='[color=F027E8]Slit mode[/color]'
            self.mode=1
        
        
 
    def Vvecstand(self,*largs):
        """Potencial del mode estandard"""
        Vvecestandard=np.array([[ck.Vharm(self.xa+i*self.dx,self.xa+j*self.dx,
                                      self.xb,self.xb) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)],dtype=np.float64)
        
        return Vvecestandard
    
    def Vvecslit(self,*largs):
        """Potencial del mode slit"""
        #Potencial slit(mode=1)
        self.yposslitd=np.int(self.Nx*(4.75/10))
        self.yposslitu=np.int(self.Nx*(5.25/10))
        self.xposslit=np.int(self.Nx/3)
        
        Vvecslit=self.Vvecstand()
        
        Vvecslit[0:self.yposslitd,self.xposslit]=100000
        
        Vvecslit[self.yposslitu:self.Nx,self.xposslit]=100000
        
        return Vvecslit
    
    def psistand(self,x0,y0,*largs):
        """Paquet corresponent al mode standard"""
        
        psiestandar=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,x0,y0) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])
        
        return psiestandar
        
    def psislit(self,*largs):
        """Paquet corresponent al mode slit"""
        
        #Paquet propi del mode slit
        psislit=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])
        
        return psislit
    
    def emptylist(self,*largs):
        """Returns an empty list"""
        
        empty=np.array([])
        return empty
    #####################################CHANGE PARAMETERs##################
    #Fucnions que s'encarregar del canvi efectiu dels parametres
    #1.changepx
    #2.cahngepy
    #3.applychange
    def changepx(self,value_px,*largs):
        """Funció que respon als buto + i - que varia el moment inicial del paquet.
        Per que aquest canvi sigui efectiu, hem de canviar tambe el propi paquet."""
        #if selfpause==True
        #Canvi el moment en pantalla
        self.px=value_px
        
        self.box3.pxchange.text='{0:.1f}'.format(self.px)
        
        #if self.mode==0:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
        #                             for i in range(self.Nx+1)]
        #                       for j in range(self.Nx+1)])
        #if self.mode==1:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
        #                         for i in range(self.Nx+1)]
        #                   for j in range(self.Nx+1)])
    def changedrawpx(self,value_px,*largs):
        
        if self.mode==1:
            self.cent=self.visu.transData.transform([(self.x0slit,self.y0slit)])
        else:
            
            self.cent=self.visu.transData.transform([(self.x0,self.y0)])
        self.px=value_px
        
        if self.t_total<self.dt:
            
            self.arrow.canvas.clear()
            with self.arrow.canvas:
                Color(self.arrowcolor[0],self.arrowcolor[1],self.arrowcolor[2])
                x0=self.cent[0,0]
                y0=self.cent[0,1]
                x1=x0+self.px*7.5
                y1=y0+self.py*7.5
                Line(points=(x0,y0,x1,y1),width=self.arrowwidth)
                p_max=np.sqrt(10**2 + 10**2)
                p=np.sqrt(self.px**2+self.py**2)
                c=p/p_max
                if self.px==0.0:
                    p1=np.array([x0-self.arrowdist*c,y1])
                    p2=np.array([x0+self.arrowdist*c,y1])
                    p3=np.array([x1,y0+self.py*self.arrowdist2])                    
                elif self.py==0.0:
                    p1=np.array([x1,y0+self.arrowdist*c])
                    p2=np.array([x1,y0-self.arrowdist*c])
                    p3=np.array([x0+self.px*self.arrowdist2,y1])
                else:
                    m=(y1-y0)/(x1-x0)
                    phi=np.pi/2.-np.arctan(m)
                    xr=x1-np.cos(phi)*self.arrowdist*c
                    xm=x1+np.cos(phi)*self.arrowdist*c
                    p1=np.array([xr,-np.tan(phi)*(xr-x1)+y1])
                    p2=np.array([xm,-np.tan(phi)*(xm-x1)+y1])
                    p3=np.array([x0 + self.px*self.arrowdist2,y0 + self.py*self.arrowdist2])
                
                
                Line(points=(p1[0],p1[1],p2[0],p2[1]),width=self.arrowwidth)
                Line(points=(p2[0],p2[1],p3[0],p3[1]),width=self.arrowwidth)
                Line(points=(p3[0],p3[1],p1[0],p1[1]),width=self.arrowwidth)
    
    def changedrawpy(self,value_py,*largs):
        
        if self.mode==1:
            self.cent=self.visu.transData.transform([(self.x0slit,self.y0slit)])
        else:
            
            self.cent=self.visu.transData.transform([(self.x0,self.y0)])
        self.py=value_py
        
        if self.t_total<self.dt:
            
            self.arrow.canvas.clear()
            with self.arrow.canvas:
                Color(self.arrowcolor[0],self.arrowcolor[1],self.arrowcolor[2])
                x0=self.cent[0,0]
                y0=self.cent[0,1]
                x1=x0+self.px*7.5
                y1=y0+self.py*7.5
                Line(points=(x0,y0,x1,y1),width=self.arrowwidth)
                p_max=np.sqrt(10**2 + 10**2)
                p=np.sqrt(self.px**2+self.py**2)
                c=p/p_max
                if self.px==0.0:
                    p1=np.array([x0-self.arrowdist*c,y1])
                    p2=np.array([x0+self.arrowdist*c,y1])
                    p3=np.array([x1,y0+self.py*self.arrowdist2])                    
                elif self.py==0.0:
                    p1=np.array([x1,y0+self.arrowdist*c])
                    p2=np.array([x1,y0-self.arrowdist*c])
                    p3=np.array([x0+self.px*self.arrowdist2,y1])
                else:
                    m=(y1-y0)/(x1-x0)
                    phi=np.pi/2.-np.arctan(m)
                    xr=x1-np.cos(phi)*self.arrowdist*c
                    xm=x1+np.cos(phi)*self.arrowdist*c
                    p1=np.array([xr,-np.tan(phi)*(xr-x1)+y1])
                    p2=np.array([xm,-np.tan(phi)*(xm-x1)+y1])
                    p3=np.array([x0 + self.px*self.arrowdist2,y0 + self.py*self.arrowdist2])
                
                
                Line(points=(p1[0],p1[1],p2[0],p2[1]),width=self.arrowwidth)
                Line(points=(p2[0],p2[1],p3[0],p3[1]),width=self.arrowwidth)
                Line(points=(p3[0],p3[1],p1[0],p1[1]),width=self.arrowwidth)
        
        
    def changepy(self,value_py,*largs):
        """Funció que respon als buto + i - que varia el moment inicial del paquet.
        Per que aquest canvi sigui efectiu, hem de canviar tambe el propi paquet."""
        #if selfpause==True
        #Canvi el moment en pantalla
        self.py=value_py
        
        self.box3.pychange.text='{0:.1f}'.format(self.py)
       
        #if self.mode==0:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
        #                             for i in range(self.Nx+1)]
        #                       for j in range(self.Nx+1)])
        #if self.mode==1:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
        #                         for i in range(self.Nx+1)]
        #                   for j in range(self.Nx+1)]) 
        
    def changemomentstate(self,*largs):
        self.box4.statechange.text='Moment changed!'+'\n'+'Do not forget to apply changes'
        #Canvi del moment efectiu
        
    def potentialstatechange(self,*largs):
        self.box4.statechange.text='You can draw an'+'\n'+'infinite potential now'
        
    
    def applychanges(self,*largs):
        """Aquesta funció s'encarrega d'aplicar les modificacions físiques
        fetes al paquet mitjançant el menú d'edició. """
        
        #Canvi del moment efectiu
        
        if self.t_total<self.dt:
            
            if self.mode==0:
                self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0,self.y0) 
                                         for i in range(self.Nx+1)]
                                   for j in range(self.Nx+1)])
            if self.mode==1:
                self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
                                     for i in range(self.Nx+1)]
                               for j in range(self.Nx+1)]) 
        
        if self.mode==1:
            if self.t_total<self.dt:
                if self.slitwidth==self.slitwidth0:
                    pass
                else:
                    self.changeslit(self.slitwidth)
        


        self.box4.statechange.text='Changes applied!'+'\n'+'Start the simulation'
   
    ################################# POTENCIAL MODE #########################
    #Aquestes funcions juntament amb el widget MyPaintWidget porten la posibilitat
    #de poder dibuixar un potencial.
    #1.editorfun(start)
    #2.editorstop(stop)
    #3.Activatepaint(pinta només si self.paint.pause_paint==False)
    #4.Potpress(apunta les coordenades on es pren el potencial)
    #5.Potrelease(apunta les coordenades on es solta el potencial)
    #6.modifypot(Agafa aquests dos punts i els ajunta, tot creaunt un potencial 
    # infinit al mig.
    #7.clear (reseteja el potencial que s'hagui dibuixat, amb tot el que implica)
       
    def editorfun(self,*largs):
        """Aquesta funció es l'encarregada d'activar el mode editor. Activa
        l'event de matplotlib que detecta la entrada al propi plot o la sortida,
        i l'enllaça amb la funció activatepaint"""
        
                                                       
        #Controla que només es pogui dibuixar dins la figura
        self.cidactivate=self.main_canvas.mpl_connect('axes_enter_event',
                                               partial(self.activatepaint,1))
        self.ciddesactivate=self.main_canvas.mpl_connect('axes_leave_event',
                                               partial(self.activatepaint,0))
        
   
        
        
        
  
    def editorstop(self,*largs):
        """Aquesta funció desactiva totes les conexions activades a editorfun,
        i per tant, desactiva el mode editor"""
        self.main_canvas.mpl_disconnect(self.cidactivate)   
        self.main_canvas.mpl_disconnect(self.ciddesactivate)
        self.main_canvas.mpl_disconnect(self.cid1)   
        self.main_canvas.mpl_disconnect(self.cid2)
        self.main_canvas.mpl_disconnect(self.cid3)
        self.main_canvas.mpl_disconnect(self.cid4)
        self.paint.pause_paint=True
        
    def activatepaint(self,n,*largs):
        """Aquesta funció s'encarrega d'activar el widget Paint(que dona la
        capactiat de pintar en pantalla, només quan el cursor està dins del
        plot). També activa quatre funcions i les conecta amb les accions 
        d'apretar i soltar el click dret del mouse per, d'aquesta manera,
        enregistar on s'ha apretat."""
        
        if n==1:

            self.paint.pause_paint=False
            self.cid1=self.main_canvas.mpl_connect('button_press_event',press)
            self.cid2=self.main_canvas.mpl_connect('button_press_event',
                                               partial(self.potpress))
            self.cid3=self.main_canvas.mpl_connect('button_release_event',release)
            self.cid4=self.main_canvas.mpl_connect('button_release_event',
                                               partial(self.potrelease))

            self.box4.statechange.text='You can draw'
            print('painting')
            
        else:
            self.paint.pause_paint=True
            self.main_canvas.mpl_disconnect(self.cid1)   
            self.main_canvas.mpl_disconnect(self.cid2)
            self.main_canvas.mpl_disconnect(self.cid3)
            self.main_canvas.mpl_disconnect(self.cid4)
            print('no painting')
            
            
    def potpress(self,*largs):
        """Funció que s'encarrega de guardar en coordenades de data el lloc
        on s'ha apretat al plot"""
        self.setcordpress=np.append(self.setcordpress,cordpress)
        self.box4.statechange.text='Potential drawn'+'\n'+'Do not forget to apply the changes'

        print(self.setcordpress)
    
    def potrelease(self,*largs):
        """Funció que s'encarrega de guardar en coordenades de data el lloc
        on s'ha soltat al plot."""
        self.setcordrelease=np.append(self.setcordrelease,cordrelease)
        print(self.setcordrelease)
        self.box4.statechange.text='Potential drawn'+'\n'+'Do not forget to apply the changes'
        self.editorstop()
        
        #self.modifypot()
        
 
    
    def modifypot(self,*largs):
        """Aquesta funció s'aplica quan es clika el buto apply. Durant el
        període que s'ha dibuixat, totes les lineas de potencial s'han en-
        registrat a les variables self.csetcorpress i setcordreleas. Aquestes
        estan en coordenades de self. data i les hem de passar al discretitzat."""
        
        #Variables on guardem les dates del dibuix
        vecpress=self.setcordpress
        vecrelease=self.setcordrelease
        print(vecpress,vecrelease)
        #linias dibuixades=vecsize/2
        vecsize=np.int(vecpress.size)
        print(vecsize)
        #Aquí guardarem els resultats.
        #Coloquem una guasiana al voltant del potencial que pintem. La sigma^2 farem que
        #sigui del mateix tamany que el pas
        s2vec=self.dx/2.
        valorVec=10000
        

        for i in range(0,vecsize,2):
            #Establim noms de variables
            index=i
            x0=vecpress[index]
            y0=vecpress[index+1]
            x1=vecrelease[index]
            y1=vecrelease[index+1]
            
            Vecgaussia=self.potential_maker(x0,y0,x1,y1,valorVec,s2vec)
            Vvecsegur=self.Vvecmarcat(x0,y0,x1,y1,valorVec*(1./(np.sqrt(s2vec*2.*np.pi))))
                #I el sumem...
            self.Vvec+=Vvecsegur+Vecgaussia
                #Coloquem una petita gaussiana al voltant de cada punt de sigma*2=self.dx/2
                #self.Vvec=Vvec
            #Modifiquem els respectius ac,vc
            #self.Vvec=Vvec

        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
            
        print('Applied!')
        
        if vecsize>0:
            self.editorstop()
        
        else:
            pass
        
    def potencial_final(self,x0,y0,x1,y1,value,s2):
        """Funció que genera un potencial infinit que va desde x0,y0 fins x1,y1 """
        
        
        Vecgaussia=self.potential_maker(x0,y0,x1,y1,value,s2)
        Vvecsegur=self.Vvecmarcat(x0,y0,x1,y1,value*(1./(np.sqrt(s2*2.*np.pi))))
                #I el sumem...
        Vvecfinal=Vvecsegur+Vecgaussia
        #Coloquem una petita gaussiana al voltant de cada punt de sigma*2=self.dx/2
        #self.Vvec=Vvec
        #Modifiquem els respectius ac,vc
        #self.Vvec=Vvec
        return Vvecfinal

    def gaussian_maker(self,x,y,x0,y0,x1,y1,value,s2,*largs):
        """Introduïm aquí la gaussiana per cada x e y... x0,y0,x1,y1, son els
        dos punts que uneixen la recta dibuixada. Utilitzarem les rectes perpen-
        diculars en aquestes a cada punt per dissenyar aquest potencial."""
        #redefinim x0,y0,x1,y1 a punts que siguin exactament part del discretitzat
        i0=np.int((x0+self.L)/self.dx)
        i1=np.int((x1+self.L)/self.dx)
        j0=np.int((y0+self.L)/self.dx)
        j1=np.int((y1+self.L)/self.dx)
        
        x0=-self.L+self.dx*np.float(i0)
        x1=-self.L+self.dx*np.float(i1)
        y0=-self.L+self.dx*np.float(j0)
        y1=-self.L+self.dx*np.float(j1)
        #Primer de tot, definim el pendent de la recta que uneix els dos punts
        if np.abs(x1-x0)<self.dx:
            if (y0<y<y1 or y1<y<y0):                
                x_c=x0
                y_c=y
                Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2**2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
            else:
                Vvecgauss=0.
                
        
        elif np.abs(y1-y0)<self.dx:
            if (x0<x<x1 or x1<x<x0):                
                x_c=x
                y_c=y0
                Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2**2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
            else:
                Vvecgauss=0.
        else:
            
            m=(y1-y0)/(x1-x0)                      
            x_c=0.5*(x0+x+(y-y0)/m)
            y_c=m*(x_c-x0)+y0
        
            if (x0<x_c<x1 or x1<x_c<x0) or (y0<y_c<y1 or y1<y_c<y0):
                Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2**2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
      
            else:
                Vvecgauss=0.
            
            if np.abs(y-self.dx)<np.abs((m)*(x-x0)+y0)<np.abs(y+self.dx):
                Vvecgauss=0.
 
            else:
                pass
                
        
        
                
                
        return Vvecgauss
    
    def potential_maker(self,x0,y0,x1,y1,value,s2,*largs):
        Vvecgauss1=np.array([[self.gaussian_maker(self.xa+i*self.dx,self.xa+j*self.dx,
                                      x0,y0,x1,y1,value,s2) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)])
        
        return Vvecgauss1
    
    def Vvecmarcat(self,x0,y0,x1,y1,value,*largs):
        
        Vvecmarcat=self.Vvecstand()
        
        if np.abs(x0-x1)<np.abs(y0-y1):
            num=np.int(np.abs(y0-y1)/self.dx)
        else:
            num=np.int(np.abs(x0-x1)/self.dx)
            
        xvec=np.linspace(x0,x1,num)
        yvec=np.linspace(y0,y1,num)

        for i in range(num):
            Vvecmarcat[np.int((+self.L+yvec[i])/self.dx),
                       np.int((+self.L+xvec[i])/self.dx)]=value
            
        return Vvecmarcat
            
            
    def changetitleslit(self,*largs):
        self.box4.statechange.text='Do not forget to apply changes!'
        
    def clear(self,*largs):
        """Aquesta funció neteja el canvas i tot els canvis introduïts 
        tan al potencial com a les coordenades del potencial. """
        self.paint.canvas.clear()
        self.setcordpress=self.emptylist()
        self.setcordrelease=self.emptylist()

        if self.mode==0:
            
            self.Vvec=self.Vvecstand()

        if self.mode==1:
            self.Vvec=self.Vvecslit()
            
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        print('Clear!')
        
    #################################### SELECT POSITION ##################
    def editorfuns(self,*largs):
        """Aquesta funció es l'encarregada d'activar el mode editor. Activa
        l'event de matplotlib que detecta la entrada al propi plot o la sortida,
        i l'enllaça amb la funció activatepaint"""
        
        if self.mode==0:
            #Controla que només es pogui dibuixar dins la figura
            self.cidactivates=self.main_canvas.mpl_connect('axes_enter_event',
                                                   partial(self.activatepaints,1))
            self.ciddesactivates=self.main_canvas.mpl_connect('axes_leave_event',
                                                   partial(self.activatepaints,0))
            
        self.box4.statechange.text='Select an initial position'
        
        
        
  
    def editorstops(self,*largs):
        """Aquesta funció desactiva totes les conexions activades a editorfun,
        i per tant, desactiva el mode editor"""
        self.main_canvas.mpl_disconnect(self.cidactivates)   
        self.main_canvas.mpl_disconnect(self.ciddesactivates)
        self.main_canvas.mpl_disconnect(self.cid1s)   
        self.main_canvas.mpl_disconnect(self.cid2s)
        self.box4.statechange.text='Do not forget to apply changes'

        print('stop editor')
        
    def selecttitle(self,*largs):
        self.box4.statechange.text='Select an initial position'
        
    def activatepaints(self,n,*largs):
        """Aquesta funció s'encarrega d'activar el widget Paint(que dona la
        capactiat de pintar en pantalla, només quan el cursor està dins del
        plot). També activa quatre funcions i les conecta amb les accions 
        d'apretar i soltar el click dret del mouse per, d'aquesta manera,
        enregistar on s'ha apretat."""
        
        if n==1:

            self.cid1s=self.main_canvas.mpl_connect('button_press_event',press_sel)
            self.cid2s=self.main_canvas.mpl_connect('button_press_event',
                                               partial(self.selpress))
            

            self.box4.statechange.text='Select an initial position'
    
            print('selecting')
            
        else:

            self.main_canvas.mpl_disconnect(self.cid1s)   
            self.main_canvas.mpl_disconnect(self.cid2s)
            print('no selecting')
            
            
    def selpress(self,*largs):
        """Funció que s'encarrega de guardar en coordenades de data el lloc
        on s'ha apretat al plot"""
        self.x0=cordpress_sel[0]
        self.y0=cordpress_sel[1]
        print(self.x0,self.y0)
            
        if self.t_total<self.dt and self.mode==0:
            self.changed_position=True
            self.psi0=self.psistand(self.x0,self.y0)
            #Rseteja la imatge
            
            self.normavec=ck.norma(self.psi0,self.Nx)
            self.visu_im.remove()
            self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                          extent=(-self.L,self.L,-self.L,self.L),
                                          cmap='inferno') 
          
            self.main_canvas.draw()
            self.activatepaints(0)
            self.editorstops()
            self.arrow.canvas.clear()
            self.cent=self.visu.transData.transform([(self.x0,self.y0)])

            self.box4.boxpx.pxslider.disabled=False
            self.box4.boxpy.pyslider.disabled=False
            self.box4.box_edition.okay_but.disabled=False
            self.box2.reset_but.disabled=False
            self.box4.box_edition.draw_but.disabled=False
            
                
            with self.arrow.canvas:
                Color(1,0,0)
                Line(points=(self.cent[0,0],self.cent[0,1],self.cent[0,0]+self.px*7.5,
                                         self.cent[0,1]+self.py*7.5))
            
        #parem la selecció
 
    def noselect(self,*largs):
        
        self.box4.boxpx.pxslider.disabled=True
        self.box4.boxpy.pyslider.disabled=True
        self.box4.box_edition.okay_but.disabled=True
        self.box2.reset_but.disabled=True
        self.box4.box_edition.draw_but.disabled=True
        
        #generem el nou paquet
    ################################### COMPUTE PARAMETES#####################
    #Aquestes funcions efectuan càlculs de diferents parametres. De moment, 
    # només de l'energia i la norma
    #1.compute_parameters
    
    def compute_parameters(self,*largs):
        """Aquesta funció s'encarrega de calcular la norma i la energia del
        paquet en un cert instant t i despres l'ensenya a pantalla."""
        
        #Calculem moments i energia inicials i els posem al lloc corresponent
        self.normavec=ck.norma(self.psi0,self.Nx)
        
        self.norma0=ck.trapezis(self.xa,self.xb,self.dx,np.copy(self.normavec))
        self.energy0=ck.Ham(self.xa,self.xb,self.dx,np.copy(self.psi0))
        self.box3.normachange.text='Norma={0:.3f}'.format(self.norma0)
        self.box3.energychange.text='Energia={0:.3f}'.format(self.energy0)
        
    ##################################### JOC ##########################
    #Funcions relacionades amb el propi joc
    #1._keyboard_closed
    #2._on_keyboard_down
    #3.on_position
    #4.on_game
    
    def _keyboard_closed(self):
        """No se ben bé que fa però es necessària"""
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)

        self._keyboard = None
    



    def _on_keyboard_down(self, keyboard, keycode, text, modifiers,*largs):
        """Aquesta funció s'ocupa de conectar el moviment del quadrat amb 
        el teclat. És a dir, conecta l'esdeveniment de teclejar  amb el de
        fer que el quadrat avanci un pas. A l'hora, també imposa els limits on
        el quadrat es pot moure.
        """
        #Definim els limits superior e inferior
        xliminf=np.int(self.cantonsnew[0,0])
        xlimsup=np.int(self.cantonsnew[2,0])
        yliminf=np.int(self.cantonsnew[0,1])
        ylimsup=np.int(self.cantonsnew[1,1])
        
        #defim el pas
        pas=10
        #definim a quina distància s'ha de parar el rectangle.
        w=self.quadrat.width
        h=self.quadrat.height
        #Hem de pensar que la posició es defineix a la  cantonada inferior esquerre
        #del rectangle
        dist=5
        
        #Conexions events amb moviment quadrat
        if keycode[1] == 'w':
            if self.quadrat.y+dist+h>ylimsup:
                pass
            else:
                self.quadrat.y += pas
                self.position=self.quadrat.pos
            return True

        if keycode[1] == 's':
            if self.quadrat.y-dist<yliminf:
                pass
            else:
                self.quadrat.y-= pas
                self.position=self.quadrat.pos
            #self.position=self.quadrat.position
            #print(self.quadrat.position)  
            return True


        if keycode[1]=='d':     
            if self.quadrat.x+dist+w>xlimsup:
                pass
            else:
                self.quadrat.x+=pas
                self.position=self.quadrat.pos
            return True
 
            
               
        elif keycode[1] =='a' :
            if self.quadrat.x-dist<xliminf:
                pass
            else:
                self.quadrat.x -=pas
                self.position=self.quadrat.pos
            return True
   

        
    def on_position(self,quadrat,pos):
        """Cada cop que el quadrat canvia de posició, això es notifica aquí.
        Utilitzarem aquesta funció (que està anclada a una propietat) per fer
        el canvi de coordenades pixels matplotlib-data matplotlib.
        Tot seguït, sabent on es troba el paquet, podem saber a sobre de quin 
        valor de densitat del paquet es troba i definir que passa quan es troba
        en segons quines situacions..."""

                
        #Utilitzem la transformació inversa que ens porta de pixels matplotlib 
        # a dades matplotlib.
        inv_data=self.visu.transData.inverted()
        
        #Posició del centre del quadrat
        pos_data=inv_data.transform((pos[0]+self.quadrat.width/2,
                                     pos[1]+self.quadrat.height/2))
        
        #Un cop tenim la posició del centre del quadrat en coordenades data, les
        #pasem a coord del discretitzat:
        self.pos_discret=np.array([(pos_data[0]+self.L)/self.dx,
                              (pos_data[1]+self.L)/self.dx],dtype=int)


        #Busquem el valor màxim de normavec cada cop que es mou el paquet
        #definim un llindar a partir del qual el quadrat detecta el paquet
        self.maxvalue=np.amax(self.normavec)
        self.llindar=self.maxvalue/20.
        
        if self.normavec[self.pos_discret[0],self.pos_discret[1]]>self.llindar:
            print('Touched')
        
    def gameon(self,*largs):
        """Aquesta funció es l'encarregada de posar el joc en marxa. Col·loca
        el quadradet a algún lloc on es pogui veure (una mica cutre però 
        ja ho canviaras). Uneix la funció teclat amb el moviment del quadrat,
        i varia el mode del joc."""

        if self.play_game==False:
            
            self.play_game=True
            self.quadrat.x=np.int((-self.cantonsnew[0,0]+self.cantonsnew[2,0])/5.
                                  +self.cantonsnew[0,0])
            self.quadrat.y=np.int(3*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/4.
                                  +self.cantonsnew[0,1])
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
            
        else:
            self.play_game=False
            self.quadrat.x=800
            self.quadrat.y=600
            
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)
            
            
class GameScreen(Screen):
    "Això és la part més important de la app. Introduïrem la pantalla inicial"
    
    #Les propietats s'han de definir a nivell de classe (millor). Per tant, 
    #les definim aquí.
    quadrat=ObjectProperty(None)
    position=ListProperty(None)
    arrow=ObjectProperty(None)
    pxslider=ObjectProperty(None)
    pyslider=ObjectProperty(None)
    fish=ObjectProperty(None)
    buttong=ObjectProperty(None)
    door=ObjectProperty(None)
    

    def __init__(self,**kwargs):
        "Unim la clase amb la paraula self"
        super(GameScreen,self).__init__(**kwargs)
        #Aquesta linia uneix keyboard amb request keyboard
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)

        #self._keyboard.bind(on_key_down=self._on_keyboard_down)
        

        
    def gpseudo_init(self):

        "Iniciem el primer dibuix de l'aplicació"
        
        #Variables globals...
        global Vvec,avec,cvec,psi0,normavec,mode,Vvecstandard
        global avec,cvec
        
        ################### PARAMETRES DE LA CAIXA,DISCRETITZAT,PARTICULA########
        self.L=3.0
        self.xa=-self.L
        self.xb=self.L
        
        #Discretitzat
        self.tb=0.1
        self.ta=0
        self.dx=0.03
        self.dt=0.02
        self.Nx=np.int((2*self.L)/self.dx)

        
        #particula i constants físiques
        self.m=1
        self.hbar=1
        self.t_total=0.00
        self.x0=0.
        self.y0=0.
        self.px=0
        self.py=0
        
        #Comptador de moments
        self.i=0
        self.cont=0
        #parametre del discretitzat
        self.r=(self.dt/(4*(self.dx**2)))*(self.hbar/self.m)
        
        

        #Canvia el text del interior de la caixa 'parameters' amb els parametr
        #es inicials escollits
        #self.box3.dxchange.text="{}".format(self.dx)
        self.box3.longitudchange.text="{}".format(self.L)
        #self.box3.dtchange.text="{}".format(self.dt)
        
        print(self.size)

        #################### PARAMETRES NECESSARIS PER EFECTUAR ELS CALCULS######
        
        
        #################### MODES
        #mode: 0, estandard
        #      1,slit
        #      2,disparo(encara no està fet)
        
        #Posició slit

        self.x0slit=self.L/2
        self.y0slit=0.

        
        ################### CONFIGURACIÓ INICIAL
            
        self.Vvec=self.Vvecstand()
        self.psi0=self.psistand(0.,0.)
        #Ens indica en quin mode estem
        self.mode=0
        
        #Per últim, construïm les matrius molt importants per efectuar els càlculs.
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        
        self.pause_state=True
        
        
        self.arrowcolor=np.array([231/255.,30/255.,30/255.])
        self.arrowwidth=2.
        self.arrowdist=15.
        self.arrowdist2=8.5
        
        ##########################PRIMER CÀLCUL ENERGIA#####
                       
        #Calculem la matriu densitat inicial
        self.normavec=ck.norma(self.psi0,self.Nx)
    
        #Calculem moments i energia inicials i els posem al lloc corresponent
        self.norma0=ck.trapezis(self.xa,self.xb,self.dx,np.copy(self.normavec))
        self.energy0=ck.Ham(self.xa,self.xb,self.dx,np.copy(self.psi0))
        self.box3.normachange.text='{0:.3f}'.format(self.norma0)
        self.box3.energychange.text='{0:.3f}'.format(self.energy0)
        
        
        ##########################PRIMER DIBUIX###################
        
        #creem la figura recipient del plot imshow
        self.main_fig=plt.figure()
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas,1)
        #Plot principal
        self.visu=plt.axes(label='princiapl')
        self.main_fig.set_facecolor(color='xkcd:grey')
        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                      extent=(-self.L,self.L,-self.L,self.L),
                                      cmap='inferno')
        
        self.visu.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            left=False,
            labelleft=False) # labels along the bottom edge are off
        #Posició de la figura
        self.visu.set_position([0,0.15,0.8,0.8])
        
        self.main_canvas.draw()

        ##################################MATPLOTLIB EVENTS/RESIZING
        #self.main_canvas.mpl_connect('motion_notify_event', motionnotify)
        
        #Conectem l'event resize de matplotlib amb la funció resize_kivy.
        #Aquesta útlima ens permet saber on estan els cantons de la caixa en
        #coordenades de pixels de la finestra en un primer instant (ja que quan
        # kivy obre matplotlib per primer cop fa ja un resize), i tambe posteriorment,
        # cada cop que fem el resize. Això ens podria ser útil en algún moment.
        
        #self.main_canvas.mpl_connect('resize_event',resize)
        self.main_canvas.mpl_connect('resize_event',partial(self.resize_kivy))
        

        
        ################################POTENCIAL MODE/Widget de dibuix
        self.ara=0
        self.lit=0

        self.paint=PaintWidget()
        self.arrow=Arrow()
        self.slitcanvas=Slit()
        #No pinta a no ser que estigui activat.
        self.paint.pause_paint=True
        self.box1.add_widget(self.paint,0)
        self.box1.add_widget(self.arrow,0)
        self.box1.add_widget(self.slitcanvas,0)
        ###############################PROPIETATS DEL QUADRAT AMB EL QUE JUGUEM
        self.quadrat.pos=[800,800]
        self.fish.pos=[100,800]
        self.buttong.pos=[100,800]
        self.door.pos=[100,800]
        self.pause_game=True
        self.lv_inicial=True
        self.life=100
        
        self.fishcatched=0
        ####Propietats de la fletxa del moment
        #self.arrow.pos=[800,800]
        #self.arrow.points=[0,0,1,1]
        ###################################BUTONS DE LA CAIXA 2###################
        #Aquests butons varian el menu principal del joc
        
                
        #Aquest activa la funció que activa,atura, o reseteja el calcul
        #self.box2.add_widget(Button(text='Play',on_press=partial(self.compute)))
        #self.box2.add_widget(Button(text='Pause',on_press=partial(self.pause)))
        #self.box2.add_widget(Button(text='Reset',on_press=partial(self.reset)))
        #self.box2.add_widget(Button(text='Standard',on_press=partial(self.standard)))
        #self.box2.add_widget(Button(text='Slit',on_press=partial(self.slit)))
        #self.box2.add_widget(Button(text='Game',on_press=partial(self.gameon)))
        #self.box2.add_widget(Button(text='Back',on_press=partial(self.transition_PS)))
        
        

        #####################################PAUSESTATE
        self.lv_1=True
        #Coses del nivell 2
        self.slitwidthlist=np.array([3,1.5,0.75,0.5])
        self.lv_2=False
        self.conf=0
        self.xslit0=(self.L*(2))*(4/10.)-self.L
        self.xslit1=(self.L*(2))*(6/10.)-self.L
        self.slitwidth=self.slitwidthlist[0]
        self.yslitinf=-self.slitwidth/2.
        self.yslitsup=+self.slitwidth/2.
        
        self.value=10000
        self.s2=self.dx/2.
        self.closedslit=False
        self.doorphase=False
        self.phaselvl2=1
        
        #Cmaps
        self.cmap='inferno'

    ##Ara venen tot el seguït de funcions que utilitzarem conjuntament amb els 
    #parametres definits a g_init

    ##############################RESIZING############################
    #1.Resize_kivy
    
    #Funcions que s'ocupen del tema resize:
        
    def resize_kivy(self,*largs):
        """Aquesta funció és trucada cada cop que la finestra canvia de tamany.
        Cada cop que això passa, les coordenades (en pixels) dels cantons del plot 
        imshow canvien de lloc. Aquests són molt importants sobretot pel joc, per
        tant, cal tenir-nos ben localitzats."""
        #Aquesta funció de matplotlib ens transforma els cantons de coordenades Data
        # a display (pixels)
        self.cantonsnew=self.visu.transData.transform([(-3,-3),(-3,3),(3,3),(3,-3)])
 
        self.ara=0
        
                

        
    ################################FUNCIONS DE CALCUL/FLUX##################
    #1.g_schedule_fired
    #2.plotpsiev
    #3.compute(play)
    #4.pause
    #5.reset
    #6.transition_PS
    def calculcantons(self):
        self.cantonsnew=self.visu.transData.transform([(-3,-3),(-3,3),(3,3),(3,-3)])
        
        print(self.cantonsnew)
    def g_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotpsiev,1/60.)
    
    def g_schedule_cancel(self):
        self.schedule.cancel()

           
    
    def plotpsiev(self,dt):
        "Update function that updates psi plot"
              
        if self.ara<1:
            self.windowsizeg=Window.size
            self.ara=self.ara+1
            self.cantonsnew=self.visu.transData.transform([(-3,-3),(-3,3),(3,3),(3,-3)])
            
            print(self.cantonsnew)
            if self.lit==0:
                self.cantonsold=self.cantonsnew
            

            
            self.lit+=1
            

            
            if (np.abs(self.px)>0. or np.abs(self.py)>0.) and self.t_total<self.dt:
                
                self.arrow.canvas.clear()
                if self.lv_2==True:
                    if self.conf==0:
                        x0=(self.L*2)*(7.3/10.)-self.L
                        y0=0.
                    elif self.conf==1:
                        x0=(self.L*2)*(3.55/10.)-self.L
                        y0=0.
                elif self.lv_1==True:
                    x0=0.
                    y0=0.
                        

                self.cent=self.visu.transData.transform([(x0,y0)])
                
                with self.arrow.canvas:
                    Color(self.arrowcolor[0],self.arrowcolor[1],self.arrowcolor[2])
                    x0=self.cent[0,0]
                    y0=self.cent[0,1]
                    x1=x0+self.px*7.5
                    y1=y0+self.py*7.5
                    Line(points=(x0,y0,x1,y1),width=self.arrowwidth)
                    p_max=np.sqrt(10**2 + 10**2)
                    p=np.sqrt(self.px**2+self.py**2)
                    c=p/p_max
                    if self.px==0.0:
                        p1=np.array([x0-self.arrowdist*c,y1])
                        p2=np.array([x0+self.arrowdist*c,y1])
                        p3=np.array([x1,y0+self.py*self.arrowdist2])                    
                    elif self.py==0.0:
                        p1=np.array([x1,y0+self.arrowdist*c])
                        p2=np.array([x1,y0-self.arrowdist*c])
                        p3=np.array([x0+self.px*self.arrowdist2,y1])
                    else:
                        m=(y1-y0)/(x1-x0)
                        phi=np.pi/2.-np.arctan(m)
                        xr=x1-np.cos(phi)*self.arrowdist*c
                        xm=x1+np.cos(phi)*self.arrowdist*c
                        p1=np.array([xr,-np.tan(phi)*(xr-x1)+y1])
                        p2=np.array([xm,-np.tan(phi)*(xm-x1)+y1])
                        p3=np.array([x0 + self.px*self.arrowdist2,y0 + self.py*self.arrowdist2])
                    
                    
                    Line(points=(p1[0],p1[1],p2[0],p2[1]),width=self.arrowwidth)
                    Line(points=(p2[0],p2[1],p3[0],p3[1]),width=self.arrowwidth)
                    Line(points=(p3[0],p3[1],p1[0],p1[1]),width=self.arrowwidth)
            
            

                
            if self.lv_2==True and self.closedslit==False:
                if self.conf==0:
                    self.xslit=self.xslit0
                elif self.conf==1:
                    self.xslit=self.xslit1
                self.slitcanvas.canvas.clear()
          
                self.yslitinf=-self.slitwidth/2.
                self.yslitsup=self.slitwidth/2.
                self.inf1=self.visu.transData.transform([(self.xslit,-self.L)])
                self.inf2=self.visu.transData.transform([(self.xslit,self.yslitinf)])
                
                self.sup1=self.visu.transData.transform([(self.xslit,self.yslitsup)])
                self.sup2=self.visu.transData.transform([(self.xslit,self.L)])
                
                with self.slitcanvas.canvas:
                    Color(1,1,1)
                    Line(points=(self.inf1[0,0],self.inf1[0,1],self.inf2[0,0],self.inf2[0,1]))
                    Line(points=(self.sup1[0,0],self.sup1[0,1],self.sup2[0,0],self.sup2[0,1]))   
            
            elif self.lv_2==True and self.closedslit==True:
                if self.conf==0:
                    self.xslit=self.xslit0
                elif self.conf==1:
                    self.xslit=self.xslit1
                self.slitcanvas.canvas.clear()
                self.yslitinf=-self.slitwidth/2.
                self.yslitsup=self.slitwidth/2.
                self.inf1=self.visu.transData.transform([(self.xslit,-self.L)])
        
                self.sup2=self.visu.transData.transform([(self.xslit,self.L)])
                
                with self.slitcanvas.canvas:
                    Color(1,1,1)
                    Line(points=(self.inf1[0,0],self.inf1[0,1],self.sup2[0,0],self.sup2[0,1]))
           
            
                
            if self.cont>=1:
                q_pos=self.visu.transData.transform([self.q_coords[0],self.q_coords[1]])
                self.quadrat.pos=[int(q_pos[0]),int(q_pos[1])]
                
                if self.lv_1==True or (self.lv_2==True and self.fishphase==True):
                    f_pos=self.visu.transData.transform([self.f_coords[0],self.f_coords[1]])
                    self.fish.pos=[int(f_pos[0]),int(f_pos[1])]
                
                if self.lv_2==True and self.closedslit==False and self.fishphase==False:
                    
                    b_pos=self.visu.transData.transform([self.b_coords[0],self.b_coords[1]])
                    self.buttong.pos=[int(b_pos[0]),int(b_pos[1])]
                    
                if self.lv_2==True and self.doorphase==True:
                    d_pos=self.visu.transData.transform([self.d_coords[0],self.d_coords[1]])
                    self.door.pos=[int(d_pos[0]),int(d_pos[1])]
                
            self.cont+=1
        
        if self.cont>=1:
            inv_data=self.visu.transData.inverted()
            q_pos0=self.quadrat.pos
            f_pos0=self.fish.pos
            b_pos0=self.buttong.pos
            d_pos0=self.door.pos
            #Posició del centre del quadrat
            self.q_coords=inv_data.transform((q_pos0[0],
                                             q_pos0[1]))
            self.f_coords=inv_data.transform((f_pos0[0],
                                             f_pos0[1]))
            self.b_coords=inv_data.transform((b_pos0[0],
                                             b_pos0[1]))
            self.d_coords=inv_data.transform((d_pos0[0],
                                             d_pos0[1]))
            
            
            
        if self.lv_inicial==True and self.t_total<self.dt:
            self.setlvl1()
            self.lv_inicial=False
            self.lv_1=True
                    
        if self.pause_state==False:
            #Primer de tot, representem el primer frame
            
           # if (self.i)==0:
            #    self.compute_parameters()    
            #Animation
            #Lanimació consisteix en elimanr primer el que teniem abans
            self.normavec=ck.norma(self.psi0,self.Nx)
            self.visu_im.remove()
            #Posar el nou
            #self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
            #                              extent=(-self.L,self.L,-self.L,self.L)) 
            self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                          extent=(-self.L,self.L,-self.L,self.L),
                                           cmap='inferno')
            #self.visu.set_facecolor(color='xkcd:warm gray')
            #I utilitzar-lo
            self.main_canvas.draw()
            #CALCUL
            #Tot següit fem el càlcul del següent step
            self.psi0=ck.Crannk_stepm(self.psi0, self.avec, self.cvec, self.r, 
                                      self.Vvec, self.dt, self.Nx, self.i)
            
            #Canviem el temps
            self.t_total=self.t_total+(self.dt)/2.
            self.box3.tempschange.text='{0:.3f}'.format(self.t_total)
            self.i+=1
            if 10>self.i>1:
                self.box4.statechange.text='Go!'
            
            if self.i>5:
                if self.lv_1==True:
                    self.box4.statechange.text='Fishing'
                    
            #Calculem en tot moment on estan tots els objectes:
            
            
               
            #Videojoc, coses del  videojoc
        if self.pause_game==False and self.lv_1==True:
                        #Utilitzem la transformació inversa que ens porta de pixels matplotlib 
                # a dades matplotlib.
                
            
            inv_data=self.visu.transData.inverted()
            pos=self.quadrat.pos
            #Posició del centre del quadrat
            pos_data=inv_data.transform((pos[0]+self.quadrat.width/2,
                                             pos[1]+self.quadrat.height/2))
            self.pos_discret=np.array([(pos_data[0]+self.L)/self.dx,
                              (pos_data[1]+self.L)/self.dx],dtype=int)
       

            self.llindar=0.03
            self.valor_actual=self.normavec[self.pos_discret[1],self.pos_discret[0]]
            if self.valor_actual>self.llindar:
                print('Touched',self.valor_actual )
                
                if self.llindar<self.valor_actual<0.05:                 
                    self.lifecontrol(-3)
                
                elif 0.05<self.valor_actual<0.1:
                    self.lifecontrol(-4)
                elif 0.1<self.valor_actual<0.15:
                    self.lifecontrol(-5)
                elif 0.15<self.valor_actual<0.2:
                    self.lifecontrol(-6)
                else:
                    self.lifecontrol(-7)
                    
            
            else:
                pass
           
                    
            ##fishthings
            if  (self.fish.x-self.fish.width/2.)<self.quadrat.x<(self.fish.x+(3*self.fish.width)/2.):
                if (self.fish.y-(self.fish.height)/2.)<self.quadrat.y<(self.fish.y+(3*(self.fish.height))/2):
                        #Generem una posició aleatoria
                        
                    x=np.random.random()*5.65-3.00
                    y=np.random.random()*5.65-3.00
                    self.drawfish(x,y)
                    self.lifecontrol(+3)
                    
                    
                
        if self.pause_game==False and self.lv_2==True:
                        #Utilitzem la transformació inversa que ens porta de pixels matplotlib 
                # a dades matplotlib.
            inv_data=self.visu.transData.inverted()
            pos=self.quadrat.pos
            #Posició del centre del quadrat
            pos_data=inv_data.transform((pos[0]+self.quadrat.width/2,
                                             pos[1]+self.quadrat.height/2))
            self.pos_discret=np.array([(pos_data[0]+self.L)/self.dx,
                              (pos_data[1]+self.L)/self.dx],dtype=int)
       

            self.llindar=0.02
            self.valor_actual=self.normavec[self.pos_discret[1],self.pos_discret[0]] 
            
            if self.valor_actual>self.llindar:
                print('Touched',self.valor_actual )
                
                if self.llindar<self.valor_actual<0.05:                 
                    self.lifecontrol(-3)
                
                elif 0.05<self.valor_actual<0.1:
                    self.lifecontrol(-5)
                elif 0.1<self.valor_actual<0.15:
                    self.lifecontrol(-7)
                elif 0.15<self.valor_actual<0.2:
                    self.lifecontrol(-9)
                elif 0.2<self.valor_actual:
                    self.lifecontrol(-12)
                    
            
            else:
                pass
            
            
            if self.closedslit==False:
                
                if self.i>10:
                    self.box4.statechange.text='Close the slit!'
                #tanquem les portes
                if  (self.buttong.x-self.buttong.width/2.)<self.quadrat.x<(self.buttong.x+(3*self.buttong.width)/2.):
                    if (self.buttong.y-(self.buttong.height)/2.)<self.quadrat.y<(self.buttong.y+(3*(self.buttong.height))/2):
                        #Tanca les portes
                
                        self.closeslit(self.slitwidth,self.conf)
                        self.buttong.pos=[100,800]
                        self.fishphase=True
                        self.fishphase0=True
                        self.box4.statechange.text='Fish'
                        
            #Un cop esta tancada la slit, comença la segona fase del joc, agafar
            #tres peixets que es colocaran aleatòriament. 
            
            if self.closedslit==True and self.fishphase==True:
   
                if self.conf==0:
                    x=np.random.random()*(5.65-(self.xslit0+self.L))+self.xslit0
                    y=np.random.random()*5.65-self.L
                    
                
                if self.conf==1:
                    x=np.random.random()*(5.65-(self.L+self.xslit0))-self.L
                    y=np.random.random()*5.65-self.L
                    

                
                if self.fishphase0==True:
                    pos_data=self.visu.transData.transform([(x,y)])
                    self.fish.x=np.int(pos_data[0,0])
                    self.fish.y=np.int(pos_data[0,1])
                    self.fishphase0=False
                #self.fishphase0=False
           
                 ##fishthings
                if  (self.fish.x-self.fish.width/2.)<self.quadrat.x<(self.fish.x+(3*self.fish.width)/2.):
                    if (self.fish.y-(self.fish.height)/2.)<self.quadrat.y<(self.fish.y+(3*(self.fish.height))/2):
                        self.lifecontrol(+15)
                        self.drawfish(x,y)
                    
            if self.doorphase==True:
                if self.conf==0:
                    x=np.random.random()*(5.65-(self.xslit0+self.L))+self.xslit0
                    y=np.random.random()*5.65-self.L
                
                if self.conf==1:
                    x=np.random.random()*(5.65-(self.L+self.xslit0))-self.L
                    y=np.random.random()*5.65-self.L
                    
                
                if self.doorphase0==True:
                    
                    pos_data=self.visu.transData.transform([(x,y)])
                    self.door.x=np.int(pos_data[0,0])
                    self.door.y=np.int(pos_data[0,1])
                    self.doorphase0=False
                
                if  (self.door.x-self.door.width/2.)<self.quadrat.x<(self.door.x+(3*self.door.width)/2.):
                    if (self.door.y-(self.door.height)/2.)<self.quadrat.y<(self.door.y+(3*(self.door.height))/2):
                        self.doorphase=False
                        self.changeconf()
                        
    def compute(self,*largs):    
        """Quan es clica el butó play, s'activa aquesta funció, que activa
        el calcul"""     
        self.pause_game=False
        self.pause_state=False
        self.gameon()
        #self.boxlife.life_slider.value=50
        self.arrow.canvas.clear()
        self.box4.statechange.text='Ready?'

      
    
    def pause(self,*largs):
        """Aquesta funció para el càlcul del joc"""
        self.pause_state=True
        self.box4.statechange.text='Pause'
        self.pause_game=True
        
        self.gameon()
        
       
    def reset(self,*largs):
        """Aquesta funció reseteja el paquet i el potencial inicials. Segons el
        mode inicial en el que ens trobem, es resetejara un o l altre."""
        if self.pause_state==True:            
            #Segons el mode, reseteja un potencial o un paquet diferent
            
            self.paint.canvas.clear()
            self.arrow.canvas.clear()
            self.slitcanvas.canvas.clear()
                 
            
            #Canviem els parametre en pantalla tambe
            self.t_total=0.00
            self.box3.tempschange.text='0.00'
            self.phaselvl2=1
            self.box4.levelchange.text='{}'.format(self.phaselvl2)
     
            #self.box3.pxchange.text='0.0'
            #self.box3.pychange.text='0.0'
         
            #•self.x0=self.zero()
            #self.y0=self.zero()
            if self.lv_1==True:
                self.activatelvl1()
                #self.box4.box_sel.select_but.disabled=False
            elif self.lv_2==True:
                self.activatelvl2()
                #self.box4.box_sel.select_but.disabled=True
        
            #Parametres que s'han de tornar a resetejar
            self.i=0
            self.px=0.
            self.py=0.
            


            
            self.selectmode_pause=False
            
            self.lifecontrol(150)
            
            

    def transition_GS(self,*largs):
            self.g_schedule_cancel()
            self.manager.transition=FadeTransition()
            
            self.manager.current='starting'
            
    def zero(self,*largs):
        a=0.
        return a
    
    
            
    ############################## MODES####################################
    #Aquestes funcions s'activen quan clikem self o slit i s'encarreguen de que
    #resetejar tot el que hi ha en pantalla (si estan en un mode difent al seu
    #propi) i posar el mode escollit. També escrivim aquí els potencials i
    #paquets propis de cada mode.
    
    #1.standard(mode 0)
    #2.slit(mode slit)
    #3.modechange (canvia entre un mode o l'altre)   
    #4.Vvecstand
    #5.Vvecslit
    #6.psistand
    #7.psislit
    
            
    def standard(self,*largs):
        """Funció que retorna el mode estandard en tots els aspectes: de càlcul,
        en pantalla, etf..."""
        
        if self.mode==1:
            #self.box4.box_sel.select_but.disabled=False
            self.remove_widget(self.box5)
            self.slitcanvas.canvas.clear()
        #self.mode=0
        self.L=3.0
        self.dx=0.03
        self.box3.longitudchange.text='{}'.format(self.L)
        self.Nx=np.int((2*self.L)/self.dx)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        self.Vvec=self.Vvecstand()
        self.psi0=self.psistand(0.,0.)
        #Llevem les líneas que hi pogui haver
        
            
        #Això es un reset, pensar fer-ho amb el reset.
        #Canviem la pantalla i el mode:
        self.i=0
        #Rseteja la imatge
            
        self.normavec=ck.norma(self.psi0,self.Nx)
        self.visu_im.remove()
        #Posar el nou
        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                         extent=(-self.L,self.L,-self.L,self.L),
                                         cmap=self.cmap) 
        

        #I utilitzar-lo
        self.main_canvas.draw()            
  
            
                    
 
    def slit(self,xslit,slitwidth,x0slit,y0slit,*largs):
        """Quan activem aquest buto, canviem el paquet de referencia i el 
        potencial de referència i ho deixem en el mode slit. Aquest mode
        correspon amb el número 1."""
        #self.mode=1
        self.L=3.0
        self.dx=0.03
        self.box3.longitudchange.text='{}'.format(self.L)
        self.Nx=np.int((2*self.L)/self.dx)
        #self.slitwidth=0.5
        #self.xslit=((self.L)*2)*(1/3.)-self.L
        self.xslit=xslit
        self.slitwidth=slitwidth
        self.yslitinf=-self.slitwidth/2.
        self.yslitsup=self.slitwidth/2.
        self.value=10000
        self.s2=self.dx/2.
        self.Vvec=self.Vvecstand()

        self.Vvec+=self.potencial_final(self.xslit,-self.L,self.xslit,self.yslitinf,
                                        self.value,self.s2)
        self.Vvec+=self.potencial_final(self.xslit,self.yslitsup,self.xslit,self.L,
                                        self.value,self.s2)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        self.psi0=self.psistand(x0slit,y0slit)
        #self.box4.box_sel.select_but.disabled=True
        
        #Parametres del forat del slit

        
        """Pintem tot lo dit i afegim una líniea que es correspón amb el potencial,
        fix en aquest mode."""
        
    #Rseteja la imatge
            
        self.normavec=ck.norma(self.psi0,self.Nx)
        self.visu_im.remove()
        #Posar el nou
        self.visu_im=self.visu.imshow(self.normavec,origin={'upper'},
                                         extent=(-self.L,self.L,-self.L,self.L),cmap='inferno') 
 
        
        
        self.inf1=self.visu.transData.transform([(self.xslit,-self.L)])
        self.inf2=self.visu.transData.transform([(self.xslit,self.yslitinf)])
        
        self.sup1=self.visu.transData.transform([(self.xslit,self.yslitsup)])
        self.sup2=self.visu.transData.transform([(self.xslit,self.L)])
        
        self.slitcanvas.canvas.clear()
        with self.slitcanvas.canvas:
            Color(1,1,1)
            Line(points=(self.inf1[0,0],self.inf1[0,1],self.inf2[0,0],self.inf2[0,1]))
            Line(points=(self.sup1[0,0],self.sup1[0,1],self.sup2[0,0],self.sup2[0,1]))               
        #Afegim un slider
        
        #if self.mode==0:
            #self.add_widget(self.box5)
            #self.box4.box_sel.select_but.disabled=True
            
        

        #I utilitzar-lo
        self.main_canvas.draw()                    
 
        
        
    def modechange(self,*largs):
        """Funció que s'encarrega de canviar de mode. De moment només en tenim
        2. Possible futura ampliació."""
        #primer de tot apliquem un reset
        
        self.reset()
        #Canviem el mode posteriorment.
        if self.mode==1:
            self.standard()
            self.mode=0
            self.box21.modechange.text='Standard'
            
        else:
            self.slit()
            self.box21.modechange.text='Slit'
            self.mode=1
    
    def Vvecstand(self,*largs):
        """Potencial del mode estandard"""
        Vvecestandard=np.array([[ck.Vharm(self.xa+i*self.dx,self.xa+j*self.dx,
                                      self.xb,self.xb) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)],dtype=np.float64)
        
        return Vvecestandard
    
    def Vvecslit(self,*largs):
        """Potencial del mode slit"""
        #Potencial slit(mode=1)
        self.yposslitd=np.int(self.Nx*(4.75/10))
        self.yposslitu=np.int(self.Nx*(5.25/10))
        self.xposslit=np.int(self.Nx/3)
        
        Vvecslit=self.Vvecstand()
        
        Vvecslit[0:self.yposslitd,self.xposslit]=100000
        
        Vvecslit[self.yposslitu:self.Nx,self.xposslit]=100000
        
        return Vvecslit
    
    def psistand(self,x0,y0,*largs):
        """Paquet corresponent al mode standard"""
        
        psiestandar=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,x0,y0) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])
        
        return psiestandar
        
    def psislit(self,x0,y0,px0,py0,*largs):
        """Paquet corresponent al mode slit"""
        
        #Paquet propi del mode slit
        psislit=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])
        
        return psislit
    
    def emptylist(self,*largs):
        """Returns an empty list"""
        
        empty=np.array([])
        return empty
    #####################################CHANGE PARAMETERs##################
    #Fucnions que s'encarregar del canvi efectiu dels parametres
    #1.changepx
    #2.cahngepy
    #3.applychange
    def changepx(self,value_px,*largs):
        """Funció que respon als buto + i - que varia el moment inicial del paquet.
        Per que aquest canvi sigui efectiu, hem de canviar tambe el propi paquet."""
        #if selfpause==True
        #Canvi el moment en pantalla
        self.px=value_px
        
        self.box3.pxchange.text='{0:.0f}'.format(self.px)
       
        #Canvi del moment efectiu
        #if self.mode==0:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
        #                             for i in range(self.Nx+1)]
        #                       for j in range(self.Nx+1)])
        #if self.mode==1:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
        #                         for i in range(self.Nx+1)]
        #                   for j in range(self.Nx+1)])
    def changedrawpx(self,value_px,x0,y0,*largs):
        
        
        self.cent=self.visu.transData.transform([(x0,y0)])
        

        self.px=value_px
        
        if self.t_total<self.dt:
            
            self.arrow.canvas.clear()
            with self.arrow.canvas:
                Color(self.arrowcolor[0],self.arrowcolor[1],self.arrowcolor[2])
                x0=self.cent[0,0]
                y0=self.cent[0,1]
                x1=x0+self.px*7.5
                y1=y0+self.py*7.5
                Line(points=(x0,y0,x1,y1),width=self.arrowwidth)
                p_max=np.sqrt(10**2 + 10**2)
                p=np.sqrt(self.px**2+self.py**2)
                c=p/p_max
                if self.px==0.0:
                    p1=np.array([x0-self.arrowdist*c,y1])
                    p2=np.array([x0+self.arrowdist*c,y1])
                    p3=np.array([x1,y0+self.py*self.arrowdist2])                    
                elif self.py==0.0:
                    p1=np.array([x1,y0+self.arrowdist*c])
                    p2=np.array([x1,y0-self.arrowdist*c])
                    p3=np.array([x0+self.px*self.arrowdist2,y1])
                else:
                    m=(y1-y0)/(x1-x0)
                    phi=np.pi/2.-np.arctan(m)
                    xr=x1-np.cos(phi)*self.arrowdist*c
                    xm=x1+np.cos(phi)*self.arrowdist*c
                    p1=np.array([xr,-np.tan(phi)*(xr-x1)+y1])
                    p2=np.array([xm,-np.tan(phi)*(xm-x1)+y1])
                    p3=np.array([x0 + self.px*self.arrowdist2,y0 + self.py*self.arrowdist2])
                
                
                Line(points=(p1[0],p1[1],p2[0],p2[1]),width=self.arrowwidth)
                Line(points=(p2[0],p2[1],p3[0],p3[1]),width=self.arrowwidth)
                Line(points=(p3[0],p3[1],p1[0],p1[1]),width=self.arrowwidth)
                
    def changedrawpy(self,value_py,x0,y0,*largs):
        
        self.cent=self.visu.transData.transform([(x0,y0)])
        self.py=value_py
        
        if self.t_total<self.dt:
            
            self.arrow.canvas.clear()
            with self.arrow.canvas:
                Color(self.arrowcolor[0],self.arrowcolor[1],self.arrowcolor[2])
                x0=self.cent[0,0]
                y0=self.cent[0,1]
                x1=x0+self.px*7.5
                y1=y0+self.py*7.5
                Line(points=(x0,y0,x1,y1),width=self.arrowwidth)
                p_max=np.sqrt(10**2 + 10**2)
                p=np.sqrt(self.px**2+self.py**2)
                c=p/p_max
                if self.px==0.0:
                    p1=np.array([x0-self.arrowdist*c,y1])
                    p2=np.array([x0+self.arrowdist*c,y1])
                    p3=np.array([x1,y0+self.py*self.arrowdist2])                    
                elif self.py==0.0:
                    p1=np.array([x1,y0+self.arrowdist*c])
                    p2=np.array([x1,y0-self.arrowdist*c])
                    p3=np.array([x0+self.px*self.arrowdist2,y1])
                else:
                    m=(y1-y0)/(x1-x0)
                    phi=np.pi/2.-np.arctan(m)
                    xr=x1-np.cos(phi)*self.arrowdist*c
                    xm=x1+np.cos(phi)*self.arrowdist*c
                    p1=np.array([xr,-np.tan(phi)*(xr-x1)+y1])
                    p2=np.array([xm,-np.tan(phi)*(xm-x1)+y1])
                    p3=np.array([x0 + self.px*self.arrowdist2,y0 + self.py*self.arrowdist2])
                
                
                Line(points=(p1[0],p1[1],p2[0],p2[1]),width=self.arrowwidth)
                Line(points=(p2[0],p2[1],p3[0],p3[1]),width=self.arrowwidth)
                Line(points=(p3[0],p3[1],p1[0],p1[1]),width=self.arrowwidth)
            
        
        
        
    def changepy(self,value_py,*largs):
        """Funció que respon als buto + i - que varia el moment inicial del paquet.
        Per que aquest canvi sigui efectiu, hem de canviar tambe el propi paquet."""
        #if selfpause==True
        #Canvi el moment en pantalla
        self.py=value_py
        
        self.box3.pychange.text='{0:.0f}'.format(self.py)
        
        #Canvi del moment efectiu
        #if self.mode==0:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
        #                             for i in range(self.Nx+1)]
        #                       for j in range(self.Nx+1)])
        #if self.mode==1:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
        #                         for i in range(self.Nx+1)]
        #                   for j in range(self.Nx+1)]) 
    
    def applychanges(self,x0,y0,*largs):
        """Aquesta funció s'encarrega d'aplicar les modificacions físiques
        fetes al paquet mitjançant el menú d'edició. """
        
        #Canvi del moment efectiu
        
        #if self.t_total<self.dt:
            
            
        self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,x0,y0) 
                                         for i in range(self.Nx+1)]
                                   for j in range(self.Nx+1)])
            
        


        
   
    ################################# POTENCIAL MODE #########################
    #Aquestes funcions juntament amb el widget MyPaintWidget porten la posibilitat
    #de poder dibuixar un potencial.
    #1.editorfun(start)
    #2.editorstop(stop)
    #3.Activatepaint(pinta només si self.paint.pause_paint==False)
    #4.Potpress(apunta les coordenades on es pren el potencial)
    #5.Potrelease(apunta les coordenades on es solta el potencial)
    #6.modifypot(Agafa aquests dos punts i els ajunta, tot creaunt un potencial 
    # infinit al mig.
    #7.clear (reseteja el potencial que s'hagui dibuixat, amb tot el que implica)
       



        #self.modifypot()
        
 
    

        
    def potencial_final(self,x0,y0,x1,y1,value,s2):
        """Funció que genera un potencial infinit que va desde x0,y0 fins x1,y1 """
        
        
        #Vecgaussia=self.potential_maker(x0,y0,x1,y1,value,s2)
        Vvecsegur=self.Vvecmarcat(x0,y0,x1,y1,value*(1./(np.sqrt(s2*2.*np.pi))))
                #I el sumem...
        Vvecfinal=Vvecsegur
        #Coloquem una petita gaussiana al voltant de cada punt de sigma*2=self.dx/2
        #self.Vvec=Vvec
        #Modifiquem els respectius ac,vc
        #self.Vvec=Vvec
        return Vvecfinal
    def gaussian_maker(self,x,y,x0,y0,x1,y1,value,s2,*largs):
        """Introduïm aquí la gaussiana per cada x e y... x0,y0,x1,y1, son els
        dos punts que uneixen la recta dibuixada. Utilitzarem les rectes perpen-
        diculars en aquestes a cada punt per dissenyar aquest potencial."""
        #redefinim x0,y0,x1,y1 a punts que siguin exactament part del discretitzat
        i0=np.int((x0+self.L)/self.dx)
        i1=np.int((x1+self.L)/self.dx)
        j0=np.int((y0+self.L)/self.dx)
        j1=np.int((y1+self.L)/self.dx)
        
        x0=-self.L+self.dx*np.float(i0)
        x1=-self.L+self.dx*np.float(i1)
        y0=-self.L+self.dx*np.float(j0)
        y1=-self.L+self.dx*np.float(j1)
        #Primer de tot, definim el pendent de la recta que uneix els dos punts
        if np.abs(x1-x0)<self.dx:
            if (y0<y<y1 or y1<y<y0):                
                x_c=x0
                y_c=y
                Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2**2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
            else:
                Vvecgauss=0.
                
        
        elif np.abs(y1-y0)<self.dx:
            if (x0<x<x1 or x1<x<x0):                
                x_c=x
                y_c=y0
                Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2**2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
            else:
                Vvecgauss=0.
        else:
            
            m=(y1-y0)/(x1-x0)                      
            x_c=0.5*(x0+x+(y-y0)/m)
            y_c=m*(x_c-x0)+y0
        
            if (x0<x_c<x1 or x1<x_c<x0) or (y0<y_c<y1 or y1<y_c<y0):
                Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2**2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
      
            else:
                Vvecgauss=0.
            
        if np.abs(y-self.dx)<np.abs(((y1-y0)/(x1-x0))*(x-x0)+y0)<np.abs(y+self.dx):
            Vvecgauss=0.
 
        else:
            pass
                
        
        
                
                
        return Vvecgauss
    
    def potential_maker(self,x0,y0,x1,y1,value,s2,*largs):
        Vvecgauss1=np.array([[self.gaussian_maker(self.xa+i*self.dx,self.xa+j*self.dx,
                                      x0,y0,x1,y1,value,s2) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)])
        
        return Vvecgauss1
    
    def Vvecmarcat(self,x0,y0,x1,y1,value,*largs):
        
        Vvecmarcat=self.Vvecstand()
        
        if np.abs(x0-x1)<np.abs(y0-y1):
            num=np.int(np.abs(y0-y1)/self.dx)
        else:
            num=np.int(np.abs(x0-x1)/self.dx)
            
        xvec=np.linspace(x0,x1,num)
        yvec=np.linspace(y0,y1,num)

        for i in range(num):
            Vvecmarcat[np.int((+self.L+yvec[i])/self.dx),
                       np.int((+self.L+xvec[i])/self.dx)]=value
            
        return Vvecmarcat
            
            
        
        
    def clear(self,*largs):
        """Aquesta funció neteja el canvas i tot els canvis introduïts 
        tan al potencial com a les coordenades del potencial. """
        self.paint.canvas.clear()


        if self.mode==0:
            
            self.Vvec=self.Vvecstand()

        if self.mode==1:
            self.Vvec=self.Vvecslit()
            
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        print('Clear!')
        

            

            
        #parem la selecció
 
        #generem el nou paquet
    ################################### COMPUTE PARAMETES#####################
    #Aquestes funcions efectuan càlculs de diferents parametres. De moment, 
    # només de l'energia i la norma
    #1.compute_parameters
    
    def compute_parameters(self,*largs):
        """Aquesta funció s'encarrega de calcular la norma i la energia del
        paquet en un cert instant t i despres l'ensenya a pantalla."""
        
        #Calculem moments i energia inicials i els posem al lloc corresponent
        self.normavec=ck.norma(self.psi0,self.Nx)
        
        self.norma0=ck.trapezis(self.xa,self.xb,self.dx,np.copy(self.normavec))
        self.energy0=ck.Ham(self.xa,self.xb,self.dx,np.copy(self.psi0))
        self.box3.normachange.text='Norma={0:.3f}'.format(self.norma0)
        self.box3.energychange.text='Energia={0:.3f}'.format(self.energy0)
        
    ##################################### JOC ##########################
    #Funcions relacionades amb el propi joc
    #1._keyboard_closed
    #2._on_keyboard_down
    #3.on_position
    #4.on_game
    
    def _keyboard_closed(self):
        """No se ben bé que fa però es necessària"""
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)

        self._keyboard = None
    



    def _on_keyboard_down(self, keyboard, keycode, text, modifiers,*largs):
        """Aquesta funció s'ocupa de conectar el moviment del quadrat amb 
        el teclat. És a dir, conecta l'esdeveniment de teclejar  amb el de
        fer que el quadrat avanci un pas. A l'hora, també imposa els limits on
        el quadrat es pot moure.
        """
        #Definim els limits superior e inferior
        xliminf=np.int(self.cantonsnew[0,0])
        xlimsup=np.int(self.cantonsnew[2,0])
        yliminf=np.int(self.cantonsnew[0,1])
        ylimsup=np.int(self.cantonsnew[1,1])
        
        #defim el pas
        pas=20
        #definim a quina distància s'ha de parar el rectangle.
        w=self.quadrat.width
        h=self.quadrat.height
        #Hem de pensar que la posició es defineix a la  cantonada inferior esquerre
        #del rectangle
        dist=5
        
        

        if self.lv_1==True:
            #Conexions events amb moviment quadrat
            if keycode[1] == 'w':
                if self.quadrat.y+dist+h>ylimsup:
                    pass
                else:
                    self.quadrat.y += pas
                    self.position=self.quadrat.pos
                return True
    
            if keycode[1] == 's':
                if self.quadrat.y-dist<yliminf:
                    pass
                else:
                    self.quadrat.y-= pas
                    self.position=self.quadrat.pos
                #self.position=self.quadrat.position
                #print(self.quadrat.position)  
                return True
    
    
            if keycode[1]=='d':     
                if self.quadrat.x+dist+w>xlimsup:
                    pass
                else:
                    self.quadrat.x+=pas
                    self.position=self.quadrat.pos
                return True
     
                
                   
            elif keycode[1] =='a' :
                if self.quadrat.x-dist<xliminf:
                    pass
                else:
                    self.quadrat.x -=pas
                    self.position=self.quadrat.pos
                return True
            
            
        if self.lv_2==True:
            #Conexions events amb moviment quadrat
            #Necessitem saber on es troba en tot moment el potencial, per tant
            if self.conf==0:
                self.inf=self.visu.transData.transform([(self.xslit0,self.yslitinf)])
                self.sup=self.visu.transData.transform([(self.xslit0,self.yslitsup)])
                
            elif self.conf==1:
                self.inf=self.visu.transData.transform([(self.xslit1,self.yslitinf)])
                self.sup=self.visu.transData.transform([(self.xslit1,self.yslitsup)])
                
            
                
            if keycode[1] == 'w':
                if self.quadrat.y+dist+h>ylimsup:
                    pass
                else:
                    self.quadrat.y += pas
                    self.position=self.quadrat.pos
                return True
    
            if keycode[1] == 's':
                if self.quadrat.y-dist<yliminf:
                    pass
                else:
                    self.quadrat.y-= pas
                    self.position=self.quadrat.pos
                #self.position=self.quadrat.position
                #print(self.quadrat.position)  
                return True
    
    
            if keycode[1]=='d':     
                
                if self.conf==0:
                    if self.quadrat.x+dist+w>xlimsup:
                        pass
                    
                    
                    elif self.inf[0,0]+w>self.quadrat.x+w+dist>self.inf[0,0] and (self.quadrat.y<self.inf[0,1] or self.quadrat.y+h>self.sup[0,1]) and self.closedslit==False:
                        
                        pass
                    
                    
                    else:
                        self.quadrat.x+=pas
                        self.position=self.quadrat.pos
                                                       
                    return True
                
                if self.conf==1:
                    if self.quadrat.x+dist+w>xlimsup:
                        pass
                    
                    
                    elif self.inf[0,0]+w>self.quadrat.x+w+dist>self.inf[0,0] and (self.quadrat.y<self.inf[0,1] or self.quadrat.y+h>self.sup[0,1]) and self.closedslit==False:
                        
                        pass
                    
                    elif self.quadrat.x+dist+w>self.inf[0,0] and self.closedslit==True:
                        pass
                    else:
                        self.quadrat.x+=pas
                        self.position=self.quadrat.pos
                                                       
                    return True
     
                
                   
            elif keycode[1] =='a' :
                
                if self.conf==0:
                    
                    if self.quadrat.x-dist<xliminf:
                        pass
                    
                    elif self.inf[0,0]-2*dist<self.quadrat.x-dist<self.inf[0,0] and (self.quadrat.y<self.inf[0,1] or self.quadrat.y+h>self.sup[0,1]) and self.closedslit==False:
                        pass
                    
                    elif self.quadrat.x-dist<self.inf[0,0] and self.closedslit==True:
                        pass
                    else:
                        self.quadrat.x -=pas
                        self.position=self.quadrat.pos
                        
                if self.conf==1:
                    if self.quadrat.x-dist<xliminf:
                        pass
                    
                    elif self.inf[0,0]-2*dist<self.quadrat.x-dist<self.inf[0,0] and (self.quadrat.y<self.inf[0,1] or self.quadrat.y+h>self.sup[0,1]) and self.closedslit==False:
                        pass
                    
                    else:
                        self.quadrat.x -=pas
                        self.position=self.quadrat.pos
                return True
   

        
    def on_position(self,quadrat,pos):
        """Cada cop que el quadrat canvia de posició, això es notifica aquí.
        Utilitzarem aquesta funció (que està anclada a una propietat) per fer
        el canvi de coordenades pixels matplotlib-data matplotlib.
        Tot seguït, sabent on es troba el paquet, podem saber a sobre de quin 
        valor de densitat del paquet es troba i definir que passa quan es troba
        en segons quines situacions..."""

                
        #Utilitzem la transformació inversa que ens porta de pixels matplotlib 
        # a dades matplotlib.
        inv_data=self.visu.transData.inverted()
        
        #Posició del centre del quadrat
        pos_data=inv_data.transform((pos[0]+self.quadrat.width/2,
                                     pos[1]+self.quadrat.height/2))
        
        #Un cop tenim la posició del centre del quadrat en coordenades data, les
        #pasem a coord del discretitzat:
        self.pos_discret=np.array([(pos_data[0]+self.L)/self.dx,
                              (pos_data[1]+self.L)/self.dx],dtype=int)
        
        


        #Busquem el valor màxim de normavec cada cop que es mou el paquet
        #definim un llindar a partir del qual el quadrat detecta el paquet
        

        
    def gameon(self,*largs):
        """Aquesta funció es l'encarregada de posar el joc en marxa. Col·loca
        el quadradet a algún lloc on es pogui veure (una mica cutre però 
        ja ho canviaras). Uneix la funció teclat amb el moviment del quadrat,
        i varia el mode del joc."""

        if self.pause_game==False:
            self._keyboard.bind(on_key_down=self._on_keyboard_down)
            
            
        else:           
            self._keyboard.unbind(on_key_down=self._on_keyboard_down)    
            
    def activatelvl1(self,*largs):
        
        """Funció que deixa preparat el nivell 1"""
        
        
        self.lv_2=False
        self.lv_1=True
        self.slitcanvas.canvas.clear()
        self.paint.canvas.clear()
        self.t_total=0.00
        self.lifecontrol(150)
        self.box3.tempschange.text='0.00'
        #self.arrow.canvas.clear()
        self.buttong.pos=[100,800]
        self.standard()
        self.setlvl1()
        self.box4.gamemode.text='[color=F027E8]Survay Mode[/color]'
    
    def drawlvl1(self,*largs):
        """Dibuixa la configuració inical del nivell 1"""
                     
        self.cantonsnew=self.visu.transData.transform([(-3,-3),(-3,3),(3,3),(3,-3)])
         
        self.quadrat.x=np.int((-self.cantonsnew[0,0]+self.cantonsnew[2,0])/10.
                                      +self.cantonsnew[0,0])
        self.quadrat.y=np.int(2*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/4.
                                      +self.cantonsnew[0,1])   
         
        self.fish.x=np.int(5*(-self.cantonsnew[0,0]+self.cantonsnew[2,0])/6.
                                      +self.cantonsnew[0,0])
         
        self.fish.y=np.int(1*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/4.
                                      +self.cantonsnew[0,1])
         
         

    def setlvl1(self,*largs):
        """Prepara la configuració del paquet pel nivell 1"""
        pxinicial=-10.00
        x0=0.
        y0=0.
        self.fishcatched=0
        self.box4.fishchange.text="{}/10".format(self.fishcatched)
        self.changepx(pxinicial)
        self.changedrawpx(pxinicial,x0,y0)
        self.applychanges(x0,y0)
        self.drawlvl1()
    
    def lifecontrol(self,value,*largs):
        """Controla el valor de salut que es suma o es resta"""
        
        if self.life+value>=100:
            self.life=100
        elif 0<self.life+value<100:
            self.life+=value
        else:
            self.life=0
            
            self.pause()
            self.box4.statechange.text='Finished :('
        self.boxlife.life_slider.value=self.life
        
    def drawfish(self,x,y,*largs):
        """En coordenades del plot les dades introduides. Dixuixa un peix
        alla on s'ha dit."""
        print(x,y)
        pos_data=self.visu.transData.transform([(x,y)])
        print(pos_data)
        #ara tenim les coordenades kivy
        
        if self.lv_1==True:
            
            self.fish.x=np.int(pos_data[0,0])
            self.fish.y=np.int(pos_data[0,1])
            self.fishcatched+=1
            self.box4.fishchange.text="{}/10".format(self.fishcatched)
            
            if self.fishcatched==10:
                self.pause()
                self.box4.statechange.text='Finished!'
                
        if self.lv_2==True:
            self.fishcatched+=1
            self.box4.fishchange.text="{}/3".format(self.fishcatched)
            
            if self.fishcatched==3:
                self.fishphase=False
                self.fish.pos=[100,800]
                self.doorphase=True
                self.doorphase0=True
                self.box4.statechange.text='Get out!'
            
            else:
                self.fish.x=np.int(pos_data[0,0])
                self.fish.y=np.int(pos_data[0,1])
      
                
    def setlvl2(self,conf,slitwidth):
        """Dibuixa les diferents configuracions del nivell 2, que com ja
        sabem, presenta diverses configuracions. El flux serà controlat des de
        plotpsiev"""
        #Paràmetre de les dues configuracions del slit
        xslit0=self.xslit0
        xslit1=self.xslit1
        #Parametres de les deus configuracions del paquet.
        x0slit0=(self.L*2)*(7.3/10.)-self.L
        y0slit=0.
        x0slit1=(self.L*2)*(3.55/10.)-self.L
        self.box4.fishchange.text="0/3".format(self.fishcatched)
        px00=-10.
        px01=10.
        self.box4.statechange.text='Close the slit!'
        if conf==0:
            self.slit(xslit0,slitwidth,x0slit0,y0slit)
            self.changepx(px00)
            self.changedrawpx(px00,x0slit0,y0slit)
            self.applychanges(x0slit0,y0slit)
            self.drawlvl2(0)
            
        elif conf==1:
            self.slit(xslit1,slitwidth,x0slit1,y0slit)
            self.changepx(px01)
            self.changedrawpx(px01,x0slit1,y0slit)
            self.applychanges(x0slit1,y0slit)
            self.drawlvl2(1)
            
        #Dibuixem diferents 
        
    def drawlvl2(self,conf,*largs):
        """Dibuixa la configuració inical del nivell 2"""
                     
        self.cantonsnew=self.visu.transData.transform([(-3,-3),(-3,3),(3,3),(3,-3)])
        self.inf=self.visu.transData.transform([(self.xslit0,self.yslitinf)])
        self.sup=self.visu.transData.transform([(self.xslit0,self.yslitsup)])
        
        if conf==0:
         
            self.quadrat.x=np.int((-self.cantonsnew[0,0]+self.cantonsnew[2,0])/10.
                                      +self.cantonsnew[0,0])
            self.quadrat.y=np.int(3*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/4.
                                      +self.cantonsnew[0,1]) 
            
            self.buttong.x=np.int(7*(-self.cantonsnew[0,0]+self.cantonsnew[2,0])/10.
                                      +self.cantonsnew[0,0])
            
            self.buttong.y=np.int(4*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/5.
                                      +self.cantonsnew[0,1]) 
        
        elif conf==1:
            self.quadrat.x=np.int(9*(-self.cantonsnew[0,0]+self.cantonsnew[2,0])/10.
                                      +self.cantonsnew[0,0])
            self.quadrat.y=np.int(3*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/4.
                                      +self.cantonsnew[0,1]) 
            
            self.buttong.x=np.int(2*(-self.cantonsnew[0,0]+self.cantonsnew[2,0])/10.
                                      +self.cantonsnew[0,0])
            
            self.buttong.y=np.int(4*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/5.
                                      +self.cantonsnew[0,1]) 
        
        self.fish.pos=[100,800]
     
        self.door.pos=[100,800]
        #self.fish.x=np.int(5*(-self.cantonsnew[0,0]+self.cantonsnew[2,0])/6.
        #                              +self.cantonsnew[0,0])
         
        #self.fish.y=np.int(1*(-self.cantonsnew[0,1]+self.cantonsnew[1,1])/4.
        #                              +self.cantonsnew[0,1])
                        
  
    def activatelvl2(self,*largs):
        
        """Aquesta funció s'encarrega de inicialitzar el nivell 2. """
        
            
          
        self.lv_1=False
                    
        self.lv_2=True
        self.t_total=0.00
        self.box3.tempschange.text='0.00'
              #Construeix la configuració incial del nivell 2
        self.conf=0
        self.setlvl2(self.conf,self.slitwidthlist[0])
        self.fish.pos=[100,800]
        self.closedslit=False
            
        self.fishcatched=0
        self.doorphase=False
        self.doorphase0=False
        self.fishphase=False
        self.fishphase0=False
        self.phaselvl2=1
        self.lifecontrol(150)
        self.box4.gamemode.text='[color=F027E8]Slit Mode[/color]'
        
    
    def game_mode(self,*largs):
        
        if self.pause_state==True:
            if self.lv_1==True:
                self.phaselvl2=1
                self.box4.levelchange.text='{}'.format(self.phaselvl2)
                self.activatelvl2()
            
            elif self.lv_2==True:
                self.phaselvl2=1
                self.box4.levelchange.text='{}'.format(self.phaselvl2)
                self.activatelvl1()
        
    
    def closeslit(self,slitwidth,conf,*largs):
        """Aquesta funció tanca el slit"""
        
        if conf==0:
            xslit=self.xslit0
            
        elif conf==1:
            xslit=self.xslit1
       
        #Canviem el potencial
        
        #self.Vvec+=self.Vvecmarcat(xslit,-slitwidth/2.,xslit,slitwidth/2.,self.value*(1./(np.sqrt(self.s2*2.*np.pi))))
        #Canviem el dibuix del potencial
        self.Vvec+=self.potencial_final(xslit,-slitwidth/2.,xslit,slitwidth/2.,
                                        self.value,self.s2)
        
        self.inf=self.visu.transData.transform([(xslit,-slitwidth/2.)])
        self.sup=self.visu.transData.transform([(xslit, slitwidth/2.)])
        
        
        with self.slitcanvas.canvas:
            Color(1,1,1)
            Line(points=(self.inf[0,0],self.inf[0,1],self.sup[0,0],self.sup[0,1]))
            
        
        self.closedslit=True
        
        
    def changeconf(self,*largs):        
        """Aquesta funció s'encarrega de canviar la pantalla del nivell 2."""
        self.closedslit=False
        self.fishcatched=0
        self.phaselvl2+=1
        

        if self.phaselvl2==self.slitwidthlist.size+1:
            self.pause()
            self.box4.statechange.text='Finished!'
            pass
        else:
            self.box4.levelchange.text='{}'.format(self.phaselvl2)
            self.slitwidth=self.slitwidthlist[np.int(self.phaselvl2-1)]
            if self.conf==0:
                self.conf=1
            elif self.conf==1:
                self.conf=0
            self.pause_game=True
            self.pause_state=True
            self.setlvl2(self.conf,self.slitwidth)
            self.door.pos=[100,800] 
            self.pause_game=False
            self.pause_state=False
            #self.compute()
        
class MyPaintWidget(Widget):

    
    def on_touch_down(self,touch):
        super(MyPaintWidget,self)
        if self.pause_paint==False:
            with self.canvas:
                Color(1, 1, 1)
                d=5.
                Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), 
                         size=(d,d))
                self.firstx=touch.x
                self.firsty=touch.y



        else:
            return super(MyPaintWidget, self).on_touch_down(touch)
        


    def on_touch_up(self, touch):
        super(MyPaintWidget,self)
        if self.pause_paint==False:
            with self.canvas:
                Color(1, 1, 1)
                d=5.
                Ellipse(pos=(touch.x - d / 2, touch.y - d / 2), 
                         size=(d,d))
                Line(points=(self.firstx,self.firsty,touch.x,touch.y))


        else:
            return super(MyPaintWidget, self).on_touch_up(touch)
        

class PaquetRectangle(Widget):
    pass

class Arrow(Widget):
    pass

class Fish(Widget):
    pass

class Slit(Widget):
    pass

class Butg(Widget):
    pass

class Door(Widget):
    pass        

class PaintWidget(Widget):
    pass

def press(event):
    print('press released from test', event.x, event.y, event.button)
    global cordpress
    cordpress=np.array([[event.xdata,event.ydata]])

def release(event):
    global cordrelease
    cordrelease=np.array([[event.xdata,event.ydata]])


def press_sel(event):
    print('press released from test', event.x, event.y, event.button)
    global cordpress_sel
    cordpress_sel=np.array([event.xdata,event.ydata])   





def keypress(event):
    print('key down', event.key)


def keyup(event):
    print('key up', event.key)


def motionnotify(event):
    print('mouse move to ', event.x, event.y)
    return event.xdata,event.ydata

def resize(event):
    print('resize from mpl ', event.width, event.height)
    global windowsizeg
    windowsizeg=np.array([event.width,event.height])
    print(windowsizeg)

def scroll(event):
    print('scroll event from mpl ', event.x, event.y, event.step)


def figure_enter(event):
    print('figure enter mpl')

def axes_enter(event):
    print('it enteeeeeeers')
    
def axes_leave(event):
    print('something')


def figure_leave(event):
    print('figure leaving mpl')


def close(event):
    print('closing figure')

def function(event):
    print(event.xdata,'hello')

   
                                             
if __name__=='__main__':
    PaquetgApp().run()
    
    
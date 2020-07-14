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
from numba import jit
from functools import partial
from kivy.properties import ObjectProperty,ReferenceListProperty,NumericProperty,StringProperty,ListProperty,BooleanProperty
from kivy.graphics import Color, Ellipse,Line,Rectangle
from kivy.core.window import Window
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
    
    

    def __init__(self,**kwargs):
        "Unim la clase amb la paraula self"
        super(PaquetScreen,self).__init__(**kwargs)
        #Aquesta linia uneix keyboard amb request keyboard
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)

        #self._keyboard.bind(on_key_down=self._on_keyboard_down)
        

        
    def ppseudo_init(self):

        "Iniciem el primer dibuix de l'aplicació"
        
        #Variables globals...
        global Vvec,avec,cvec,psi0,normavec,mode,Vvecstandard
        global setcordpress,setcordrelease,avec,cvec
        
        ################### PARAMETRES DE LA CAIXA,DISCRETITZAT,PARTICULA########
        self.L=3
        self.xa=-self.L
        self.xb=self.L
        
        #Discretitzat
        self.tb=0.1
        self.ta=0
        self.dx=0.05
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
        
        
        
        #Canvia el text del interior de la caixa 'parameters' amb els parametr
        #es inicials escollits
        self.box3.dxchange.text="{}".format(self.dx)
        self.box3.longitudchange.text="{}".format(self.L)
        self.box3.dtchange.text="{}".format(self.dt)
        


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
                                      extent=(-self.L,self.L,-self.L,self.L))
                
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
        
        self.main_canvas.mpl_connect('resize_event',resize)
        self.main_canvas.mpl_connect('resize_event',partial(self.resize_kivy))
        

        
        ################################POTENCIAL MODE/Widget de dibuix
        
        self.setcordpress=np.array([])
        self.setcordrelease=np.array([])
        self.paint=MyPaintWidget()
        #No pinta a no ser que estigui activat.
        self.paint.pause_paint=True
        self.box1.add_widget(self.paint)
        

        ###############################PROPIETATS DEL QUADRAT AMB EL QUE JUGUEM
        self.quadrat.pos=[800,800]
        self.play_game=False

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
        self.pause_state=True

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
        
        print(self.cantonsnew)
        
    ################################FUNCIONS DE CALCUL/FLUX##################
    #1.g_schedule_fired
    #2.plotpsiev
    #3.compute(play)
    #4.pause
    #5.reset
    #6.transition_PS
    
    def p_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotpsiev,1/60.)
    
    def p_schedule_cancel(self):
        self.schedule.cancel()

           
    
    def plotpsiev(self,dt):
        "Update function that updates psi plot"
        
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
                                          extent=(-self.L,self.L,-self.L,self.L)) 
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
            
                
            self.i+=1


        

    
    def compute(self,*largs):    
        """Quan es clica el butó play, s'activa aquesta funció, que activa
        el calcul"""     
        
        self.pause_state=False
               
    
    def pause(self,*largs):
        """Aquesta funció para el càlcul del joc"""
        self.pause_state=True
        
       
    def reset(self,*largs):
        """Aquesta funció reseteja el paquet i el potencial inicials. Segons el
        mode inicial en el que ens trobem, es resetejara un o l altre."""
        if self.pause_state==True:
            #Segons el mode, reseteja un potencial o un paquet diferent
            if self.mode==0:
                self.psi0=self.psistand(0.,0.)
                self.Vvec=self.Vvecstand()
            else:
                self.psi0=self.psislit()
                self.Vvec=self.Vvecslit()
        
            #Parametres que s'han de tornar a resetejar
            self.i=0
            self.px=0.
            self.py=0.
            self.t_toal=0.
            
            if self.setcordpress.size>0:
                self.editorstop()
            else:
                pass

            self.paint.canvas.clear()
            self.setcordpress=self.emptylist()
            self.setcordrelease=self.emptylist()          
            self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
            print('Clear!')
            #Rseteja la imatge
            
            self.normavec=ck.norma(self.psi0,self.Nx)
            self.visu_im.remove()
            self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                          extent=(-self.L,self.L,-self.L,self.L)) 
          
            self.main_canvas.draw()
            
            #Canviem els parametre en pantalla tambe
            
            self.box3.tempschange.text='0.0'
            self.box3.pxchange.text='0.0'
            self.box3.pychange.text='0.0'
            self.box4.boxpx.pxslider.value=0.
            self.box4.boxpy.pyslider.value=0.
            self.x0=self.zero()
            self.y0=self.zero()
            
            
            
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
        
        self.mode=0
        self.Vvec=self.Vvecstand()
        self.psi0=self.psistand(0.,0.)
        #Llevem les líneas que hi pogui haver

        self.lineslit1.remove()
        self.lineslit2.remove()
            
        #Això es un reset, pensar fer-ho amb el reset.
        #Canviem la pantalla i el mode:
        self.i=0
        #Rseteja la imatge
            
        self.normavec=ck.norma(self.psi0,self.Nx)
        self.visu_im.remove()
        #Posar el nou
        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                         extent=(-self.L,self.L,-self.L,self.L)) 
        #I utilitzar-lo
        self.main_canvas.draw()            

            
                    
 
    def slit(self,*largs):
        """Quan activem aquest buto, canviem el paquet de referencia i el 
        potencial de referència i ho deixem en el mode slit. Aquest mode
        correspon amb el número 1."""
        self.mode=1
        self.Vvec=self.Vvecslit()
        self.psi0=self.psislit()
        
        """Pintem tot lo dit i afegim una líniea que es correspón amb el potencial,
        fix en aquest mode."""
        
    #Rseteja la imatge
            
        self.normavec=ck.norma(self.psi0,self.Nx)
        self.visu_im.remove()
        #Posar el nou
        self.visu_im=self.visu.imshow(self.normavec,origin={'upper'},
                                         extent=(-self.L,self.L,-self.L,self.L)) 
 
        
        
        self.lineslit1=self.visu.axvline(x=-self.L+self.xposslit*self.dx,
                                         ymin=0,
                                         ymax=0.475,
                                         color='white')
        
        
        
        self.lineslit2=self.visu.axvline(x=-self.L+self.xposslit*self.dx,
                                         ymin=0.525,
                                         ymax=1,color='white',)
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
            self.box21.modechange.text='Standard'
        if self.mode==0:
            self.slit()
            self.box21.modechange.text='Slit'
    
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
        
        self.box3.pxchange.text='{0:.2f}'.format(self.px)
        
        #Canvi del moment efectiu
        #if self.mode==0:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
        #                             for i in range(self.Nx+1)]
        #                       for j in range(self.Nx+1)])
        #if self.mode==1:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
        #                         for i in range(self.Nx+1)]
        #                   for j in range(self.Nx+1)])
        
    def changepy(self,value_py,*largs):
        """Funció que respon als buto + i - que varia el moment inicial del paquet.
        Per que aquest canvi sigui efectiu, hem de canviar tambe el propi paquet."""
        #if selfpause==True
        #Canvi el moment en pantalla
        self.py=value_py
        
        self.box3.pychange.text='{0:.2f}'.format(self.py)
        
        #Canvi del moment efectiu
        #if self.mode==0:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
        #                             for i in range(self.Nx+1)]
        #                       for j in range(self.Nx+1)])
        #if self.mode==1:
        #    self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
        #                         for i in range(self.Nx+1)]
        #                   for j in range(self.Nx+1)]) 
    
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
    
            print('painting')
            
        else:
            self.paint.pause_paint=True
            self.main_canvas.mpl_disconnect(self.cid1)   
            self.main_canvas.mpl_disconnect(self.cid2)
            self.main_canvas.mpl_disconnect(self.cid3)
            self.main_canvas.mpl_disconnect(self.cid4)
            
            
    def potpress(self,*largs):
        """Funció que s'encarrega de guardar en coordenades de data el lloc
        on s'ha apretat al plot"""
        self.setcordpress=np.append(self.setcordpress,cordpress)
        print(self.setcordpress)
    
    def potrelease(self,*largs):
        """Funció que s'encarrega de guardar en coordenades de data el lloc
        on s'ha soltat al plot."""
        self.setcordrelease=np.append(self.setcordrelease,cordrelease)
        print(self.setcordrelease)

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
                #I el sumem...
            self.Vvec+=Vecgaussia
                #Coloquem una petita gaussiana al voltant de cada punt de sigma*2=self.dx/2
                #self.Vvec=Vvec
            #Modifiquem els respectius ac,vc
            #self.Vvec=Vvec
        print(self.Vvec)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
            
        print('Applied!')
        
        if vecsize>0:
            self.editorstop()
        
        else:
            pass
    def gaussian_maker(self,x,y,x0,y0,x1,y1,value,s2,*largs):
        """Introduïm aquí la gaussiana per cada x e y... x0,y0,x1,y1, son els
        dos punts que uneixen la recta dibuixada. Utilitzarem les rectes perpen-
        diculars en aquestes a cada punt per dissenyar aquest potencial."""
        
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
        
            if (x0<x_c<x1 or x1<x_c<x0):
                Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2**2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
            else:
                Vvecgauss=0.
                
                
        return Vvecgauss
    
    def potential_maker(self,x0,y0,x1,y1,value,s2,*largs):
        Vvecgauss1=np.array([[self.gaussian_maker(self.xa+i*self.dx,self.xa+j*self.dx,
                                      x0,y0,x1,y1,value,s2) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)])
        
        return Vvecgauss1
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
        
                                                       
        #Controla que només es pogui dibuixar dins la figura
        self.cidactivates=self.main_canvas.mpl_connect('axes_enter_event',
                                               partial(self.activatepaints,1))
        self.ciddesactivates=self.main_canvas.mpl_connect('axes_leave_event',
                                               partial(self.activatepaints,0))
        
        
        
  
    def editorstops(self,*largs):
        """Aquesta funció desactiva totes les conexions activades a editorfun,
        i per tant, desactiva el mode editor"""
        self.main_canvas.mpl_disconnect(self.cidactivates)   
        self.main_canvas.mpl_disconnect(self.ciddesactivates)

        
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
            

            
    
            print('painting')
            
        else:

            self.main_canvas.mpl_disconnect(self.cid1s)   
            self.main_canvas.mpl_disconnect(self.cid2s)
 
            
            
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
                                          extent=(-self.L,self.L,-self.L,self.L)) 
          
            self.main_canvas.draw()
        
        #parem la selecció
        self.editorstops()
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
    
    

    def __init__(self,**kwargs):
        "Unim la clase amb la paraula self"
        super(GameScreen,self).__init__(**kwargs)
        #Aquesta linia uneix keyboard amb request keyboard
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)

        #self._keyboard.bind(on_key_down=self._on_keyboard_down)
        

        
    def gpseudo_init(self):

        "Iniciem el primer dibuix de l'aplicació"
        
        #Variables globals...
        global Vvec,avec,cvec,psi0,normavec,mode
        
        ################### PARAMETRES DE LA CAIXA,DISCRETITZAT,PARTICULA########
        self.L=3
        self.xa=-self.L
        self.xb=self.L
        
        #Discretitzat
        self.tb=0.1
        self.ta=0
        self.dx=0.05
        self.dt=0.02
        self.Nx=np.int((2*self.L)/self.dx)

        
        #particula i constants físiques
        self.m=1
        self.hbar=1
        self.t_total=0.
        self.x0=0.
        self.y0=0.
        self.px=0
        self.py=0
        
        #Comptador de moments
        self.i=0
        #parametre del discretitzat
        self.r=(self.dt/(4*(self.dx**2)))*(self.hbar/self.m)
        
        
        
        #Canvia el text del interior de la caixa 'parameters' amb els parametr
        #es inicials escollits
        self.box3.dxchange.text="dx={}".format(self.dx)
        self.box3.longitudchange.text="L={}".format(self.L)
        self.box3.dtchange.text="dt={}".format(self.dt)
        


        #################### PARAMETRES NECESSARIS PER EFECTUAR ELS CALCULS######
        
        
        #################### POTENCIALS
        
        #Potencial estandar(mode=0)
        self.Vvecestandar=np.array([[ck.Vharm(self.xa+i*self.dx,self.xa+j*self.dx,
                                      self.xb,self.xb) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)],dtype=np.float64)
        
        
        #Potencial slit(mode=1)
        self.yposslitd=np.int(self.Nx*(4.75/10))
        self.yposslitu=np.int(self.Nx*(5.25/10))
        self.xposslit=np.int(self.Nx/3)
        self.x0slit=self.L/2
        self.y0slit=0.
        self.Vvecslit=np.copy(self.Vvecestandar)
        
        self.Vvecslit[0:self.yposslitd,self.xposslit]=100000
        
        self.Vvecslit[self.yposslitu:self.Nx,self.xposslit]=100000
        
        
        
        ##################### PAQUETS
        
        #Paquet propi del mode estandar
        self.psiestandar=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0,self.y0) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])#dtype=np.complex128)
        
        #Paquet propi del mode slit
        self.psislit=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])
        
        ##################### MODES
        
        #mode: 0, estandard
        #      1,slit
        #      2,disparo(encara no està fet)
        #definm el mode inicial tal que:
            
        self.Vvec=self.Vvecestandar
        self.psi0=self.psiestandar
        #Ens indica en quin mode estem
        self.mode=0
        
        #Per últim, construïm les matrius molt importants per efectuar els càlculs.
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        

        
        
        
        
        ##########################PRIMER CÀLCUL ENERGIA#####
                       
        #Calculem la matriu densitat inicial
        self.normavec=ck.norma(self.psi0,self.Nx)
    
        #Calculem moments i energia inicials i els posem al lloc corresponent
        self.norma0=ck.trapezis(self.xa,self.xb,self.dx,np.copy(self.normavec))
        self.energy0=ck.Ham(self.xa,self.xb,self.dx,np.copy(self.psi0))
        self.box3.normachange.text='Norma={0:.3f}'.format(self.norma0)
        self.box3.energychange.text='Energia={0:.3f}'.format(self.energy0)
        
        
        ##########################PRIMER DIBUIX###################
        
        #creem la figura recipient del plot imshow
        self.main_fig=plt.figure()
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas,1)
        #Plot principal
        self.visu=plt.axes()
        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                      extent=(-self.L,self.L,-self.L,self.L))
                
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
        
        self.main_canvas.mpl_connect('resize_event',resize)
        self.main_canvas.mpl_connect('resize_event',partial(self.resize_kivy))
        

        
        ################################POTENCIAL MODE/Widget de dibuix
        
        self.setcordpress=np.array([])
        self.setcordrelease=np.array([])
        self.paint=MyPaintWidget()
        #No pinta a no ser que estigui activat.
        self.paint.pause_paint=True
        self.box1.add_widget(self.paint)
        

        ###############################PROPIETATS DEL QUADRAT AMB EL QUE JUGUEM
        self.quadrat.pos=[800,800]
        self.play_game=False

        ###################################BUTONS DE LA CAIXA 2###################
        #Aquests butons varian el menu principal del joc

                
        #Aquest activa la funció que activa,atura, o reseteja el calcul
        self.box2.add_widget(Button(text='Play',on_press=partial(self.compute)))
        self.box2.add_widget(Button(text='Pause',on_press=partial(self.pause)))
        self.box2.add_widget(Button(text='Reset',on_press=partial(self.reset)))
        self.box2.add_widget(Button(text='Standard',on_press=partial(self.standard)))
        self.box2.add_widget(Button(text='Slit',on_press=partial(self.slit)))
        self.box2.add_widget(Button(text='Game',on_press=partial(self.gameon)))
        self.box2.add_widget(Button(text='Back',on_press=partial(self.transition_GS)))
        
        

        #####################################PAUSESTATE
        self.pause_state=True

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
        
        print(self.cantonsnew)
        
    ################################FUNCIONS DE CALCUL/FLUX##################
    #1.g_schedule_fired
    #2.plotpsiev
    #3.compute(play)
    #4.pause
    #5.reset
    
    def g_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotpsiev,1/60.)
    
    def g_schedule_cancel(self):
        self.schedule.cancel()

           
    
    def plotpsiev(self,dt):
        "Update function that updates psi plot"
        
        if self.pause_state==False:
            #Primer de tot, representem el primer frame
            
            if (self.i)==0:
                self.compute_parameters()    
            #Animation
            #Lanimació consisteix en elimanr primer el que teniem abans
            self.normavec=ck.norma(self.psi0,self.Nx)
            self.visu_im.remove()
            #Posar el nou
            self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                          extent=(-self.L,self.L,-self.L,self.L)) 
            #I utilitzar-lo
            self.main_canvas.draw()
            #CALCUL
            #Tot següit fem el càlcul del següent step
            self.psi0=ck.Crannk_stepm(self.psi0, self.avec, self.cvec, self.r, 
                                      self.Vvec, self.dt, self.Nx, self.i)
            
            #Canviem el temps
            self.t_total=self.t_total+(self.dt)/2.
            self.box3.tempschange.text='temps={0:.3f}'.format(self.t_total)
            
            #Videojoc, coses del  videojoc
            if self.play_game==True:
                self.maxvalue=np.amax(self.normavec)
                self.llindar=self.maxvalue/20.

                if self.normavec[self.pos_discret[0],self.pos_discret[1]]>self.llindar:
                    print('Touched')
     
            
            
            #Cada 31 pasos de temps calculem el temps:
            
                
            self.i+=1


        

    
    def compute(self,*largs):    
        """Quan es clica el butó play, s'activa aquesta funció, que activa
        el calcul"""     
        
        self.pause_state=False
               
    
    def pause(self,*largs):
        """Aquesta funció para el càlcul del joc"""
        self.pause_state=True
        
       
    def reset(self,*largs):
        """Aquesta funció reseteja el paquet i el potencial inicials. Segons el
        mode inicial en el que ens trobem, es resetejara un o l altre."""
        if self.pause_state==True:
            #Segons el mode, reseteja un potencial o un paquet diferent
            if self.mode==0:
                self.psi0=self.psiestandar
                self.Vvec=self.Vvecestandar
            if self.mode==1:
                self.psi0=self.psislit
                self.Vvec=self.Vvecslit
            
            #Parametres que s'han de tornar a resetejar
            self.i=0
            self.px=0
            self.py=0
            self.t_toal=0.
            self.paint.canvas.clear()
            self.setcordpress=np.array([])
            self.setcordrelease=np.array([])
            #Rseteja la imatge
            
            self.normavec=ck.norma(self.psi0,self.Nx)
            self.visu_im.remove()
            self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                          extent=(-self.L,self.L,-self.L,self.L)) 
          
            self.main_canvas.draw()
            
            #Canviem els parametre en pantalla tambe
            
            self.box3.tempschange.text='0.00'
            self.box3.pxchange.text='0.00'
            self.box3.pychange.text='0.00'
            
    def transition_GS(self,*largs):
            self.g_schedule_cancel()
            self.manager.transition=FadeTransition()
            self.manager.current='starting'
            
    
    #####################################CHANGE PARAMETERs##################
    #Fucnions que s'encarregar del canvi efectiu dels parametres
    #1.changep
    def changep(self,add_px,add_py,*largs):
        """Funció que respon als buto + i - que varia el moment inicial del paquet.
        Per que aquest canvi sigui efectiu, hem de canviar tambe el propi paquet."""
        #if selfpause==True
        #Canvi el moment en pantalla
        self.px+=add_px
        self.py+=add_py
        print(self.px,self.py)
        
        self.box3.pxchange.text='px={}'.format(self.px)
        self.box3.pychange.text='py={}'.format(self.py)
        
        #Canvi del moment efectiu
        if self.mode==0:
            self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
                                     for i in range(self.Nx+1)]
                               for j in range(self.Nx+1)])
        if self.mode==1:
            self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
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
    
            print('painting')
            
        else:
            self.paint.pause_paint=True
            self.main_canvas.mpl_disconnect(self.cid1)   
            self.main_canvas.mpl_disconnect(self.cid2)
            self.main_canvas.mpl_disconnect(self.cid3)
            self.main_canvas.mpl_disconnect(self.cid4)
            
            
    def potpress(self,*largs):
        """Funció que s'encarrega de guardar en coordenades de data el lloc
        on s'ha apretat al plot"""
        self.setcordpress=np.append(self.setcordpress,cordpress)
        print(self.setcordpress)
    
    def potrelease(self,*largs):
        """Funció que s'encarrega de guardar en coordenades de data el lloc
        on s'ha soltat al plot."""
        self.setcordrelease=np.append(self.setcordrelease,cordrelease)
        print(self.setcordrelease)
        self.editorstop()
        
 
    
    def modifypot(self,*largs):
        """Aquesta funció s'aplica quan es clika el buto apply. Durant el
        període que s'ha dibuixat, totes les lineas de potencial s'han en-
        registrat a les variables self.csetcorpress i setcordreleas. Aquestes
        estan en coordenades de self. data i les hem de passar al discretitzat."""
        
        #Variables on guardem les dates del dibuix
        vecpress=self.setcordpress
        vecrelease=self.setcordrelease
        #linias dibuixades=vecsize/2
        vecsize=np.int(vecpress.size)
        #Aquí guardarem els resultats.
        transformvectorx=np.array([],dtype=int)
        transformvectory=np.array([],dtype=int)
        
        #Coloquem una guasiana al voltant del potencial que pintem. La sigma^2 farem que
        #sigui del mateix tamany que el pas
        s2vec=self.dx
        valorVec=10000.

        #Bucle de conversió de les dades
        for i in range(0,vecsize,2):
            #Establim noms de variables
            index=i
            x0=vecpress[index]
            y0=vecpress[index+1]
            x1=vecrelease[index]
            y1=vecrelease[index+1]


            Vecgaussia=self.potential_maker(x0,y0,x1,y1,valorVec,s2vec)
            #I el sumem...
            self.Vvec+=Vecgaussia
            #Coloquem una petita gaussiana al voltant de cada punt de sigma*2=self.dx/2
            #self.Vvec=Vvec
        #Modifiquem els respectius ac,vc
        #self.Vvec=Vvec
        
        print(self.Vvec)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        
        print('Applied!')
        
    def gaussian_maker(self,x,y,x0,y0,x1,y1,value,s2,*largs):
        """Introduïm aquí la gaussiana per cada x e y... x0,y0,x1,y1, son els
        dos punts que uneixen la recta dibuixada. Utilitzarem les rectes perpen-
        diculars en aquestes a cada punt per dissenyar aquest potencial."""
        
        #Primer de tot, definim el pendent de la recta que uneix els dos punts
        if (x1-x0)==0.:
            x_c=x0
            y_c=y
        
        else:
        
            m=(y1-y0)/(x1-x0)
            
            if m==0:
                y_c=y0
                x_c=x
            else:
                x_c=0.5*(x0+x-(y0-y)/m)
                y_c=m*(x_c-x0)+y0
        #definim condicio per si posar-hi gaussiana o no:
        if x0-self.dx<x_c<x1+self.dx:
            #Obteim el punt en y
            #Ara només hem de calcular quant val la gaussiana:
            Vvecgauss=value*(1./(np.sqrt(s2*2.*np.pi)))*np.exp(-(0.5/s2)*((x-x_c)**2
                                                                    +(y-y_c)**2))
        else:
            Vvecgauss=0.
        return Vvecgauss
    
    def potential_maker(self,x0,y0,x1,y1,value,s2,*largs):
        Vvecgauss1=np.array([[self.gaussian_maker(self.xa+i*self.dx,self.xa+j*self.dx,
                                      x0,y0,x1,y1,value,s2) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)])
        
        return Vvecgauss1
    def clear(self,*largs):
        """Aquesta funció neteja el canvas i tot els canvis introduïts 
        tan al potencial com a les coordenades del potencial. """
        self.paint.canvas.clear()
        self.setcordpress=np.array([])
        self.setcordrelease=np.array([])
        if self.mode==0:
            
            self.Vvec=self.Vvecestandar

        if self.mode==1:
            self.Vvec=self.Vvecslit
            
    ############################## MODES####################################
    #Aquestes funcions s'activen quan clikem self o slit i s'encarreguen de que
    #resetejar tot el que hi ha en pantalla (si estan en un mode difent al seu
    #propi) i posar el mode escollit
    #1.standard(mode 0)
    #2.slit(mode slit)
        
    def standard(self,*largs):
        """Funció que retorna el mode estandard en tots els aspectes: de càlcul,
        en pantalla, etf..."""
        if self.mode==0:
            pass
        else:
            
            self.Vvec=self.Vvecestandar
            self.psi0=self.psiestandar
            #Llevem les líneas que hi pogui haver
            if self.mode==1:
                self.lineslit1.remove()
                self.lineslit2.remove()
            
            #Això es un reset, pensar fer-ho amb el reset.
            #Canviem la pantalla i el mode:
            self.i=0
            #Rseteja la imatge
            
            self.normavec=ck.norma(self.psi0,self.Nx)
            self.visu_im.remove()
            #Posar el nou
            self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                          extent=(-self.L,self.L,-self.L,self.L)) 
            #I utilitzar-lo
            self.main_canvas.draw()
            #Canviem el temps tambe
            self.t_total=0.
            self.box3.tempschange.text='temps=0.'
            self.px=0
            self.py=0
            self.box3.pxchange.text='px=0'
            self.box3.pychange.text='py=0'
            self.mode=0
            
                    
 
    def slit(self,*largs):
        """Quan activem aquest buto, canviem el paquet de referencia i el 
        potencial de referència i ho deixem en el mode slit. Aquest mode
        correspon amb el número 1."""
        self.mode=1
        self.Vvec=self.Vvecslit
        self.psi0=self.psislit
        
        """Pintem tot lo dit i afegim una líniea que es correspón amb el potencial,
        fix en aquest mode."""
        
    #Rseteja la imatge
            
        self.normavec=ck.norma(self.psi0,self.Nx)
        self.visu_im.remove()
        #Posar el nou
        self.visu_im=self.visu.imshow(self.normavec,origin={'upper'},
                                         extent=(-self.L,self.L,-self.L,self.L)) 
 
        
        
        self.lineslit1=self.visu.axvline(x=-self.L+self.xposslit*self.dx,
                                         ymin=0,
                                         ymax=0.475,
                                         color='white')
        
        
        
        self.lineslit2=self.visu.axvline(x=-self.L+self.xposslit*self.dx,
                                         ymin=0.525,
                                         ymax=1,color='white',)
        #I utilitzar-lo
        self.main_canvas.draw()
    
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


def figure_leave(event):
    print('figure leaving mpl')


def close(event):
    print('closing figure')

def function(event):
    print(event.xdata,'hello')

   
                                             
if __name__=='__main__':
    PaquetgApp().run()
    
    
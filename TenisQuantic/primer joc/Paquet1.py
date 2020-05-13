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
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.widget import Widget
import numpy as np
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
import matplotlib.pyplot as plt 
from numba import jit
from functools import partial
from kivy.properties import ObjectProperty,ReferenceListProperty,NumericProperty,StringProperty
from kivy.graphics import Color, Ellipse,Line,Rectangle
############
class PaquetApp(App):
    """Inicialitzem la aplicació"""
    
    title='PaquetApp'
    def build(self):
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    "Activarà la pantalla d'inici.Manetja les screens, claro..."
    
    def __init__(self,**kwargs):
        super(MyScreenManager,self).__init__(**kwargs)
        self.get_screen('paquet').gpseudo_init()
        self.get_screen('paquet').g_schedule_fired()

class PaquetScreen(Screen):
    "Això és la part més important de la app. Introduïrem la pantalla inicial"
    def __init__(self,**kwargs):
        "Unim la clase amb la paraula self"
        super(PaquetScreen,self).__init__(**kwargs)
        
    def gpseudo_init(self):

        "Iniciem el primer dibuix de l'aplicació"
        #Parametres utiltizats
        self.L=3
        self.xa=-self.L
        self.xb=self.L
        self.tb=0.1
        self.ta=0
        self.dx=0.045
        self.dt=0.015
        self.Nx=np.int((2*self.L)/self.dx)
        self.Nt=np.int((self.tb-self.ta)/self.dt)
        self.m=1
        self.hbar=1
        self.t_total=0.
        self.x0=0.
        self.y0=0.
        #Canviem el que es mou a la caixa.
        self.box3.dxchange.text="dx={}".format(self.dx)
        self.box3.longitudchange.text="L={}".format(self.L)
        self.box3.dtchange.text="dt={}".format(self.dt)
        
        
        #Moments inicials que no es corresponen amb els finals

        self.px=0
        self.py=0
        
        #Comptador
        self.i=0
        #Generem les variables necessàries per fer tots els càlculs
        #Variable que marca el pas al mètode
        self.r=(self.dt/(4*(self.dx**2)))*(self.hbar/self.m)
        
        #Definim variables que utilitzarem en el càlcul.  Globals perquè 
        # diverses funcions podran canviarales.
        global Vvec,avec,cvec,psi0,normavec,mode
        #Definim el potencial i el paquet del mode estandard, és a dir, aquell amb el que 
        #s'inicialitza el mètode:
            
        #Definim els potencials i paquets que utilitzarem.
        self.Vvecestandar=np.array([[ck.Vharm(self.xa+i*self.dx,self.xa+j*self.dx,
                                      self.xb,self.xb) 
                             for i in range(self.Nx+1)]
                            for j in range(self.Nx+1)],dtype=np.float64)
        
        #Definim el potencial del slit i un parametre propi: self.sizeslit (en coord
        #del discretitzat)
        self.yposslitd=np.int(self.Nx*(4.75/10))
        self.yposslitu=np.int(self.Nx*(5.25/10))
        self.xposslit=np.int(self.Nx/3)
        self.x0slit=self.L/2
        self.y0slit=0.
        self.Vvecslit=np.copy(self.Vvecestandar)
        
        self.Vvecslit[0:self.yposslitd,self.xposslit]=100000
        
        self.Vvecslit[self.yposslitu:self.Nx,self.xposslit]=100000
        
        #Definim els diferents paquets que utilitzarem.
        
        self.psiestandar=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0,self.y0) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])#dtype=np.complex128)
        
        self.psislit=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,self.x0slit,self.y0slit) 
                             for i in range(self.Nx+1)]
                       for j in range(self.Nx+1)])
        
        #Definim els potencials actuals com els estandars.
        self.Vvec=self.Vvecestandar
        self.psi0=self.psiestandar
        #Ens indica en quin mode estem
        self.mode=0
        #mode: 0, estandard
        #      1,slit
        #      2,disparo
        
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        
        
        
        
        #Aquí tenim variables que utiltizarem per la construcció dels nous potencials.
        self.setcordpress=np.array([])
        self.setcordrelease=np.array([])
        
        #Marc del dibuix
                       
        #Aquesta imatge es la imatge inicial, no té res a veure amb el que es veura
        #al final.
        self.normavec=ck.norma(self.psi0,self.Nx)
    
        #Calculem moments i energia inicials i els posem al lloc corresponent
        self.norma0=ck.trapezis(self.xa,self.xb,self.dx,np.copy(self.normavec))
        self.energy0=ck.Ham(self.xa,self.xb,self.dx,np.copy(self.psi0))
        self.box3.normachange.text='Norma={0:.3f}'.format(self.norma0)
        self.box3.energychange.text='Energia={0:.3f}'.format(self.energy0)
        #FIGURA: Dibuix
        
        
        self.main_fig=plt.figure()
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas,1)
        #Plot principal
        self.visu=plt.axes()

        self.visu_im=self.visu.imshow(self.normavec,origin={'lower'},
                                      extent=(-self.L,self.L,-self.L,self.L))
        
        #self.line=self.visu.axvline(x=0,ymin=-3,ymax=3,color='white')
        
        #Posició de la figura
        self.visu.set_position([0,0.15,0.8,0.8])

        #Dibuixem tot lo dit
        #self.main_fig.tight_layout()        
        self.main_canvas.draw()
        
        
        
        
        #WIDGET DIBUIX
        self.paint=MyPaintWidget()
        #No pinta a no ser que estigui activat.
        self.paint.pause_paint=True
        self.box1.add_widget(self.paint)
        
       
        #BUTONS
        #Aquests butons varian el moment inicial del paquet

                
        #Aquest activa la funció que activa,atura, o reseteja el calcul
        self.box2.add_widget(Button(text='Play',on_press=partial(self.compute)))
        self.box2.add_widget(Button(text='Pause',on_press=partial(self.pause)))
        self.box2.add_widget(Button(text='Reset',on_press=partial(self.reset)))
        self.box2.add_widget(Button(text='Standard',on_press=partial(self.standard)))
        self.box2.add_widget(Button(text='Slit',on_press=partial(self.slit)))
        

       
        #PAUSESTATE
        self.pause_state=True
    
        
    ###FUNCIONS DE CÀLCUL
    def g_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotpsiev,1/60.)
        
    
    
    def plotpsiev(self,dt):
        "Update function that updates psi plot"
        
        if self.pause_state==False:
            #Primer de tot, representem el primer frame
            
            if (self.i)%30==0:
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
            
           
            #Cada 31 pasos de temps calculem el temps:
            
                
            self.i+=1

    #Afeim això que ens canvia px e py
    def changep(self,add_px,add_py,*largs):
        self.px+=add_px
        self.py+=add_py
        print(self.px,self.py)
        
        self.box3.pxchange.text='px={}'.format(self.px)
        self.box3.pychange.text='py={}'.format(self.py)
        #Canvia tambe la propia funció d'ona
        
        if self.mode==0:
            self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,0.,0.) 
                                     for i in range(self.Nx+1)]
                               for j in range(self.Nx+1)])
        if self.mode==1:
            self.psi0=np.array([[ck.psi0f(-self.L+i*self.dx,-self.L+j*self.dx,0.25,self.px,self.py,1.5,0) 
                                 for i in range(self.Nx+1)]
                           for j in range(self.Nx+1)])
        
    #Aquesta funció s'encarreguera d'activar el calcul
    
    def compute(self,*largs):        
        #un cop decidit el px i el py per l'usuari, hem de generar la funció
        #d'ona corresponent
        
        self.pause_state=False
        
        
    
    def pause(self,*largs):
        """Aquesta funció para el càlcul del joc"""
        self.pause_state=True
       
    def reset(self,*largs):
        """Aquesta funció reseteja el paquet i el potencial inicials."""
        if self.pause_state==True:
            
            if self.mode==0:
                self.psi0=self.psiestandar
                self.Vvec=self.Vvecestandar
            if self.mode==1:
                self.psi0=self.psislit
                self.Vvec=self.Vvecslit
            
            
            self.i=0
            self.px=0
            self.py=0
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
            self.box3.pxchange.text='px=0'
            self.box3.pychange.text='py=0'
            
    
    ####### FUNCIONS DE L'EDITOR
       
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
        
        #Bucle de conversió de les dades
        for i in range(0,vecsize,2):
            #Establim noms de variables
            index=i
            x0=vecpress[index]
            y0=vecpress[index+1]
            x1=vecrelease[index]
            y1=vecrelease[index+1]
            #Calculem quants pasos mesura cada línea 
            elementx=np.absolute(np.int((x1-x0)/self.dx))
            elementy=np.absolute(np.int((y1-y0)/self.dx))
            
            #Segons quin tingui més o menys tamany, haurem d'establir el tamany
            #del qeu sigui més gran
            if elementx>elementy:
                contsize=elementx
            else:
                contsize=elementy
                
            #S'explica per si sol, per a que tinguin el mateix tamany.
            containery=np.linspace(y0,y1,contsize)
            containerx=np.linspace(x0,x1,contsize)
            #Bucle en el que guardem les dades en coordenades del discretitzat
            for m in range(contsize):
                containery[m]=np.rint((containery[m]+self.L)/self.dx)
                containerx[m]=np.rint((containerx[m]+self.L)/self.dx)
                                          
            transformvectorx=np.append(transformvectorx,containerx)
            transformvectory=np.append(transformvectory,containery)
            
         #I ara modifiquem el potencial
        elements=transformvectorx.size
        Vvec=self.Vvec
        #Bucle de modificació del potencial.
        for l in range(elements):
            x_i=np.int(transformvectorx[l])
            y_i=np.int(transformvectory[l])
            #PEnsa en el canvi matriu i coordenades no és intuitiu.
            Vvec[y_i,x_i]=1000000000000.

        #Modifiquem els respectius ac,vc
        self.Vvec=Vvec
        print(self.Vvec)
        self.avec,self.cvec=ck.ac(self.r,self.Vvec,self.Nx)
        
        print('Applied!')
    
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
        
    def compute_parameters(self,*largs):
        """Aquesta funció s'encarrega de calcular la norma i la energia del
        paquet en un cert instant t i despres l'ensenya a pantalla."""
        
        #Calculem moments i energia inicials i els posem al lloc corresponent
        self.normavec=ck.norma(self.psi0,self.Nx)
        
        self.norma0=ck.trapezis(self.xa,self.xb,self.dx,np.copy(self.normavec))
        self.energy0=ck.Ham(self.xa,self.xb,self.dx,np.copy(self.psi0))
        self.box3.normachange.text='Norma={0:.3f}'.format(self.norma0)
        self.box3.energychange.text='Energia={0:.3f}'.format(self.energy0)
                
        
        

        
   
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
        

        


def press(event):
    print('press released from test', event.x, event.y, event.button)
    global cordpress
    cordpress=np.array([[event.xdata,event.ydata]])

def release(event):
    global cordrelease
    cordrelease=np.array([[event.xdata,event.ydata]])


    





def keypress(event):
    print('key down', event.key)


def keyup(event):
    print('key up', event.key)


def motionnotify(event):
    print('mouse move to ', event.xdata, event.ydata)
    return event.xdata,event.ydata

def resize(event):
    print('resize from mpl ', event.width, event.height)


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
    PaquetApp().run()
    
    
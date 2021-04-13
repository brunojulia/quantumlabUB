# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:13:39 2020

@author: llucv
"""
import matplotlib
from kivy.app import App 
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.uix.screenmanager import FadeTransition, SlideTransition   
from kivy.uix.popup import Popup
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.core.text import LabelBase
from kivy.core.window import Window
from functools import partial
from kivy.graphics import Color, Ellipse,Line,Rectangle,InstructionGroup
from kivy.core.window import Window
from kivy.properties import ObjectProperty,ReferenceListProperty,\
    NumericProperty,StringProperty,ListProperty,BooleanProperty
from matplotlib.patches import Circle
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')

import numpy as np
import matplotlib.pyplot as plt

pi=np.pi
e=np.e

Nx_inty=0
Ny_inty=0
intensitat_llum=0

class qm_timelineApp(App):
    def build(self):
        self.title='quantum physics timeline'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('Discoveries').dcpseudo_init()
        self.get_screen("YoungSlit").yspseudo_init()
    pass

class StartingScreen(Screen):
    def __init__(self,**kwargs):
        super(StartingScreen,self).__init__(**kwargs)
        
    def transition_SD(self):
        """Transició des de la pantalla inicial a la pantalla Discoveries"""
        
        dcscreen=self.manager.get_screen('Discoveries')
        dcscreen.dc_schedule_fired()
        self.manager.transition=FadeTransition()
        self.manager.current='Discoveries'

class DiscoveriesScreen(Screen):
    def __init__(self,**kwargs):
        super(DiscoveriesScreen,self).__init__(**kwargs)
        
    def transition_DY(self):
        """Transició des de la pantalla Discoveries a la pantalla YoungSlit"""
        
        ysscreen=self.manager.get_screen('YoungSlit')
        ysscreen.ys_schedule_fired()
        self.manager.transition=FadeTransition()
        self.manager.current='YoungSlit'
        
    def dcpseudo_init(self):
        return
    
    def dc_schedule_fired(self):
        return

class YoungSlitScreen(Screen):
    slt_number = NumericProperty(0)
    is_source_on = BooleanProperty(False)
    
    def __init__(self,**kwargs):
        super(YoungSlitScreen,self).__init__(**kwargs)
        
    def yspseudo_init(self):
        
        #temps inicial (0, evidentment)
        self.t=0
        self.source_on=False
        
        #recinte i discretitzat
        self.dt=1/30
        self.dl=3*self.dt
        self.Nx=301
        self.Ny=301
        
        global Nx_inty,Ny_inty
        
        Nx_inty=self.Nx
        Ny_inty=self.Ny
        
        #visualització del recinte que apareix a l'App
        self.h_display=int(self.Ny/3)
        self.w_display=int(2*self.Nx/3)
        
        #variables de l'ona
        self.w=4*pi
        self.c=2
        self.amp=2.5
        self.tau=2*pi/self.w
        self.Ntau=int(1+self.tau/self.dt)
        self.rao=(self.c*self.dt/self.dl)**2
        print(self.rao)
        
        #variables paret
        self.sgm_max=0.02615
        self.m=1.54
        
        #llistes de l'ona i la paret en el recinte
        self.a=np.zeros((self.Nx,self.Ny,3))
        self.a2=np.zeros((self.Nx,self.Ny,self.Ntau))
        self.inty=np.zeros((self.Nx,self.Ny,3))
        
        self.sgm=np.zeros((self.Nx,self.Ny))
        self.sgm_wall=np.zeros((self.Nx,self.Ny))
        self.sgm_det=np.zeros((self.Nx,self.Ny))
        
        #font
        self.s=np.zeros((self.Nx,self.Ny))
        
        #slits
        self.Nslt=0
        self.w_slt=8
        
        #gruix i posició de les parets
        self.x_wall=int(self.Nx/5)
        self.w_wall=int(self.Nx/30)
        self.w_det=int(self.Nx/3)
        
        global w_det_ys
        w_det_ys=self.w_det        
        
        
        self.slit_presence=np.zeros((self.Nx,self.Ny))
        self.slit_presence[self.x_wall,:]=1
        
        self.separation = int(self.Ny/10)
        
        #coef d'absorció a les parets del detector
        for k in range(self.w_det):
            self.sgm_det[self.Nx-1-k,0+k:self.Ny-1-k]=\
                self.sgm_max*((self.w_det-k)/self.w_det)**self.m
            self.sgm_det[self.x_wall:self.Nx-k,k]=\
                self.sgm_max*((self.w_det-k)/self.w_det)**self.m
            self.sgm_det[self.x_wall:self.Nx-k,self.Ny-1-k]=\
                self.sgm_max*((self.w_det-k)/self.w_det)**self.m
        

        ##########################PRIMER DIBUIX###################
        
        #creem la figura recipient del plot imshow
        self.main_fig,self.axs=plt.subplots(2, sharex=True, sharey=True)
        
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas,1)
        
        
        #Plot de l'ona i de l'intensitat
        self.wave_visu=self.axs[0]
        self.inty_visu=self.axs[1]
        
        
        self.at=self.a[:,:,2]
        self.it=self.inty[:,:,2]
        
        #Dibuix de les figures
        self.cmap = plt.get_cmap('Reds')
        self.cmap.set_under('k', alpha=0)
        
        #diagrama d'ones
        self.wave_visu_im=self.wave_visu.imshow(\
            self.at.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=2*self.amp,vmin=-2*self.amp,origin='lower',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        #representació de la paret al diagrama d'ones
        self.wave_slit_im=self.wave_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        #diagrama d'intensitat
        self.inty_visu_im=self.inty_visu.imshow(\
            self.it.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=self.amp*self.amp*0.8,origin='lower',cmap='gray',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))

        #representació de la paret al diagrama d'intensitat
        self.inty_slit_im=self.inty_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))

        self.main_canvas.draw()
        
        self.main_canvas.mpl_connect('resize_event',partial(self.resize_kivy))
        
        
        
    def resize_kivy(self,*largs):
        """Aquesta funció és trucada cada cop que la finestra canvia de tamany.
        Cada cop que això passa, les coordenades (en pixels) dels cantons del plot 
        imshow canvien de lloc. Aquests són molt importants sobretot pel joc, per
        tant, cal tenir-nos ben localitzats."""
        #Aquesta funció de matplotlib ens transforma els cantons de coordenades Data
        # a display (pixels)
        self.wave_cantonsnew=self.wave_visu.transData.transform([(0,0),
                    (0,int(self.h_display*self.dl)),
                    (int(self.w_display*self.dl),int(self.h_display*self.dl)),
                    (int(self.w_display*self.dl),0)])
        
        self.inty_cantonsnew=self.inty_visu.transData.transform([(0,0),
                    (0,int(self.h_display*self.dl)),
                    (int(self.w_display*self.dl),int(self.h_display*self.dl)),
                    (int(self.w_display*self.dl),0)])
        
                
    def ys_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotysev,self.dt)
        
    #integral per trapezis, on h és el pas, i f, una llista amb els valors de 
    #la funció a inetgrar, és una llista 3D d'una funció 2D amb dependència temps
    #i com que estem integrant en el temps ens retorna una llista 2D
    def trapezis(self,h,f):
        val=(np.sum(f[:,:,0:-1],axis=2)+np.sum(f[:,:,1:],axis=2))*h/2
        return val
    
    #funció de la font
    def p_s(self,t,amp,w):
        val=amp*np.sin(w*t)
        return val
    
    #plot young slit evolution 
    def plotysev(self,dt):
        k=2
        if self.source_on==True:
            self.s[1,:]=self.p_s(self.t,self.amp,self.w)
        
        if self.source_on==False:
            self.s[:,:]=0
        
        
        """càlculs que fa el programa a cada pas"""
        
        
        slt_i=np.zeros((self.Nslt+2),dtype=int)
        slt_f=np.zeros((self.Nslt+2),dtype=int)
        slt_n=np.linspace(1,self.Nslt,self.Nslt,dtype=int)
        wall_presence=np.zeros((self.Ny),dtype=int)
        
        if self.Nslt==2:
            #posicio del final i l'inici de cada escletxa, ara amb separació variable
            slt_i[1]=int(self.Ny/2)-int(self.separation/2)-\
                int(self.w_slt/2)
            slt_i[2]=int(self.Ny/2)+int(self.separation/2)-\
                int(self.w_slt/2)
            slt_f[1]=int(self.Ny/2)-int(self.separation/2)+\
                int(self.w_slt/2)
            slt_f[2]=int(self.Ny/2)+int(self.separation/2)+\
                int(self.w_slt/2)

            slt_f[0]=0
            slt_f[self.Nslt+1]=self.Ny
            slt_i[self.Nslt+1]=self.Ny
        
        else:
            #posicio del final i l'inici de cada escletxa
            slt_i[1:self.Nslt+1]=int(self.Ny/2)-int(self.h_display/2)-\
                int(self.w_slt/2)+slt_n[:]*int(self.h_display/(1+self.Nslt))
            slt_f[1:self.Nslt+1]=int(self.Ny/2)-int(self.h_display/2)+\
                int(self.w_slt/2)+slt_n[:]*int(self.h_display/(1+self.Nslt))
            slt_f[0]=0
            slt_f[self.Nslt+1]=self.Ny
            slt_i[self.Nslt+1]=self.Ny
        
               
        #les escletxes van de splt_i a splt_f-1, en aquests punts no hi ha paret,
        # a slpt_f ja hi ha paret
        for n in range(1,self.Nslt+2):
            wall_presence[slt_f[n-1]:slt_i[n]]=1
            wall_presence[slt_i[n]:slt_f[n]]=0
        
        # matriu que, amb el gruix de la paret com a nombre de files, ens diu si 
        # hi ha paret o escletxes a cada una de les y(representades en les columnes)
        wall_presence=np.tile(np.array([wall_presence],dtype=int),
                              (self.w_wall,1))
    
        #matriu que diu com de "dins" som a la paret
        wall_n=np.linspace(1,self.w_wall,self.w_wall)
        wall_ny=np.tile(np.array([wall_n],dtype=int).transpose(),(1,self.Ny))
    
        #valors de coeficient d'absorció a les parets
        self.sgm_wall[self.x_wall-self.w_wall:self.x_wall,:]=wall_presence[:,:]\
                    *self.sgm_max*((wall_ny[:,:])/self.w_wall)**self.m
        
        #llista per a l'última capa de la paret, on l'amplitud d'ona és 0
        self.wave_presence=np.ones((self.Nx,self.Ny))
        self.wave_presence[self.x_wall,:]=(1-wall_presence[0,:])
        
        self.sgm=self.sgm_wall+self.sgm_det
        
        #resolució de l'equació d'ones a cada temps a l'interior del recinte
        self.a[1:-1,1:-1,k]=\
        (self.rao*(self.a[2:,1:-1,k-1]+self.a[0:-2,1:-1,k-1]\
        +self.a[1:-1,2:,k-1]+self.a[1:-1,0:-2,k-1]\
        -4*self.a[1:-1,1:-1,k-1])+self.s[1:-1,1:-1]\
        +2*self.a[1:-1,1:-1,k-1]-self.a[1:-1,1:-1,k-2]\
        +self.sgm[1:-1,1:-1]*self.a[1:-1,1:-1,k-2]/(2*self.dt))\
        /(1+self.sgm[1:-1,1:-1]/(2*self.dt))\
        *self.wave_presence[1:-1,1:-1]
        
        #condicions periòdiques de contorn a les parets superior i inferior
        self.a[1:self.x_wall,0,k]=\
        (self.rao*(self.a[2:self.x_wall+1,0,k-1]+self.a[0:self.x_wall-1,0,k-1]\
        +self.a[1:self.x_wall,1,k-1]+self.a[1:self.x_wall,self.Ny-1,k-1]\
        -4*self.a[1:self.x_wall,0,k-1])+self.s[1:self.x_wall,0]\
        +2*self.a[1:self.x_wall,0,k-1]-self.a[1:self.x_wall,0,k-2]\
        +self.sgm[1:self.x_wall,0]*self.a[1:self.x_wall,0,k-2]/(2*self.dt))\
        /(1+self.sgm[1:self.x_wall,0]/(2*self.dt))\
        *self.wave_presence[1:self.x_wall,0]
                    
        self.a[1:self.x_wall,self.Ny-1,k]=\
        (self.rao*(self.a[2:self.x_wall+1,self.Ny-1,k-1]\
        +self.a[0:self.x_wall-1,self.Ny-1,k-1]\
        +self.a[1:self.x_wall,0,k-1]+self.a[1:self.x_wall,self.Ny-2,k-1]\
        -4*self.a[1:self.x_wall,self.Ny-1,k-1])+self.s[1:self.x_wall,self.Ny-1]\
        +2*self.a[1:self.x_wall,self.Ny-1,k-1]\
        -self.a[1:self.x_wall,self.Ny-1,k-2]\
        +self.sgm[1:self.x_wall,self.Ny-1]\
        *self.a[1:self.x_wall,self.Ny-1,k-2]/(2*self.dt))\
        /(1+self.sgm[1:self.x_wall,self.Ny-1]/(2*self.dt))\
        *self.wave_presence[1:self.x_wall,self.Ny-1]
                    
        #calculs de la intensitat
        self.a2[:,:,:-1]=self.a2[:,:,1:]
        self.a2[:,:,self.Ntau-1]=self.a[:,:,k]*self.a[:,:,k]
        
        self.inty[:,:,k]=self.trapezis(self.dt,self.a2)/self.tau
        
        #preparació del següent pas
        self.a[:,:,:-1]=self.a[:,:,1:]
        self.t=self.t+self.dt
        
        self.slit_presence=1-self.wave_presence
        
        
        
        """Representació a cada pas"""
        
        #eliminar l'anterior
        self.wave_visu_im.remove()
        self.wave_slit_im.remove()
        self.inty_visu_im.remove()
        self.inty_slit_im.remove()
        
        #Posar el nou
        self.at=self.a[:,:,2]
        self.it=self.inty[:,:,2]
        
        global intensitat_llum
        intensitat_llum=self.it
        
        
        self.wave_visu_im=self.wave_visu.imshow(\
            self.at.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=2*self.amp,vmin=-2*self.amp,origin='lower',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
            
        self.wave_slit_im=self.wave_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
            
        self.inty_visu_im=self.inty_visu.imshow(\
            self.it.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=self.amp*self.amp*0.8,origin='lower',cmap='gray',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
            
        self.inty_slit_im=self.inty_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        #I utilitzar-lo
        self.main_canvas.draw()
        

    def ys_schedule_cancel(self):
        self.schedule.cancel()
        
    def add_slit(self):
        if self.Nslt<5:
            self.Nslt+=1
        else:
            pass
    
    def remove_slit(self):
        if self.Nslt>0:
            self.Nslt-=1
        else:
            pass
    
    def inc_slt_width(self):
        if (self.Nslt*self.w_slt)<self.h_display:
            self.w_slt+=2
        else:
            pass
    
    def dec_slt_width(self):
        if self.w_slt>0:
            self.w_slt-=2
        else:
            pass
        
    def change_source_state(self):
        self.t=0
        if self.source_on==False:
            self.source_on=True
        else:
            self.source_on=False
            
    def clear_waves(self):
        self.a=np.zeros((self.Nx,self.Ny,3))
    
    def inc_separation(self):
        if (self.Nslt*self.w_slt+self.separation)<self.h_display:
            self.separation+=2
        else:
            pass
    
    def dec_separation(self):
        if self.separation>0:
            self.separation-=2
        else:
            pass
        
        
class IntensityPopup(Popup):
    def __init__(self):
        super(Popup, self).__init__()
        
        global intensitat_llum,Nx_inty,Ny_inty,w_det_ys
        
        self.inty_fig=plt.figure()
        self.inty_canvas=FigureCanvasKivyAgg(self.inty_fig)
        self.box2.add_widget(self.inty_canvas,1)
        self.visu=plt.axes()
        self.inty_plot=self.visu.plot(intensitat_llum[int(Nx_inty-1.5*w_det_ys)
                                                ,w_det_ys:Ny_inty-w_det_ys])
        
        self.inty_canvas.draw()
        

        
if __name__ == '__main__':
    qm_timelineApp().run()
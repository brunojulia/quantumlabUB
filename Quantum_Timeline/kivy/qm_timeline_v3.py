# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 12:13:39 2020

@author: llucv
"""

from kivy.app import App 
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.uix.screenmanager import FadeTransition, SlideTransition   
from kivy.uix.popup import Popup
from functools import partial
from kivy.properties import ObjectProperty,ReferenceListProperty,\
    NumericProperty,StringProperty,ListProperty,BooleanProperty

import numpy as np
import matplotlib.pyplot as plt
import schro_gauss_packet as sch

pi=np.pi
e=np.e

Nx_inty=0
Ny_inty=0
intensitat_llum=0

class qm_timeline_v3App(App):
    def build(self):
        self.title='quantum physics timeline'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('Discoveries').dcpseudo_init()
        self.get_screen("YoungSlit").yspseudo_init()
        self.get_screen("SchroSlit").sspseudo_init()
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
        
    def transition_DSS(self):
        """Transició des de la pantalla Discoveries a la pantalla SchrodingerSlit"""
        
        ssscreen=self.manager.get_screen('SchroSlit')
        ssscreen.ss_schedule_fired()
        self.manager.transition=FadeTransition()
        self.manager.current='SchroSlit'
        
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
        self.frame_t=1/18
        
        #temps inicial (0, evidentment)
        self.t=0
        self.source_on=False
        
        #recinte i discretitzat
        self.dt=1/18
        self.dl=3*self.dt
        self.Nx=201
        self.Ny=201
        
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
        self.sgm_det=np.zeros((self.Nx,self.Ny))
        
        
        #font
        self.s=np.zeros((self.Nx,self.Ny))
        
        #slits
        self.Nslt=0
        self.w_slt=4
        
        #gruix i posició de les parets
        self.x_wall=int(self.Nx/5)
        self.w_wall=int(self.Nx/25)
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
                
        self.wall_presence=sch.young_slit_wall(self.Nslt,self.w_slt,
                                               self.separation,
                                               self.h_display,self.Ny)
        self.sgm_wall,self.wave_presence=sch.young_slit_sgm(self.wall_presence,
                                         self.w_wall,self.x_wall,
                                         self.sgm_max,self.m,
                                         self.Nx,self.Ny)
        

        ##########################PRIMER DIBUIX###################
        
        #creem la figura recipient del plot imshow
        self.main_fig,self.axs=plt.subplots(2)
        self.axs[0].set(xlim=[0, int(self.w_display*self.dl)], 
                        ylim=[0, int(self.h_display*self.dl)], 
                        adjustable='box',aspect=1)
        self.axs[1].set(xlim=[0, int(self.w_display*self.dl)], 
                        ylim=[0, int(self.h_display*self.dl)], 
                        adjustable='box',aspect=1)
        #self.axs[0].set_aspect

        
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
                interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        print(int(self.w_display*self.dl))
        
        #representació de la paret al diagrama d'ones
        self.wave_slit_im=self.wave_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
               interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        #diagrama d'intensitat
        self.inty_visu_im=self.inty_visu.imshow(\
            self.it.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
                ,vmax=self.amp*self.amp*0.8,origin='lower',cmap='gray',
                interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))

        #representació de la paret al diagrama d'intensitat
        self.inty_slit_im=self.inty_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
               interpolation='gaussian',
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
        self.schedule=Clock.schedule_interval(self.plotysev,self.frame_t)
        
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
               interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
            
        self.wave_slit_im=self.wave_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
               interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
            
        self.inty_visu_im=self.inty_visu.imshow(\
            self.it.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
        ,vmax=self.amp*self.amp*0.8,origin='lower',cmap='gray',
        interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
            
        self.inty_slit_im=self.inty_visu.imshow(\
            self.slit_presence.transpose()[int((self.Ny-self.h_display)/2):\
                                int((self.Ny+self.h_display)/2),
                                0:self.w_display]
               ,vmax=1,vmin=0.5,origin='lower',cmap=self.cmap,
               interpolation='gaussian',
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
        
        self.wall_presence=sch.young_slit_wall(self.Nslt,self.w_slt,
                                               self.separation,
                                               self.h_display,self.Ny)
        self.sgm_wall,self.wave_presence=sch.young_slit_sgm(self.wall_presence,
                                         self.w_wall,self.x_wall,
                                         self.sgm_max,self.m,
                                         self.Nx,self.Ny)
    
    def remove_slit(self):
        if self.Nslt>0:
            self.Nslt-=1
        else:
            pass
        
        self.wall_presence=sch.young_slit_wall(self.Nslt,self.w_slt,
                                               self.separation,
                                               self.h_display,self.Ny)
        self.sgm_wall,self.wave_presence=sch.young_slit_sgm(self.wall_presence,
                                         self.w_wall,self.x_wall,
                                         self.sgm_max,self.m,
                                         self.Nx,self.Ny)
    
    def inc_slt_width(self):
        if (self.Nslt*self.w_slt)<self.h_display:
            self.w_slt+=2
        else:
            pass
        
        self.wall_presence=sch.young_slit_wall(self.Nslt,self.w_slt,
                                               self.separation,
                                               self.h_display,self.Ny)
        self.sgm_wall,self.wave_presence=sch.young_slit_sgm(self.wall_presence,
                                         self.w_wall,self.x_wall,
                                         self.sgm_max,self.m,
                                         self.Nx,self.Ny)
    
    def dec_slt_width(self):
        if self.w_slt>0:
            self.w_slt-=2
        else:
            pass
        
        self.wall_presence=sch.young_slit_wall(self.Nslt,self.w_slt,
                                               self.separation,
                                               self.h_display,self.Ny)
        self.sgm_wall,self.wave_presence=sch.young_slit_sgm(self.wall_presence,
                                         self.w_wall,self.x_wall,
                                         self.sgm_max,self.m,
                                         self.Nx,self.Ny)
        
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
        
        self.wall_presence=sch.young_slit_wall(self.Nslt,self.w_slt,
                                               self.separation,
                                               self.h_display,self.Ny)
        self.sgm_wall,self.wave_presence=sch.young_slit_sgm(self.wall_presence,
                                         self.w_wall,self.x_wall,
                                         self.sgm_max,self.m,
                                         self.Nx,self.Ny)
    
    def dec_separation(self):
        if self.separation>0:
            self.separation-=2
        else:
            pass
        
        self.wall_presence=sch.young_slit_wall(self.Nslt,self.w_slt,
                                               self.separation,
                                               self.h_display,self.Ny)
        self.sgm_wall,self.wave_presence=sch.young_slit_sgm(self.wall_presence,
                                         self.w_wall,self.x_wall,
                                         self.sgm_max,self.m,
                                         self.Nx,self.Ny)
        
        
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

class SchroSlitScreen(Screen):
    slt_number = NumericProperty(0)
    is_setup_on = BooleanProperty(True)
    is_compute_on = BooleanProperty(False)
    Nx_ss = NumericProperty(300)
    Ny_ss = NumericProperty(150)
    showing_re = BooleanProperty(True)
    showing_im = BooleanProperty(True)
    
    def __init__(self,**kwargs):
        super(SchroSlitScreen,self).__init__(**kwargs)
        
    def sspseudo_init(self):
        self.frame_t=1/30
        self.compute=False
        self.psi_setup=True
        
        self.Nx=300
        self.Ny=150
        self.dt=1/20
        self.dl=np.sqrt(self.dt/2)
        self.r=self.dt/(8*self.dl**2)
    
        self.h_display=self.Ny
        self.w_display=self.Nx
        
        self.x0=int(self.Nx/4)
        self.y0=int(self.Ny/2)
        self.px0=0
        self.py0=0
        self.dev=7.5
        
        self.max_V=100
        self.x_wall=int(self.w_display/2)
        self.h_slits=int(self.h_display/2)
        self.Nslt=0
        self.separation=40
        self.w_slt=16
        self.devV=1
        
        self.psi=np.zeros((self.Nx,self.Ny),dtype=np.complex128)
        self.psi=sch.psi_0(self.Nx,self.Ny,
                            self.x0,self.y0,self.px0,self.py0,self.dev,
                            self.dl,self.dt)
        
        self.V=sch.Potential_slits_gauss(self.max_V,self.x_wall,self.h_slits,
                                         self.separation,self.w_slt,self.devV,
                                         self.Nslt,self.dl,self.Nx,self.Ny)
        self.pot=np.real(self.V)
        
        self.prob=sch.prob_dens(self.psi)
        self.prob_max=np.max(self.prob)
        
        self.im_psi=np.imag(self.psi)
        self.im_max=np.max(self.im_psi)
        self.im_min=np.min(self.im_psi)
        
        self.re_psi=np.real(self.psi)
        self.re_max=np.max(self.re_psi)
        self.re_min=np.min(self.re_psi)
        
        print(self.re_max)
        print(self.im_max)
        print(self.re_min)
        print(self.im_min)
        
                ##########################PRIMER DIBUIX###################
        self.re_show=1
        self.im_show=1
        
        #creem la figura recipient del plot imshow
        self.main_fig,self.axs=plt.subplots(2)
        self.axs[0].set(xlim=[0, int(self.w_display*self.dl)], 
                        ylim=[0, int(self.h_display*self.dl)], 
                        adjustable='box',aspect=1)
        self.axs[1].set(xlim=[0, int(self.w_display*self.dl)], 
                        ylim=[0, int(self.h_display*self.dl)], 
                        adjustable='box',aspect=1)
        #self.axs[0].set_aspect

        
        #L'associem amb kivy
        self.main_canvas=FigureCanvasKivyAgg(self.main_fig)
        self.box1.add_widget(self.main_canvas,1)
        
        
        #Plot de l'ona i de l'intensitat
        self.wave_visu=self.axs[0]
        self.prob_visu=self.axs[1]
        
        self.cmap = plt.get_cmap('Greys')
        self.cmap.set_under('k', alpha=0)
        
        
        self.wave_im=self.wave_visu.imshow(self.im_psi.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap="seismic",
             vmax=0.5,vmin=-0.5,alpha=self.im_show*0.5,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        
        self.wave_re=self.wave_visu.imshow(self.re_psi.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap="PuOr",
             vmax=0.5,vmin=-0.5,alpha=self.re_show*0.5,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        self.wave_slit=self.wave_visu.imshow(self.pot.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap=self.cmap,
             vmax=self.max_V,vmin=0.5*self.max_V,alpha=0.75,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        
        self.prob_prob=self.prob_visu.imshow(self.prob.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap="Blues",
             vmax=self.prob_max/2,vmin=0,alpha=1,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        self.prob_slit=self.prob_visu.imshow(self.pot.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap=self.cmap,
             vmax=self.max_V,vmin=0.5*self.max_V,alpha=0.75,
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
        
        self.prob_cantonsnew=self.prob_visu.transData.transform([(0,0),
                    (0,int(self.h_display*self.dl)),
                    (int(self.w_display*self.dl),int(self.h_display*self.dl)),
                    (int(self.w_display*self.dl),0)])
        
                
    def ss_schedule_fired(self):
        "Ara, definim l'update, a partir d'aqui farem l'animació"
        self.schedule=Clock.schedule_interval(self.plotssev,self.frame_t)
        
    def plotssev(self,dt):
        """ càlculs """
        
        if self.compute==True:
            self.psi=sch.psi_ev_ck(self.psi,self.V,self.r,self.dl,self.dt)
        
        else:
            pass
        
        self.pot=np.real(self.V)
        self.prob=sch.prob_dens(self.psi)
        self.im_psi=np.imag(self.psi)
        self.re_psi=np.real(self.psi)
        
        """ representació """
        self.wave_im.remove()
        self.wave_re.remove()
        self.wave_slit.remove()
        self.prob_slit.remove()
        self.prob_prob.remove()
        
        #new frame
        self.wave_im=self.wave_visu.imshow(self.im_psi.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap="seismic",
             vmax=0.5,vmin=-0.5,alpha=self.im_show*0.5,
             interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        self.wave_re=self.wave_visu.imshow(self.re_psi.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap="PuOr",
             vmax=0.3,vmin=-0.3,alpha=self.re_show*0.5,
             interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        self.wave_slit=self.wave_visu.imshow(self.pot.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap=self.cmap,
             vmax=self.max_V,vmin=0.5*self.max_V,alpha=0.75,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))

        
        
        self.prob_prob=self.prob_visu.imshow(self.prob.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap="Blues",
             vmax=self.prob_max/3,vmin=0,alpha=1,
             interpolation='gaussian',
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
        
        self.prob_slit=self.prob_visu.imshow(self.pot.transpose()\
            [int((self.Ny-self.h_display)/2):int((self.Ny+self.h_display)/2),
             0:self.w_display],origin='lower',cmap=self.cmap,
             vmax=self.max_V,vmin=0.5*self.max_V,alpha=0.75,
        extent=(0,int(self.w_display*self.dl),0,int(self.h_display*self.dl)))
            
        self.main_canvas.draw()
    
    def ss_schedule_cancel(self):
        self.schedule.cancel()
        
    def start_stop(self):
        if self.compute==True:
            self.compute=False
        
        else:
            self.compute=True
            self.psi=sch.psi_ev_ck(self.psi,self.V,self.r,self.dl,self.dt)
        
    
    def add_slit(self):
        if self.Nslt<5:
            self.Nslt+=1
        else:
            pass

        self.V=sch.Potential_slits_gauss(self.max_V,self.x_wall,self.h_slits,
                                         self.separation,self.w_slt,self.devV,
                                         self.Nslt,self.dl,self.Nx,self.Ny)

    def remove_slit(self):
        if self.Nslt>0:
            self.Nslt-=1
        else:
            pass
        
        self.V=sch.Potential_slits_gauss(self.max_V,self.x_wall,self.h_slits,
                                         self.separation,self.w_slt,self.devV,
                                         self.Nslt,self.dl,self.Nx,self.Ny)
    
    def inc_slt_width(self):
        if (self.Nslt*self.w_slt)<self.h_display:
            self.w_slt+=2
        else:
            pass
        
        self.V=sch.Potential_slits_gauss(self.max_V,self.x_wall,self.h_slits,
                                         self.separation,self.w_slt,self.devV,
                                         self.Nslt,self.dl,self.Nx,self.Ny)
    
    def dec_slt_width(self):
        if self.w_slt>0:
            self.w_slt-=2
        else:
            pass
        
        self.V=sch.Potential_slits_gauss(self.max_V,self.x_wall,self.h_slits,
                                         self.separation,self.w_slt,self.devV,
                                         self.Nslt,self.dl,self.Nx,self.Ny)
    
    def inc_separation(self):
        if (self.Nslt*self.w_slt+self.separation)<self.h_display:
            self.separation+=2
        else:
            pass
        
        self.V=sch.Potential_slits_gauss(self.max_V,self.x_wall,self.h_slits,
                                         self.separation,self.w_slt,self.devV,
                                         self.Nslt,self.dl,self.Nx,self.Ny)
    
    def dec_separation(self):
        if self.separation>0:
            self.separation-=2
        else:
            pass
        
        self.V=sch.Potential_slits_gauss(self.max_V,self.x_wall,self.h_slits,
                                         self.separation,self.w_slt,self.devV,
                                         self.Nslt,self.dl,self.Nx,self.Ny)
            
    def reset_psi(self):
        if self.psi_setup==False:
            self.psi_setup=True
            self.psi=sch.psi_0(self.Nx,self.Ny,
                                self.x0,self.y0,self.px0,self.py0,self.dev,
                                self.dl,self.dt)
            
        else:
            self.psi_setup=False

    
    def change_px0(self,value_px0,*largs):
        self.px0=value_px0
        self.psi=sch.psi_0(self.Nx,self.Ny,
                    self.x0,self.y0,self.px0,self.py0,self.dev,
                    self.dl,self.dt)
        
    def change_py0(self,value_py0,*largs):
        self.py0=value_py0
        self.psi=sch.psi_0(self.Nx,self.Ny,
                    self.x0,self.y0,self.px0,self.py0,self.dev,
                    self.dl,self.dt)
    
    def change_dev(self,value_dev,*largs):
        self.dev=value_dev
        self.psi=sch.psi_0(self.Nx,self.Ny,
                    self.x0,self.y0,self.px0,self.py0,self.dev,
                    self.dl,self.dt)

    def change_x0(self,value_x0,*largs):
        self.x0=value_x0
        self.psi=sch.psi_0(self.Nx,self.Ny,
                    self.x0,self.y0,self.px0,self.py0,self.dev,
                    self.dl,self.dt)
    
    def change_y0(self,value_y0,*largs):
        self.y0=value_y0
        self.psi=sch.psi_0(self.Nx,self.Ny,
                    self.x0,self.y0,self.px0,self.py0,self.dev,
                    self.dl,self.dt)
    
    def imag_visu(self):
        if self.im_show==0:
            self.im_show=1
        
        else:
            self.im_show=0
            
    def real_visu(self):
        if self.re_show==0:
            self.re_show=1
        
        else:
            self.re_show=0
            
          
        
        
if __name__ == '__main__':
    qm_timeline_v3App().run()
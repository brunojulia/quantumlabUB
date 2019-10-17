
import numpy as np
import matplotlib.pyplot as plt
from particle import *
from potentials import *

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty,ListProperty,NumericProperty,StringProperty
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle,Color,Ellipse,Line
from kivy.clock import Clock
from kivy.uix.popup import Popup
from matplotlib import cm
import pickle
import os
import time
from kivy.config import Config
Config.set('kivy','window_icon','ub.png')

L = 200


T = 30
dt = 0.1


class savewindow(FloatLayout):
    save = ObjectProperty(None)
    text_input = ObjectProperty(None)
    cancel = ObjectProperty(None)   
    
class loadwindow(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class main(BoxLayout):
    
    charge = 1.
    
    particles = []
    init_conds = []
    
    plot_texture = ObjectProperty()
    
    
    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.pot = Phi()
        self.set_texture()
        self.time = 0.
        self.T = 45
        self.speedindex = 3
        self.change_speed()
        self.running = False
        self.paused = False
        self.ready = False
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)
        self.previewlist = []
        self.progress = 0.
        
    def set_texture(self):
        L = 200
        dx = 1
        self.nx = int(L/dx)
        self.im = np.zeros((self.nx,self.nx),dtype=np.uint8)
        self.plot_texture = Texture.create(size=self.im.shape,colorfmt='luminance',bufferfmt='uint')
        
    def background(self):
        xx,yy, = np.meshgrid(np.linspace(-L/2.,L/2.,self.nx,endpoint=True),np.linspace(-L/2.,L/2.,self.nx,endpoint=True))
        self.im = np.zeros((self.nx,self.nx))
        if(self.pot.functions.size == 0):
            self.im = np.uint8(self.im)
        else:
            self.im = self.pot.val(xx,yy)
            self.im = self.im + np.abs(self.im.min())
            self.im = np.uint8(255.*(self.im/self.im.max()))
            
    def update_texture(self):
        L = 200
        dx = 1
        self.nx = int(L/dx)
        with self.plotbox.canvas:
            cx = self.plotbox.pos[0]
            cy = self.plotbox.pos[1]
            w = self.plotbox.size[0]
            h = self.plotbox.size[1]
            b = min(w,h)
            
            
            self.plot_texture.blit_buffer(self.im.reshape(self.im.size),colorfmt='luminance')
            Color(1.0,1.0,1.0)
            Rectangle(texture = self.plot_texture, pos = (cx,cy),size = (b,b))
            
        
    def update_pos(self,touch):
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/200.
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale

        if(self.menu.current_tab.text == 'Potentials'):
            self.param0slider.value = x
            self.param1slider.value = y
        if(self.menu.current_tab.text == 'Particles'):
            self.x0slider.value = x
            self.y0slider.value = y
            
    def update_angle(self,touch):
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/200.
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale
        xdif = x-self.x0slider.value
        ydif = y-self.y0slider.value
        if(np.abs(xdif)<0.01):
            if(ydif>0):
                angle = np.pi/2.
            else:
                angle = -np.pi/2.
        elif(xdif > 0 and ydif > 0):
            angle = np.arctan((ydif)/(xdif))
        elif(xdif < 0 and ydif < 0):
            angle = np.arctan((ydif)/(xdif)) + np.pi
        elif(xdif < 0 and ydif > 0):
            angle = np.arctan((ydif)/(xdif)) + np.pi
        else:
            angle = np.arctan((ydif)/(xdif)) + 2*np.pi
        if(np.abs(x) < 100. and np.abs(y) < 100.):   
#            if(self.partmenu.current_tab.text == 'Single'):
#                v = np.sqrt(self.vx0slider.value**2 + self.vy0slider.value**2)
#                self.vx0slider.value = round(v*np.cos(angle),0)
#                self.vy0slider.value = round(v*np.sin(angle),0)
                
            if(self.partmenu.current_tab.text == 'Dispersion'):
                self.thetaslider.value = int(round(angle*(180/np.pi),0))
                
            if(self.partmenu.current_tab.text == 'Line'):
                self.thetalslider.value = int(round(angle*(180/np.pi),0))
                
#            if(self.partmenu.current_tab.text == 'Free Part.'):
#                v = np.sqrt(self.vxfslider.value**2 + self.vyfslider.value**2)
#                self.vxfslider.value = round(v*np.cos(angle),1)
#                self.vyfslider.value = round(v*np.sin(angle),1)
            
    def add_pot_list(self):
        self.stop()
        if(self.potmenu.current_tab.text == 'Gauss'):
            self.pot.add_function(gauss,dgaussx,dgaussy,[self.param0slider.value,self.param1slider.value,self.param2gslider.value,self.param3gslider.value])
        elif(self.potmenu.current_tab.text == 'Woods-Saxon'):
            self.pot.add_function(woodsaxon,dwoodsaxonx,dwoodsaxony,[self.param0slider.value,self.param1slider.value,self.param2wsslider.value,self.param3wsslider.value/2.,self.param4wsslider.value/2.,self.param5wsslider.value])
        self.background()
        self.update_texture()
        
        self.ready = False
#        self.pcbutton.text = "Compute"
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
            
    def reset_pot_list(self):
        self.stop()
        self.pot.clear()
        self.plotbox.canvas.clear()
        self.background()
        
        self.ready = False
#        self.pcbutton.text = "Compute"
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
        
    def add_particle_list(self):
        self.stop()
        if(self.partmenu.current_tab.text == 'Single'):
            self.particles.append(Particle(self.massslider.value,self.charge,dt))
            self.init_conds.append([self.x0slider.value,self.y0slider.value,self.vx0slider.value,self.vy0slider.value])
            
            self.previewlist.append('Single')
            self.previewlist.append([self.x0slider.value,self.y0slider.value,self.vx0slider.value,self.vy0slider.value])
        elif(self.partmenu.current_tab.text == 'Dispersion'):
            
            delta = self.alphaslider.value/(self.nslider.value-1)
            theta = self.thetaslider.value - self.alphaslider.value/2.
            for k in range(0,int(self.nslider.value)):
                vx = self.vslider.value * np.cos(theta*(np.pi/180.))
                vy = self.vslider.value * np.sin(theta*(np.pi/180.))
                
                self.particles.append(Particle(self.massslider.value,self.charge,dt))
                self.init_conds.append([self.x0slider.value,self.y0slider.value,vx,vy])
                
                theta = theta + delta
            
            self.previewlist.append('Dispersion')
            self.previewlist.append([self.x0slider.value,self.y0slider.value,self.vslider.value,self.thetaslider.value,self.alphaslider.value])
        elif(self.partmenu.current_tab.text == 'Line'):
            
            delta = self.lslider.value/(self.nlslider.value-1)
            r = np.array([self.x0slider.value,self.y0slider.value]) - self.lslider.value*0.5*np.array([-np.sin(self.thetalslider.value*(np.pi/180.)),np.cos(self.thetalslider.value*(np.pi/180.))])
            
            vx = self.vlslider.value*np.cos(self.thetalslider.value*(np.pi/180.))
            vy = self.vlslider.value*np.sin(self.thetalslider.value*(np.pi/180.))
            for k in range(0,int(self.nlslider.value)):
                self.particles.append(Particle(self.massslider.value,self.charge,dt))
                self.init_conds.append([r[0],r[1],vx,vy])
                
                r = r + delta*np.array([-np.sin(self.thetalslider.value*(np.pi/180.)),np.cos(self.thetalslider.value*(np.pi/180.))])
            
            self.previewlist.append('Line')
            self.previewlist.append([self.x0slider.value,self.y0slider.value,self.nlslider.value,self.vlslider.value,self.thetalslider.value,self.lslider.value])
                
        elif(self.partmenu.current_tab.text == 'Free Part.'):
            
            x,y = acceptreject(int(self.nfslider.value),-100,100,1/np.sqrt(2*np.pi*self.sigfslider.value**2),freepart,[self.x0slider.value,self.y0slider.value,self.vxfslider.value*self.massslider.value,self.vyfslider.value*self.massslider.value,self.sigfslider.value])
            px,py = acceptreject(int(self.nfslider.value),-10,10,1/(np.sqrt(np.pi)),freepartp,[self.x0slider.value,self.y0slider.value,self.vxfslider.value*self.massslider.value,self.vyfslider.value*self.massslider.value,self.sigfslider.value])
            
            for i in range(0,int(self.nfslider.value)):
                self.particles.append(Particle(self.massslider.value,self.charge,dt))
                self.init_conds.append([x[i],y[i],px[i]/self.massslider.value,py[i]/self.massslider.value])  
                
            self.previewlist.append('Free Part.')
            self.previewlist.append([self.x0slider.value,self.y0slider.value,self.vxfslider.value,self.vyfslider.value,self.sigfslider.value])
            
        self.ready = False
#        self.pcbutton.text = "Compute"
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
        
            
    def reset_particle_list(self):
        self.stop()
        self.particles = []
        self.init_conds = []
        self.previewlist = []
        
        self.ready = False
#        self.pcbutton.text = "Compute"
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
    
    def playcompute(self):
        if(self.ready==False):
            self.statuslabel.text = 'Computing...'
            Clock.schedule_once(self.computation)

        elif(self.ready==True):
            if(self.running==False):
                self.timer = Clock.schedule_interval(self.animate,0.04)
                self.running = True
                self.paused = False
            elif(self.running==True):
                pass
            
    def computation(self,*args):
        print('---Computation Start---')
        self.progress = 0.
        start = time.time()
        for i,p in enumerate(self.particles,0):
            p.ComputeTrajectoryF(self.init_conds[i],self.T,self.pot)
            print('Particle ',i+1,' done')
            self.progress += 1
        
#        self.energycheck()
        print('---Computation End---')
        print('Exec time = ',time.time() - start)
        self.ready = True
#        self.pcbutton.text = "Play"
        self.pcbutton.background_normal = 'Icons/play.png'
        self.pcbutton.background_down = 'Icons/playb.png'
        self.statuslabel.text = 'Ready'
    
    def updateprogress(self,*args):
        val = (self.progress+1)/len(self.particles)
        self.progressbar.value = val*100
        
    def energycheck(self):
        ok = 0
        tol = 10**(-6)
        for i,p in enumerate(self.particles,0):
#            if(p.Energy().std() < tol):
            if(np.all(np.diff(p.Energy()) < tol)):
                ok += 1
        print('{} particles conserved the total energy up to {}'.format(ok,tol))
        print('{} particles did not conserve the total energy'.format(len(self.particles)-ok))
        
    def pause(self):
        if(self.running==True):
            self.paused = True
            self.timer.cancel()
            self.running = False
        else:
            pass
        
    def stop(self):
        self.pause()
        self.paused = False
        self.time = 0
        self.plotbox.canvas.clear()
        self.update_texture()
    
        
    def change_speed(self):
        sl = [1,2,5,10]
        if(self.speedindex == len(sl)-1):
            self.speedindex = 0
        else:
            self.speedindex += 1
        self.speed = sl[self.speedindex]
        self.speedbutton.text = str(self.speed)+'x'
    
    def save(self,path,name,comp=False):
        if(comp == False):
            self.particles = []
            self.init_conds = []
            self.previewlist = []
        
        savedata = np.array([self.pot.functions,self.pot.dfunctionsx,self.pot.dfunctionsy,self.particles,self.init_conds,self.previewlist])
        with open(os.path.join(path,name+'.dat'),'wb') as file:
            pickle.dump(savedata,file)
        self.dismiss_popup()
    
    def savepopup(self):
        content = savewindow(save = self.save, cancel = self.dismiss_popup)
        self._popup = Popup(title='Save File', content = content, size_hint=(1,1))
        self._popup.open()
    
    def load(self,path,name,demo=False):
        self.stop()
        with open(os.path.join(path,name[0]),'rb') as file:
            savedata = pickle.load(file)
        
        self.pot.functions = savedata[0]
        self.pot.dfunctionsx = savedata[1]
        self.pot.dfunctionsy = savedata[2]
        self.particles = savedata[3]
        self.init_conds = savedata[4]
        self.previewlist = savedata[5]
        if(len(self.particles) > 0):
            if(self.particles[0].steps.size>1):
                self.ready = True
#               self.pcbutton.text = "Play"
                self.pcbutton.background_normal = 'Icons/play.png'
                self.pcbutton.background_down = 'Icons/playb.png'
                self.statuslabel.text = 'Ready'
                print('Loaded simulation {} with computation'.format(name))
        else: 
            self.ready = False
#           self.pcbutton.text = "Compute"
            self.pcbutton.background_normal = 'Icons/compute.png'
            self.pcbutton.background_down = 'Icons/computeb.png'
            self.statuslabel.text = 'Not Ready'
            print('Loaded simulation {}'.format(name))
        
        self.background()
        self.update_texture()
        if(demo==False):
            self.dismiss_popup()
    
    def loadpopup(self):
        content = loadwindow(load = self.load, cancel = self.dismiss_popup)
        self._popup = Popup(title='Load File', content = content, size_hint=(1,1))
        self._popup.open()
    
    def dismiss_popup(self):
        self._popup.dismiss()
        
    def timeinversion(self):
        if(self.ready==True):
            self.pause()
            t = self.time
            self.stop()
            reversedpart = []
            reversedconds = []
            reversedpreview = []
            
            for p in self.particles:
                reversedpart.append(Particle(self.massslider.value,self.charge,dt))
                reversedconds.append([p.trax(t),p.tray(t),-p.travx(t),-p.travy(t)])
                reversedpreview.append('Single')
                reversedpreview.append([p.trax(t),p.tray(t),-p.travx(t),-p.travy(t)])
                
            self.particles = reversedpart
            self.init_conds = reversedconds
            self.previewlist = reversedpreview
            
            self.ready = False
#            self.pcbutton.text = "Compute"
            self.pcbutton.background_normal = 'Icons/compute.png'
            self.pcbutton.background_down = 'Icons/computeb.png'
            self.statuslabel.text = 'Not Ready'
        else:
            pass
        
    
    def preview(self,interval):
        if(self.running == False and self.paused == False):
            if(self.menu.current_tab.text == 'Particles'):
                if(self.partmenu.current_tab.text == 'Single'):
                    w = self.plotbox.size[0]
                    h = self.plotbox.size[1]
                    b = min(w,h)
                    scalew = b/200.
                    scaleh = b/200.
                    self.plotbox.canvas.clear()
                    self.update_texture()
                    with self.plotbox.canvas:
                        Color(1.0,0.5,0.0)
                        Ellipse(pos=(self.x0slider.value*scalew+w/2.-5.,self.y0slider.value*scaleh+h/2.-5.),size=(10,10))
                        Line(points=[self.x0slider.value*scalew+w/2.,self.y0slider.value*scaleh+h/2.,self.vx0slider.value*scalew+w/2.+self.x0slider.value*scalew,self.vy0slider.value*scalew+w/2.+self.y0slider.value*scalew])
                elif(self.partmenu.current_tab.text == 'Dispersion'):
                    w = self.plotbox.size[0]
                    h = self.plotbox.size[1]
                    b = min(w,h)
                    scalew = b/200.
                    scaleh = b/200.
                    
                    vx1 = self.vslider.value * np.cos((self.thetaslider.value - self.alphaslider.value/2.)*(np.pi/180.))
                    vy1 = self.vslider.value * np.sin((self.thetaslider.value - self.alphaslider.value/2.)*(np.pi/180.))
                    vx2 = self.vslider.value * np.cos((self.thetaslider.value + self.alphaslider.value/2.)*(np.pi/180.))
                    vy2 = self.vslider.value * np.sin((self.thetaslider.value + self.alphaslider.value/2.)*(np.pi/180.))
                    
                    self.plotbox.canvas.clear()
                    self.update_texture()
                    with self.plotbox.canvas:
                        Color(1.0,0.5,0.0)
                        Line(points=[self.x0slider.value*scalew+w/2.,self.y0slider.value*scaleh+h/2.,vx1*scalew+w/2.+self.x0slider.value*scalew,vy1*scalew+w/2.+self.y0slider.value*scalew])
                        Line(points=[self.x0slider.value*scalew+w/2.,self.y0slider.value*scaleh+h/2.,vx2*scalew+w/2.+self.x0slider.value*scalew,vy2*scalew+w/2.+self.y0slider.value*scalew])
                elif(self.partmenu.current_tab.text == 'Line'):
                    w = self.plotbox.size[0]
                    h = self.plotbox.size[1]
                    b = min(w,h)
                    scalew = b/200.
                    scaleh = b/200.
                    
                    r1 = np.array([self.x0slider.value,self.y0slider.value]) - self.lslider.value*0.5*np.array([-np.sin(self.thetalslider.value*(np.pi/180.)),np.cos(self.thetalslider.value*(np.pi/180.))])
                    r2 = np.array([self.x0slider.value,self.y0slider.value]) + self.lslider.value*0.5*np.array([-np.sin(self.thetalslider.value*(np.pi/180.)),np.cos(self.thetalslider.value*(np.pi/180.))])
                    r = r1
                    delta = self.lslider.value/(self.nlslider.value-1)
                    
                    vx = self.vlslider.value*np.cos(self.thetalslider.value*(np.pi/180.))
                    vy = self.vlslider.value*np.sin(self.thetalslider.value*(np.pi/180.))
                    
                    self.plotbox.canvas.clear()
                    self.update_texture()
                    with self.plotbox.canvas:
                        Color(1.0,0.5,0.0)
                        Line(points=[r1[0]*scalew+w/2.,r1[1]*scaleh+h/2.,r2[0]*scalew+w/2.,r2[1]*scaleh+h/2.])

                        for k in range(0,int(self.nlslider.value)):
                            Line(points =[r[0]*scalew+w/2.,r[1]*scaleh+h/2.,r[0]*scalew+w/2. + vx*scalew,r[1]*scaleh+h/2. + vy*scalew])
                            r = r + delta*np.array([-np.sin(self.thetalslider.value*(np.pi/180.)),np.cos(self.thetalslider.value*(np.pi/180.))])
                elif(self.partmenu.current_tab.text == 'Free Part.'):
                    w = self.plotbox.size[0]
                    h = self.plotbox.size[1]
                    b = min(w,h)
                    scalew = b/200.
                    scaleh = b/200.
                    self.plotbox.canvas.clear()
                    self.update_texture()
                    with self.plotbox.canvas:
                        Color(1.0,0.5,0.0)
                        Line(circle=(self.x0slider.value*scalew+w/2.,self.y0slider.value*scaleh+h/2.,self.sigfslider.value*scalew))
                        Line(points=[self.x0slider.value*scalew+w/2.,self.y0slider.value*scaleh+h/2.,self.vxfslider.value*scalew+w/2.+self.x0slider.value*scalew,self.vyfslider.value*scalew+w/2.+self.y0slider.value*scalew])
                    
                else:
                    self.plotbox.canvas.clear()
                    self.update_texture()
                    
            else:
                self.plotbox.canvas.clear()
                self.update_texture() 
                
            with self.plotbox.canvas:
                for i in range(0,len(self.previewlist),2):
                    if(self.previewlist[i] == 'Single'):
                        x0 = self.previewlist[i+1][0]
                        y0 = self.previewlist[i+1][1]
                        vx0 = self.previewlist[i+1][2]
                        vy0 = self.previewlist[i+1][3]
                        
                        w = self.plotbox.size[0]
                        h = self.plotbox.size[1]
                        b = min(w,h)
                        scalew = b/200.
                        scaleh = b/200.
                        
                        Color(0.0,0.0,1.0)
                        Ellipse(pos=(x0*scalew+w/2.-5.,y0*scaleh+h/2.-5.),size=(10,10))
                        Line(points=[x0*scalew+w/2.,y0*scaleh+h/2.,vx0*scalew+w/2.+x0*scalew,vy0*scalew+w/2.+y0*scalew])
                    if(self.previewlist[i] == 'Dispersion'):
                        x0 = self.previewlist[i+1][0]
                        y0 = self.previewlist[i+1][1]
                        v = self.previewlist[i+1][2]
                        theta = self.previewlist[i+1][3]
                        alpha = self.previewlist[i+1][4]
                    
                        w = self.plotbox.size[0]
                        h = self.plotbox.size[1]
                        b = min(w,h)
                        scalew = b/200.
                        scaleh = b/200.
                        
                        vx1 = v * np.cos((theta - alpha/2.)*(np.pi/180.))
                        vy1 = v * np.sin((theta - alpha/2.)*(np.pi/180.))
                        vx2 = v * np.cos((theta + alpha/2.)*(np.pi/180.))
                        vy2 = v * np.sin((theta + alpha/2.)*(np.pi/180.))
                        
                        with self.plotbox.canvas:
                            Color(0.0,0.0,1.0)
                            Line(points=[x0*scalew+w/2.,y0*scaleh+h/2.,vx1*scalew+w/2.+x0*scalew,vy1*scalew+w/2.+y0*scalew])
                            Line(points=[x0*scalew+w/2.,y0*scaleh+h/2.,vx2*scalew+w/2.+x0*scalew,vy2*scalew+w/2.+y0*scalew])
                    if(self.previewlist[i] == 'Line'):
                        x0 = self.previewlist[i+1][0]
                        y0 = self.previewlist[i+1][1]
                        n = self.previewlist[i+1][2]
                        v = self.previewlist[i+1][3]
                        theta = self.previewlist[i+1][4]
                        l = self.previewlist[i+1][5]
                        
                        w = self.plotbox.size[0]
                        h = self.plotbox.size[1]
                        b = min(w,h)
                        scalew = b/200.
                        scaleh = b/200.
                        
                        r1 = np.array([x0,y0]) - l*0.5*np.array([-np.sin(theta*(np.pi/180.)),np.cos(theta*(np.pi/180.))])
                        r2 = np.array([x0,y0]) + l*0.5*np.array([-np.sin(theta*(np.pi/180.)),np.cos(theta*(np.pi/180.))])
                        r = r1
                        delta = l/(n-1)
                        
                        vx = v*np.cos(theta*(np.pi/180.))
                        vy = v*np.sin(theta*(np.pi/180.))
                        with self.plotbox.canvas:
                            Color(0.0,0.0,1.0)
                            Line(points=[r1[0]*scalew+w/2.,r1[1]*scaleh+h/2.,r2[0]*scalew+w/2.,r2[1]*scaleh+h/2.])
    
                            for k in range(0,int(self.nlslider.value)):
                                Line(points =[r[0]*scalew+w/2.,r[1]*scaleh+h/2.,r[0]*scalew+w/2. + vx*scalew,r[1]*scaleh+h/2. + vy*scalew])
                                r = r + delta*np.array([-np.sin(theta*(np.pi/180.)),np.cos(theta*(np.pi/180.))])
                    if(self.previewlist[i] == 'Free Part.'):
                        x0 = self.previewlist[i+1][0]
                        y0 = self.previewlist[i+1][1]
                        vx = self.previewlist[i+1][2]
                        vy = self.previewlist[i+1][3]
                        sig = self.previewlist[i+1][4]
                        
                        w = self.plotbox.size[0]
                        h = self.plotbox.size[1]
                        b = min(w,h)
                        scalew = b/200.
                        scaleh = b/200.
                        with self.plotbox.canvas:
                            Color(0.0,0.0,1.0)
                            Line(circle=(x0*scalew+w/2.,y0*scaleh+h/2.,sig*scalew))
                            Line(points=[x0*scalew+w/2.,y0*scaleh+h/2.,vx*scalew+w/2.+x0*scalew,vy*scalew+w/2.+y0*scalew])
               
    def animate(self,interval):
        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scalew = b/200.
        scaleh = b/200.
        self.plotbox.canvas.clear()
        self.update_texture()
        with self.plotbox.canvas:
            for p in self.particles: 
                Color(1.0,0.0,0.0)
                Ellipse(pos=(p.trax(self.time)*scalew+w/2.-5.,p.tray(self.time)*scaleh+h/2.-5.),size=(10,10))
        
        self.time += interval*self.speed
        if(self.time >= self.T):
            self.time = 0.

    

            
class simApp(App):

    def build(self):
        return main()


if __name__ == '__main__':
    simApp().run()
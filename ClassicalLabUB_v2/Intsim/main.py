# -*- coding: utf-8 -*-
"""
Jofre Vallès Muns, Març 2020
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from physystem import *

#Kivy imports
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty
from kivy.graphics import Rectangle,Color,Ellipse,Line #Used to draw
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg #Used to do matplotlib plots
from kivy.clock import Clock
from kivy.uix.popup import Popup

#This two lines should set the icon of the application to an ub logo
#but only works rarely
from kivy.config import Config
Config.set('kivy','window_icon','ub.png')

class main(BoxLayout):
    
    charge = 1.
    
    particles = np.array([])

    plot_texture = ObjectProperty()
    
    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.time = 0.
        #Here you can modify the time of computation and the step
        self.T = 540
        self.dt = 0.01

        #Initialization of the speed button
        self.speedindex = 3
        self.change_speed()

        #Set flags to False
        self.running = False #Checks if animation is running
        self.paused = False #Checks if animation is paused
        self.ready = False #Checks if computation is done
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)#Always, runs, shows previews
        self.previewlist = []
        self.progress = 0.
        
        #Here you can modify the units of the simulation as well as the size of the box.
        self.V0 = 0.01 #eV
        self.R = 3.405 #A
        self.L = 200. #A
        self.M = 0.04 #kg/mol

    def update_pos(self,touch):
        """This function updates the position parameters
        when you click the screen"""

        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale

        if(self.menu.current_tab.text == 'Particles'):
            self.x0slider.value = x
            self.y0slider.value = y
            
    def update_angle(self,touch):
        """This function sets the theta angle of the
        single particle addition mode (which is not really used
        for this program) when clicking and dragging"""

        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
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
            if(self.partmenu.current_tab.text == 'Single'):
                self.thetasslider.value = int(round(angle*(180/np.pi),0))


    def add_particle_list(self):

        self.stop() #I stop the simultion to avoid crashes

        #Check in which mode the user is
        if(self.partmenu.current_tab.text == 'Single'):
            vx = self.vsslider.value * np.cos(self.thetasslider.value*(np.pi/180.))
            vy = self.vsslider.value * np.sin(self.thetasslider.value*(np.pi/180.))

            self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([self.x0slider.value,self.y0slider.value])/self.R,np.array([vx,vy]),2))
            
            self.previewlist.append('Single')
            self.previewlist.append([self.x0slider.value,self.y0slider.value,vx,vy])
        elif(self.partmenu.current_tab.text == 'Random Lattice'):
            #Initialization process for the lattice, see documentation for the operations in this part

            n = int(self.nrslider.value)
            x,y = np.linspace(-self.L/2*0.8,self.L/2*0.8,n),np.linspace(-self.L/2*0.8,self.L/2*0.8,n)
            vmax = 10
            temp = 2.5
            
            temp = 3.
            theta = np.random.ranf(n**2)*2*np.pi
            vx,vy = 0.5*np.cos(theta),0.5*np.sin(theta)
            
            vcm = np.array([np.sum(vx),np.sum(vy)])/n**2
            kin = np.sum(vx**2+vy**2)/(n**2)
            
            vx = (vx-vcm[0])*np.sqrt(2*temp/kin)
            vy = (vy-vcm[1])*np.sqrt(2*temp/kin)
            k = 0
            for i in range(0,n):
                for j in range(0,n):
                    self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([x[i],y[j]])/self.R,np.array([vx[k],vy[k]]),2))
                
                    self.previewlist.append('Single')
                    self.previewlist.append([x[i],y[j],vx[k]*self.R,vy[k]*self.R])
                    k += 1

        #This block of code is present at different points in the program
        #It updates the ready flag and changes the icons for compute/play button and the status label.
        self.ready = False
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
        
        V = np.sqrt(vx**2+vy**2)
        self.histax.set_xlabel('v')
        self.histax.set_xlim([0,V.max()])
        self.histax.set_ylim([0,1]) 
            
        self.histax.hist(V,bins=np.arange(0, V.max() + 1, 1),density=True)
        self.histcanvas.draw()
        
            
    def reset_particle_list(self):
        #Empties particle list
        self.stop()
        self.particles = np.array([])
        self.previewlist = []
        
        self.ready = False
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'
    
    def playcompute(self):
        #If ready is false it starts the computation, if true starts the animation
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
        #Computation process
        print('---Computation Start---')

        start = time.time()

        self.s = PhySystem(self.particles,[self.V0,self.R,self.L/self.R])
        self.s.solveverlet(self.T,self.dt)

        print('---Computation End---')
        print('Exec time = ',time.time() - start)

        self.ready = True
        self.pcbutton.background_normal = 'Icons/play.png'
        self.pcbutton.background_down = 'Icons/playb.png'
        self.statuslabel.text = 'Ready'
        
        #This also saves the temperatures and energies to files
        np.savetxt('Kenergy.dat',self.s.K,fmt='%10.5f')
        np.savetxt('Uenergy.dat',self.s.U,fmt='%10.5f')
        np.savetxt('Tenergy.dat',self.s.K + self.s.U,fmt='%10.5f')
        np.savetxt('Temps.dat',self.s.T,fmt='%10.5f')
        
        
        
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
        
    
        
    def change_speed(self):
        #This simply cicles the sl list with the speed multipliers, self.speed is later
        #used to speed up the animation
        sl = [1,2,5,10]
        if(self.speedindex == len(sl)-1):
            self.speedindex = 0
        else:
            self.speedindex += 1
        self.speed = sl[self.speedindex]
        self.speedbutton.text = str(self.speed)+'x'
    
    #Saving and loading processes and popups, the kivy documentation
    #has a good explanation on the usage of the filebrowser widget and the
    #process of creating popups in general.
    def save(self,path,name):
        #I put all the relevant data in a numpy array and save it with pickle
        #The order is important for the loading process.
        savedata = np.array([self.s,self.T,self.dt,self.L,self.previewlist])
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
        
        self.s = savedata[0]
        self.T = savedata[1]
        self.dt = savedata[2]
        self.L = savedata[3]
        self.previewlist = savedata[4]
        
        
        self.ready = True
        self.pcbutton.background_normal = 'Icons/play.png'
        self.pcbutton.background_down = 'Icons/playb.png'
        self.statuslabel.text = 'Ready'
        print('Loaded simulation {} with computation'.format(name))
        if(demo==False):
            self.dismiss_popup()
    
    def loadpopup(self):
        content = loadwindow(load = self.load, cancel = self.dismiss_popup)
        self._popup = Popup(title='Load File', content = content, size_hint=(1,1))
        self._popup.open()
        


    def plotpopup(self):
        """This plotpopu show the energy plots on a popup when the giant 'Energy' button
        on the UI is pressed, this was originally and experiment and I ran out of time to 
        change it. It should be done like the histograms and embed the FigureCanvasKivyAgg in
        the UI directly"""
        self.eplot = Figure()
        t = np.arange(self.dt,self.T+self.dt,self.dt)
        ax = self.eplot.add_subplot(111)
        
        ax.plot(t,self.s.K,'r-',label = 'Kinetic Energy')
        ax.plot(t,self.s.U,'b-',label = 'Potential Energy')
        ax.plot(t,self.s.K+self.s.U,'g-',label = 'Total Energy')
#        plt.plot(t,self.s.Kmean,'g-',label = 'Mean Kinetic Energy')
        ax.legend(loc=1)
        ax.set_xlabel('t')
        
        self.ecanvas = FigureCanvasKivyAgg(self.eplot)
        content = self.ecanvas
        self._popup = Popup(title ='Energy conservation',content = content, size_hint=(0.9,0.9))
        self._popup.open()
    
    def dismiss_popup(self):
        self._popup.dismiss()
        
                
            
    def timeinversion(self):
#        TO FIX
        """This function comes from the other program and is linked to the time inversion button.
        Right now the button won't do anything because I have emptied this function. If you delete this
        Function the program will crash if you press the buttons. If you ask why I haven't deleted
        the button is because it would mess up the aspect ratio of the icons"""
        pass
        
    
    def preview(self,interval):
        """Draws the previews of the particles when the animation is not running or before adding
        the preview of the lattice mode before adding is not programmed (mainly because it is a random process)"""
        if(self.running == False and self.paused == False):
            if(self.menu.current_tab.text == 'Particles'):
                if(self.partmenu.current_tab.text == 'Single'):
                    w = self.plotbox.size[0]
                    h = self.plotbox.size[1]
                    b = min(w,h)
                    scale = b/self.L
                    self.plotbox.canvas.clear()
                    
                    vx = self.vsslider.value * np.cos(self.thetasslider.value*(np.pi/180.))
                    vy = self.vsslider.value * np.sin(self.thetasslider.value*(np.pi/180.))
                    
                    with self.plotbox.canvas:
                        Color(1.0,0.5,0.0)
                        Ellipse(pos=(self.x0slider.value*scale+w/2.-self.R*scale/2.,self.y0slider.value*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                        Line(points=[self.x0slider.value*scale+w/2.,self.y0slider.value*scale+h/2.,vx*scale+w/2.+self.x0slider.value*scale,vy*scale+w/2.+self.y0slider.value*scale])
                else:
                    self.plotbox.canvas.clear()
                    
                    
            else:
                self.plotbox.canvas.clear()
                 
                
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
                        scale = b/self.L
                        
                        Color(0.0,0.0,1.0)
                        Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                        Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])
               
    def animate(self,interval):
        """Draw all the particles for the animation"""

        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        self.plotbox.canvas.clear()
        
        N = self.s.particles.size
        n = int(self.T/self.dt)
        i = int(self.time/self.dt)
        delta = 5./self.dt

        with self.plotbox.canvas:
            for j in range(0,N): 
                Color(1.0,0.0,0.0)
                Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
        
        self.time += interval*self.speed #Here is where speed accelerates animation
        self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.
        
        self.acucounter += 1
        
        if(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
            vs = np.linspace(0,self.s.V.max(),100)
            
            self.histax.clear()
            self.histax.set_xlabel('v')
            self.histax.set_xlim([0,self.s.V.max()])
            self.histax.set_ylim([0,np.ceil(self.s.MB.max())])
            
        
            self.histax.hist(self.s.V[i,:],bins=np.arange(0, self.s.V.max() + 1, 1),density=True)
            self.histax.plot(vs,self.s.MB[i,:],'r-')
            self.histcanvas.draw()
        
        
        if(self.plotmenu.current_tab.text == 'Acu' and self.time>40.): #Accumulated momentum histogram
            vs = np.linspace(0,self.s.V.max(),100)
            
            
            self.acuhistax.clear()
            self.acuhistax.set_xlabel('v')
            self.acuhistax.set_xlim([0,self.s.V.max()])
            self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])
            
        
            self.acuhistax.hist(self.s.Vacu[int((i-int(n - (40./self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.2),density=True)
            self.acuhistax.plot(vs,self.s.MBacu[int((i-int(n - (40./self.dt)))/delta)],'r-')
            self.acuhistcanvas.draw()
                

        """This block of code is for building the accumulated histograms as the animation progresses, this
        is extremely slow and will slowdown the program, I will leave it if you want to take a look at it."""


#        if(self.plotmenu.current_tab.text == 'Acu'):
#            vs = np.linspace(0,self.s.V.max(),100)
#            
#            
#            self.acuhistax.clear()
#            self.acuhistax.set_xlabel('v')
#            self.acuhistax.set_xlim([0,self.s.V.max()])
#            self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])
#            
#        
#            self.acuhistax.hist(self.Vacu,bins=np.arange(0, self.s.V.max() + 0.5, 0.5),density=True)
#            self.acuhistax.plot(vs,self.MBacu,'r-')
#            self.acuhistcanvas.draw()
            
#        if(self.time>self.T/2. and self.acucounter == int(0.4/self.dt)):
#            print('hola')
#            self.Vacu = np.append(self.Vacu,self.s.V[i,:])
#            Temp = np.sum(self.Vacu**2)/(self.Vacu.size - 2)
#            self.MBacu = (vs/(Temp)*np.exp(-vs**2/(2*Temp)))    
#            print(self.Vacu.shape)
#            
#        if(self.acucounter >= int(1./self.dt)):
#            self.acucounter = 0 
#        
#        if(self.time >= self.T):
#            self.time = 0.
#            self.Vacu = np.array([])

    

            
class intsimApp(App):

    def build(self):
        return main()


if __name__ == '__main__':
    intsimApp().run()  
# -*- coding: utf-8 -*-
"""
Editat i continuat per Jofre Vallès Muns, Març 2020
del codi de l'Arnau Jurado Romero.

Els comentaris en anglès corresponen als de l'Arnau,
els que estan en català són els d'en Jofre
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
from kivy.core.window import Window #Per canviar el color de fons

import time

#This two lines should set the icon of the application to an ub logo
#but only works rarely
from kivy.config import Config
Config.set('kivy','window_icon','ub.png')

class main(BoxLayout):

    charge = 1.

    particles = np.array([])

    plot_texture = ObjectProperty()

    #Definition for the matplotlib figures of the histograms
    #Histograma del moment
    hist = Figure()
    histax = hist.add_subplot(111, xlabel='v', ylabel = 'Number of particles relative')
    histax.set_xlim([0,1])
    histax.set_ylim([0,25])
    hist.subplots_adjust(0.125,0.19,0.9,0.9) #Ajustem el subplot perquè es pugui veure bé els eixos i els seus labels
    histax.yaxis.labelpad = 10 #Ajustem els eixos per acabar d'afinar la posició que ens interessa
    histax.xaxis.labelpad = -0.5
    histcanvas = FigureCanvasKivyAgg(hist)

    #Histograma del moment acumulat
    acuhist = Figure()
    acuhistax = acuhist.add_subplot(111, xlabel='v', ylabel = 'Number of particles relative')
    acuhistax.set_xlim([0,1])
    acuhistax.set_ylim([0,25])
    acuhist.subplots_adjust(0.125,0.19,0.9,0.9)
    acuhistax.yaxis.labelpad = 10
    acuhistax.xaxis.labelpad = -0.5
    acuhistcanvas = FigureCanvasKivyAgg(acuhist)

    #(Sub)Plot de l'energia
    enplot = Figure()
    enplotax = enplot.add_subplot(111, xlabel='t', ylabel = 'Energy')
    enplotax.set_xlim([0,10]) #Recordar canviar eixos al canviar el temps que computa
    enplotax.set_ylim([0,25])
    enplot.subplots_adjust(0.125,0.19,0.9,0.9)
    enplotax.yaxis.labelpad = 10
    enplotax.xaxis.labelpad = -0.5
    enplotcanvas = FigureCanvasKivyAgg(enplot)

    #These are for a different method of accumulation (see comments animation function)
    Vacu = np.array([])
    MBacu = np.zeros(100)
    acucounter = 0

    #Per canviar el fons de pantalla (de la part de les boles)
#    Window.clearcolor = (0.15, 0, 0.3, 1)

    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.time = 0.
        #Here you can modify the time of computation and the step
        self.T = 50
        self.dt = 0.01

        #Initialization of the speed button
        self.speedindex = 3
        self.change_speed()

        #Set flags to False
        self.running = False #Checks if animation is running
        self.paused = False #Checks if animation is paused
        self.ready = False #Checks if computation is done
        self.previewtimer = Clock.schedule_interval(self.preview,0.04)#Always, runs, shows previews. Crida la funció preview cada 0.04 segons
        self.previewlist = []
        self.progress = 0.
        self.submenu = 'Random Lattice' #Comencem al submenú de "random lattice"
        self.n = 1 #Modificar aquest valor si es vol que a l'obrir el programa ja hi hagi més d'una partícula
        self.n1 = 1 #Modificar aquest valor si es vol que a l'obrir el submenú subsystems ja hi hagi més d'una partícula al subsistema 1
        self.n2 = 1 #Ídem pel subsistema 2

        #Initialization of histogram plots i l'energia
        self.histbox.add_widget(self.histcanvas)
        self.acuhistbox.add_widget(self.acuhistcanvas)
        self.enplotbox.add_widget(self.enplotcanvas)

        #Here you can modify the units of the simulation as well as the size of the box.
        self.V0 = 0.01 #eV
        self.R = 3.405 #A
        self.L = 200. #A
        self.M = 0.04 #kg/mol

        #Això ho fem perquè així el programa s'obri ja amb la simulació d'una partícula i no 0
        self.add_particle_list()

    def update_pos(self,touch):
        """This function updates the position parameters
        when you click the screen"""

        w = self.plotbox.size[0]
        h = self.plotbox.size[1]
        b = min(w,h)
        scale = b/self.L
        x = (touch.pos[0] - b/2.)/scale
        y = (touch.pos[1] - b/2.)/scale

    def on_touch_Slider(self):
        """Funció cridada quan Slider detecta un click, voldrem que afageixi una partícula sempre
        i quan el nombre de partícules canvii, si no, ho interpretarem com un click accidental
        (això ho fem perquè sovint s'emeten "clicks" quan realment no volem canviar el valor"""

        if(self.partmenu.current_tab.text == 'Random Lattice'):
            if(self.nrslider.value == self.n):
                pass
            else:
                self.add_particle_list()
        elif(self.partmenu.current_tab.text == 'Subsystems'):
            if(self.n1slider.value == self.n1 and self.n2slider.value == self.n2):
                  pass
            else:
                  self.add_particle_list()

    def on_touch_Submenu(self):
        """Funció cridada quan els botons de submenú detecten un click, voldrem que borri les partícules
        que hi ha a la pantalla per posar les adients al nou submenú"""

        if(self.partmenu.current_tab.text == 'Random Lattice' and self.submenu == 'Random Lattice'):
            pass
        elif(self.partmenu.current_tab.text == 'Random Lattice' and self.submenu == 'Subsystems'):
            self.submenu = 'Random Lattice'
            self.add_particle_list()
        elif(self.partmenu.current_tab.text == 'Subsystems' and self.submenu == 'Subsystems'):
            pass
        elif(self.partmenu.current_tab.text == 'Subsystems' and self.submenu == 'Random Lattice'):
            self.submenu = 'Subsystems'
            self.add_particle_list()

    def add_particle_list(self):

        self.stop() #I stop the simultion to avoid crashes

        self.reset_particle_list();

        #Fem check de la part del menú que som
        if(self.partmenu.current_tab.text == 'Random Lattice'):
            self.n = int(self.nrslider.value)
            x,y = np.linspace(-self.L/2*0.8,self.L/2*0.8,self.n),np.linspace(-self.L/2*0.8,self.L/2*0.8,self.n)
            vmax = 10
            temp = 2.5

            temp = 3.
            theta = np.random.ranf(self.n**2)*2*np.pi
            vx,vy = 0.5*np.cos(theta),0.5*np.sin(theta)

            vcm = np.array([np.sum(vx),np.sum(vy)])/self.n**2
            kin = np.sum(vx**2+vy**2)/(self.n**2)

            if(self.n == 1): #Pel cas d'una sola partícula, no es complirà que la velocitat del cm és 0
                vx = vx*np.sqrt(2*temp/kin)
                vy = vy*np.sqrt(2*temp/kin)
            else:
                vx = (vx-vcm[0])*np.sqrt(2*temp/kin)
                vy = (vy-vcm[1])*np.sqrt(2*temp/kin)
            k = 0
            for i in range(0,self.n):
                for j in range(0,self.n):
                    #A "particles", tenim les posicions i les velocitats en unitats de kivy (les velocitats ja les teníem així,
                    #però per les posicions les haurem de treure aquest factor)
                    self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([x[i],y[j]])/self.R,np.array([vx[k],vy[k]]),2))

                    #En aquesta altra array tindrem les posicions i les velocitats amb les unitats que toca (Angstrom)
                    self.previewlist.append([x[i],y[j],vx[k]*self.R,vy[k]*self.R])
                    k += 1

        elif(self.partmenu.current_tab.text == 'Subsystems'):
            self.n1 = int(self.n1slider.value)
            self.n2 = int(self.n2slider.value)

            x1,y1 = np.linspace(-self.L/2*0.8,-self.L/2*0.1,self.n1),np.linspace(-self.L/2*0.8,self.L/2*0.8,self.n1)
            x2,y2 = np.linspace(self.L/2*0.1,self.L/2*0.8,self.n2),np.linspace(-self.L/2*0.8,self.L/2*0.8,self.n2)

            vmax1 = 10
            temp1 = 3
            theta1 = np.random.ranf(self.n1**2)*2*np.pi
            vx1,vy1 = 0.5*np.cos(theta1),0.5*np.sin(theta1)
            vcm1 = np.array([np.sum(vx1),np.sum(vy1)])/self.n1**2
            kin1 = np.sum(vx1**2+vy1**2)/(self.n1**2)

            if(self.n1 == 1): #Pel cas d'una sola partícula, no es complirà que la velocitat del cm és 0
                vx1 = vx1*np.sqrt(2*temp1/kin1)
                vy1 = vy1*np.sqrt(2*temp1/kin1)
            else:
                vx1 = (vx1-vcm1[0])*np.sqrt(2*temp1/kin1)
                vy1 = (vy1-vcm1[1])*np.sqrt(2*temp1/kin1)

            vmax2 = 10
            temp2 = 10
            theta2 = np.random.ranf(self.n2**2)*2*np.pi
            vx2,vy2 = 0.5*np.cos(theta2),0.5*np.sin(theta2)
            vcm2 = np.array([np.sum(vx2),np.sum(vy2)])/self.n2**2
            kin2 = np.sum(vx2**2+vy2**2)/(self.n2**2)

            if(self.n2 == 1): #Pel cas d'una sola partícula, no es complirà que la velocitat del cm és 0
                vx2 = vx2*np.sqrt(2*temp2/kin2)
                vy2 = vy2*np.sqrt(2*temp2/kin2)
            else:
                vx2 = (vx2-vcm2[0])*np.sqrt(2*temp2/kin2)
                vy2 = (vy2-vcm2[1])*np.sqrt(2*temp2/kin2)

            k = 0
            for i in range(0,self.n1):
                for j in range(0,self.n1):
                    #A "particles", tenim les posicions i les velocitats en unitats de kivy (les velocitats ja les teníem així,
                    #però per les posicions les haurem de treure aquest factor)
                    self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([x1[i],y1[j]])/self.R,np.array([vx1[k],vy1[k]]),2))

                    #En aquesta altra array tindrem les posicions i les velocitats amb les unitats que toca (Angstrom)
                    self.previewlist.append([x1[i],y1[j],vx1[k]*self.R,vy1[k]*self.R])
                    k += 1

            k = 0
            for i in range(0,self.n2):
                for j in range(0,self.n2):
                    #A "particles", tenim les posicions i les velocitats en unitats de kivy (les velocitats ja les teníem així,
                    #però per les posicions les haurem de treure aquest factor)
                    self.particles = np.append(self.particles,particle(self.massslider.value,self.charge,np.array([x2[i],y2[j]])/self.R,np.array([vx2[k],vy2[k]]),2))

                    #En aquesta altra array tindrem les posicions i les velocitats amb les unitats que toca (Angstrom)
                    self.previewlist.append([x2[i],y2[j],vx2[k]*self.R,vy2[k]*self.R])
                    k += 1

        #This block of code is present at different points in the program
        #It updates the ready flag and changes the icons for compute/play button and the status label.
        self.ready = False
        self.pcbutton.background_normal = 'Icons/compute.png'
        self.pcbutton.background_down = 'Icons/computeb.png'
        self.statuslabel.text = 'Not Ready'

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

        #Creem una classe PhySystem passant-li l'array de "particles" i com a paràmetres les unitats del sistema
        self.s = PhySystem(self.particles,[self.V0,self.R,self.L/self.R])
        #Posem el +10 perquè generi uns quants valors més dels que reproduirem, per poder aturar-ho quan toqui.
        #És una mica poc elegant, però funciona. Revisar en un futur, es podria corregir segurament modificant els passos de temps
        #Si es modifica com hem dit anteriorment, corregir la part de la representació d'energia en la funció animate()
        self.s.solveverlet(self.T+10,self.dt)

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
                #Afegir condicions per quan hi hagi diferents submenús, de moment no cal

#            else:
                self.plotbox.canvas.clear()

            with self.plotbox.canvas:
                if(self.partmenu.current_tab.text == 'Random Lattice'):
                    for i in range(0,len(self.previewlist),1):
                        x0 = self.previewlist[i][0]
                        y0 = self.previewlist[i][1]
                        vx0 = self.previewlist[i][2]
                        vy0 = self.previewlist[i][3]

                        w = self.plotbox.size[0]
                        h = self.plotbox.size[1]
                        b = min(w,h)
                        scale = b/self.L

                        Color(0.34,0.13,1.0)
                        Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                        Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])

                elif(self.partmenu.current_tab.text == 'Subsystems'):
                    for i in range(0,len(self.previewlist),1):
                        if (i < (self.n1)**2):
                            x0 = self.previewlist[i][0]
                            y0 = self.previewlist[i][1]
                            vx0 = self.previewlist[i][2]
                            vy0 = self.previewlist[i][3]

                            w = self.plotbox.size[0]
                            h = self.plotbox.size[1]
                            b = min(w,h)
                            scale = b/self.L

                            Color(0.34,0.13,1.0)
                            Ellipse(pos=(x0*scale+w/2.-self.R*scale/2.,y0*scale+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                            Line(points=[x0*scale+w/2.,y0*scale+h/2.,vx0*scale+w/2.+x0*scale,vy0*scale+w/2.+y0*scale])

                        else:
                            x0 = self.previewlist[i][0]
                            y0 = self.previewlist[i][1]
                            vx0 = self.previewlist[i][2]
                            vy0 = self.previewlist[i][3]

                            w = self.plotbox.size[0]
                            h = self.plotbox.size[1]
                            b = min(w,h)
                            scale = b/self.L

                            Color(0.035,0.61,0.17)
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

        #Això comprova si l'animació ha arribat al final de la reproducció, si és així, es para
        if(i >= n):
            self.stop()

        if(self.partmenu.current_tab.text == 'Random Lattice'):
            with self.plotbox.canvas:
                for j in range(0,N):
                    Color(1.0,0.0,0.0)
                    Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))

            self.time += interval*self.speed #Here is where speed accelerates animation
            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.

            self.acucounter += 1

            if(self.plotmenu.current_tab.text == 'Energy'): #Instantaneous momentum histogram
                #El +10 del temps és degut a que calculem 10 unitats de temps més de les que toca, mirar funció computation
                #És una mica poc elegant, però funciona. Revisar en un futur, es podria corregir segurament modificant els passos de temps
                t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                self.enplotax.clear()
                self.enplotax.set_xlabel('t')
                self.enplotax.set_ylabel('Energy')

                self.enplotax.set_xlim([0,self.T])
                self.enplotax.set_ylim([0,(self.s.K[0:n].max()+self.s.U[0:n].max())+np.uint(self.s.K[0:n].max()+self.s.U[0:n].max())/40])

                #Fem la línia vermella una mica més gran perquè es vegi i no quedi amagada en alguns casos sota l'E.total
                self.enplotax.plot(t[0:i],self.s.K[0:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(t[0:i],self.s.U[0:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(t[0:i],self.s.K[0:i]+self.s.U[0:i],'g-',label = 'Total Energy')

                self.enplotax.legend(loc=7)

                self.enplotcanvas.draw()

            if(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                vs = np.linspace(0,self.s.V.max()+0.5,100) #Posem el +0.5 perquè es vegi del tot l'última barra

                self.histax.clear()
                self.histax.set_xlabel('v')
                self.histax.set_ylabel('Number of particles relative')
                self.histax.set_xlim([0,self.s.V.max()+0.5])
                self.histax.set_ylim([0,np.ceil(self.s.MB.max())])

                self.histax.hist(self.s.V[i,:],bins=np.arange(0,self.s.V.max()+1, 1),rwidth=0.5,density=True,color=[0.0,0.0,1.0])
                self.histax.plot(vs,self.s.MB[i,:],'r-')
                self.histcanvas.draw()


            if(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                self.acuhistax.clear() #Netegem el gràfic tot i no dibuixar res (per si quedava dibuix d'una anterior simulació)
                if(self.time > 40.):
                    vs = np.linspace(0,self.s.V.max()+0.5,100)

                    self.acuhistax.set_xlabel('v')
                    self.acuhistax.set_ylabel('Number of particles relative')
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])

                    #print(i,n,delta,int((i-int((40./self.dt)))/delta),len(self.s.Vacu))
                    self.acuhistax.hist(self.s.Vacu[int((i-int((40./self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.2),density=True)
                    self.acuhistax.plot(vs,self.s.MBacu[int((i-int((40./self.dt)))/delta)],'r-')

                self.acuhistcanvas.draw()

        elif(self.partmenu.current_tab.text == 'Subsystems'):
            with self.plotbox.canvas:
                for j in range(0,(self.n1)**2+(self.n2)**2):
                    if(j < self.n1**2):
                        Color(0.32,0.86,0.86)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))
                    else:
                        Color(0.43,0.96,0.16)
                        Ellipse(pos=((self.s.X[i,j])*scale*self.R+w/2.-self.R*scale/2.,(self.s.Y[i,j])*scale*self.R+h/2.-self.R*scale/2.),size=(self.R*scale,self.R*scale))

            self.time += interval*self.speed #Here is where speed accelerates animation
            self.progressbar.value = (self.time/self.T)*100 #Updates the progress bar.

            self.acucounter += 1


            #Anem a fer els histogrames per aquest altre apartat. Mirar comentaris de l'anterior per més info
            #Aquí dibuixarem els dos subsistemes en colors diferents
            if(self.plotmenu.current_tab.text == 'Energy'):
                t = np.arange(self.dt,self.T+10+self.dt,self.dt)

                self.enplotax.clear()
                self.enplotax.set_xlabel('t')
                self.enplotax.set_ylabel('Energy')

                self.enplotax.set_xlim([0,self.T])
                self.enplotax.set_ylim([0,(self.s.K[0:n].max()+self.s.U[0:n].max())+np.uint(self.s.K[0:n].max()+self.s.U[0:n].max())/40])

                self.enplotax.plot(t[0:i],self.s.K[0:i],'r-',label = 'Kinetic Energy', linewidth = 2.2)
                self.enplotax.plot(t[0:i],self.s.U[0:i],'b-',label = 'Potential Energy')
                self.enplotax.plot(t[0:i],self.s.K[0:i]+self.s.U[0:i],'g-',label = 'Total Energy')

                self.enplotax.legend(loc=7)

                self.enplotcanvas.draw()

            if(self.plotmenu.current_tab.text == 'Momentum'): #Instantaneous momentum histogram
                vs = np.linspace(0,self.s.V.max()+0.5,100) #Posem el +0.5 perquè es vegi del tot l'última barra

                self.histax.clear()
                self.histax.set_xlabel('v')
                self.histax.set_ylabel('Number of particles relative')
                self.histax.set_xlim([0,self.s.V.max()+0.5])
                self.histax.set_ylim([0,np.ceil(self.s.MB.max())])

                #self.histax.hist(self.s.V[i,0:self.n1**2],bins=np.arange(0, self.s.V.max() + 1, 1),rwidth=0.75,density=True,color=[0.32,0.86,0.86])
                self.histax.hist([self.s.V[i,0:self.n1**2],self.s.V[i,self.n1**2:self.n1**2+self.n2**2]],bins=np.arange(0, self.s.V.max() + 1, 1),rwidth=0.75,density=True,color=[[0.32,0.86,0.86],[0.43,0.96,0.16]])
                self.histax.plot(vs,self.s.MB[i,:],'r-')
                self.histcanvas.draw()


            if(self.plotmenu.current_tab.text == 'Acu'): #Accumulated momentum histogram
                self.acuhistax.clear()
                if(self.time > 40.):
                    vs = np.linspace(0,self.s.V.max()+0.5,100)

                    self.acuhistax.set_xlabel('v')
                    self.acuhistax.set_ylabel('Number of particles relative')
                    self.acuhistax.set_xlim([0,self.s.V.max()+0.5])
                    self.acuhistax.set_ylim([0,np.ceil(self.s.MB.max())])


                    self.acuhistax.hist(self.s.Vacu[int((i-int((40./self.dt)))/delta)],bins=np.arange(0, self.s.V.max() + 0.2, 0.2),density=True)
                    self.acuhistax.plot(vs,self.s.MBacu[int((i-int((40./self.dt)))/delta)],'r-')

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
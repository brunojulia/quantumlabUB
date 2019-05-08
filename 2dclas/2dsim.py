
import numpy as np
import matplotlib.pyplot as plt
from particle import *
from potentials import *

from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.properties import ObjectProperty,ListProperty,NumericProperty
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from matplotlib.figure import Figure

L = 200


T = 30
dt = 0.1


class main(BoxLayout):
    L = 200
    dx = 1
    nx = int(L/dx)

    T = 30
    dt = 0.1
    
    
    param0 = NumericProperty()
    param1 = NumericProperty()
    param2 = NumericProperty()
    param3 = NumericProperty()
    
    mass = NumericProperty()
    charge = 1.
    x0 = NumericProperty()
    y0 = NumericProperty()
    vx0 = NumericProperty()
    vy0 = NumericProperty()
    
    
    potentials = ListProperty()
    particlestrings = ListProperty()
    particles = []
    init_conds = []
    
    
    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.pot = Phi()
        
        '''
        self.plot = Figure()
        self.plot.set_xlabel('x')
        self.plot.set_ylabel('y')
        
        xx,yy, = np.meshgrid(np.linspace(-L/2,L/2,nx,endpoint=True),np.linspace(-L/2,L/2,nx,endpoint=True))
        im = np.zeros((nx,nx))
        for i in range(0,nx):
            for j in range(0,nx):
                im[i,j] = pot.val(xx[i,j],yy[i,j])
        self.image = FigureCanvasKivyAgg(self.plot)
        self.plot_box.add_widget(self.plot)
        '''
    
    def add_pot_list(self):
        self.potentials.append('Gauss:x0 = {}, y0 = {}, V0 = {}, Sig = {}'.format(round(self.param0,2),round(self.param1,2),round(self.param2,2),round(self.param3,2)))
        self.pot.add_function(gauss,dgaussx,dgaussy,[self.param0,self.param1,self.param2,self.param3])
    def reset_pot_list(self):
        self.pot.clear()
        self.potentials = []
        
    def add_particle_list(self):
        self.particlestrings.append('P{}: m = {}, x0 = {}, y0 = {}, vx0 = {}, vy0 = {}'.format(len(self.particlestrings)+1,round(self.mass,2),round(self.x0,2),round(self.y0,2),round(self.vx0,2),round(self.vy0,2)))
        self.particles.append(Particle(self.mass,self.charge,np.ones([1,4]),dt))
        self.init_conds.append([self.x0,self.y0,self.vx0,self.vy0])
    def reset_particle_list(self):
        self.particlestrings = []
        self.particles = []
        self.init_conds = []
    
    def compute(self):
        for i,p in enumerate(self.particles,0):
            p.ComputeTrajectoryF(self.init_conds[i],self.pot)
        print('Done')
        for p in self.particles:
            print(p.trajectory[-1,:])
            
            
class simApp(App):

    def build(self):
        return main()


if __name__ == '__main__':
    simApp().run()
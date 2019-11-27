import numpy as np
import os.path
import pickle as pkl

import wavef
import potentials

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import ObjectProperty,ListProperty,NumericProperty,StringProperty
from kivy.graphics.texture import Texture
from kivy.graphics import Rectangle,Color
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.image import Image


class main(BoxLayout):
    
    N=100
    dx=0.05
    dt=0.0025
    ntime=1000
    L=N*dx
    
    mass = 1.
    charge = 1.
    x0 = NumericProperty()
    y0 = NumericProperty()    
    
    xx,yy = np.meshgrid(np.arange(-L/2,L/2,dx),np.arange(-L/2,L/2,dx),sparse=True)
    
    potentials = ListProperty()
    potsave = []
    particles = []
    init_conds = []
    
    plot_texture = ObjectProperty()
    
    
    def __init__(self, **kwargs):
        super(main, self).__init__(**kwargs)
        self.pot = wavef.Phi()
        self.wav = wavef.Phi()
        self.demo = 0
        
        self.set_texture()
        self.n = 100
        self.time = 0.
        self.T = 630
        self.dt = 0.0025
        self.speed = 0.3
     #   self.change_speed()
        self.running = False
        
    def set_texture(self):
        self.n = 100
        self.im = np.zeros((self.n,self.n),dtype=np.uint8)
        self.plot_texture = Texture.create(size=self.im.shape,colorfmt='luminance',bufferfmt='uint')
        self.plotwave_texture = Texture.create(size=self.im.shape,colorfmt='luminance',bufferfmt='uint')
        
    def background(self):
        self.im = np.zeros((self.n,self.n))
        if(self.pot.functions.size == 0):
            self.im = np.uint8(self.im)
        else:
            self.im = self.pot.val(self.xx,self.yy)
            self.im = self.im + np.abs(self.im.min())
            self.im = np.uint8(255.*(self.im/self.im.max()))

    def update_texture(self):
        with self.plotbox.canvas:
            cx = self.plotbox.pos[0]
            cy = self.plotbox.pos[1]
            w = self.plotbox.size[0]
            h = self.plotbox.size[1]
            b = min(w,h)
            
            self.plot_texture.blit_buffer(self.im.reshape(self.im.size),colorfmt='luminance')
            Color(1.0,1.0,1.0)
            Rectangle(texture = self.plot_texture, pos = (cx,cy),size = (b,b))

        
    def update_parameters(self,touch):
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
              
    def add_pot_list(self):
        if(self.potmenu.current_tab.text == 'Oscilador'):
            self.potentials.append('Oscilador: x0 = {}, y0 = {}, k = {}'.format(round(self.param0slider.value,3),round(self.param1slider.value,3),round(self.param2oslider.value,2)))
   #         self.potentialsave.append('Oscilador:x0 = {}, y0 = {}, k = {}'.format(round(self.param0slider.value,2),round(self.param1slider.value,2),round(self.param2oslider.value,2)))
            self.pot.add_function(potentials.osc,[self.param0slider.value,self.param1slider.value,self.param2oslider.value])
        elif(self.potmenu.current_tab.text == 'Gaussian'):
            self.potentials.append('Gaussian: x0 = {}, y0 = {}, V0 = {}, Sigma = {}'.format(round(self.param0slider.value,3),round(self.param1slider.value,3),round(self.param2gslider.value,2),round(self.param3gslider.value,2)))
   #         self.potentialsave.append('Gaussian:x0 = {}, y0 = {}, V0 = {}, Sigma = {}'.format(round(self.param0slider.value,2),round(self.param1slider.value,2),round(self.param2gslider.value,2),round(self.param3gslider.value,2)))
            self.pot.add_function(potentials.gauss,[self.param0slider.value,self.param1slider.value,self.param2gslider.value,self.param3gslider.value])
        
        self.background()
        self.update_texture()

        
    def reset_pot_list(self):
        self.pot.clear()
        self.potentials = []
        self.potentialsave =[]
        self.plotbox.canvas.clear()
        self.background()
        
        '''
        with self.statuslabel.canvas:
            Color(1,0,0)
            Rectangle(pos=self.statuslabel.pos,size=self.statuslabel.size)
        '''
    
    def background_main(self):
        self.im = np.zeros((self.n,self.n))
        if(self.wav.functions.size == 0):
            self.im = np.uint8(self.im)
        else:               
            self.inwavef = self.wav.val(self.xx,self.yy) #Complex array, value of initial wavef
            self.im = wavef.Wave.Probability(self,self.inwavef) #Float array, probability from initial wavef
            self.im = self.im + np.abs(self.im.min())
            self.im = np.uint8(255.*(self.im/self.im.max()))
    
    def update_texture_main(self):
        with self.wavebox.canvas:
            cx = self.wavebox.pos[0]
            cy = self.wavebox.pos[1]
            w = self.wavebox.size[0]
            h = self.wavebox.size[1]
            b = min(w,h)
            
            self.plotwave_texture.blit_buffer(self.im.reshape(self.im.size),colorfmt='luminance')
            Color(1.0,1.0,1.0)
            Rectangle(texture = self.plotwave_texture, pos = (cx,cy),size = (b,b))
    
    def add_wave_list(self):
        if(self.partmenu.current_tab.text == 'Eig. Osci.'):
         #   self.particlestrings.append('Eig.Osci.: x0 = {}, y0 = {}, k = {}, a = {}, b={}'.format(round(self.x0slider.value,2),round(self.y0slider.value,2),round(self.kslider.value,2),round(self.aslider.value,2),round(self.bslider.value,2)))
         #   self.particlesave.append('Eig.Osci.: x0 = {}, y0 = {}, k = {}, a = {}, b={}'.format(round(self.x0slider.value,2),round(self.y0slider.value,2),round(self.kslider.value,2),round(self.aslider.value,2),round(self.bslider.value,2)))
         #   self.particles.append(wavef.Wave(self.pot,wavef.InitWavef.OsciEigen([self.xx,self.yy],[self.x0slider.value,self.y0slider.value,self.kslider.value,self.aslider.value,self.bslider.value])))
            self.init_conds.append(wavef.InitWavef.OsciEigen([self.xx,self.yy],[self.x0slider.value,self.y0slider.value,self.kslider.value,self.aslider.value,self.bslider.value]))
            self.wav.add_function(wavef.InitWavef.OsciEigen,[self.x0slider.value,self.y0slider.value,self.kslider.value,self.aslider.value,self.bslider.value])
            
        self.background_main()
        self.update_texture_main()

    def reset_wave_list(self):
        self.particlestrings = []
        self.particlesave = []
        self.particles = []
        self.init_conds = []
        self.wavebox.canvas.clear()
        self.wav.clear()

    
    def demo1(self):
        ' Demo1 = harm osci ground eigenstate not centered '
        self.demo = 1
        self.demolabel = 'harmonic oscillatior ground eigenstate not centered'
        self.T = 630
        
        x0 = -0.5
        y0 = 0.
        w = 4.
        
        x0po = 0.
        y0po = 0.
        
        #Add potential
        
        self.pot.add_function(potentials.osc,[x0po,y0po,w])
        
        self.background()
        self.update_texture()
        
         #Add wave function
        
        self.wav.add_function(wavef.InitWavef.OsciEigen,[x0,y0,w,0,0])
                    
        self.background_main()
        self.update_texture_main()
        
        #Update time evolution from file
        
        file = open('Demo1','rb')
        probevol = pkl.load(file)

        self.probability = probevol
       
    
    def demo2(self):
        'Demo2 = harm osci ground estate pulsating'
        self.demo = 2
        self.demolabel = 'harmonic oscillatior ground eigenstate with different frequency from the potential'
        self.T = 840
        
        x0 = 0.
        y0 = 0.
        ww = 4.
        wp = 3.
        
        #Add potential
        
        self.pot.add_function(potentials.osc,[0.,0.,wp])
        
        self.background()
        self.update_texture()
        
        #Add wave function
        
        self.wav.add_function(wavef.InitWavef.OsciEigen,[x0,y0,ww,0,0])
            
        self.background_main()
        self.update_texture_main()
        
        #Update time evolution from file
        
        file = open('Demo2','rb')
        probevol = pkl.load(file)

        self.probability = probevol
        
    
    def demo3(self):
        'Demo3 = box'
        self.demo = 3
        self.demolabel = 'smaller box'
        self.T = 630
        
        x0 = 0.
        y0 = 0.
        sigma = 0.25
        px0 = 0.
        py0 = 0.
        
        V0pot = 100.
        
        #Add potential
        
        self.pot.add_function(potentials.box,[V0pot,x0,y0])
        
        self.background()
        self.update_texture()
        
        #Add wave function
        
        self.wav.add_function(wavef.InitWavef.Gauss,[x0,y0,sigma,px0,py0])
            
        self.background_main()
        self.update_texture_main()
        
        #Update time evolution from file
    
        file = open('Demo3','rb')
        probevol = pkl.load(file)

        self.probability = probevol
        
        
    def demo31(self):
        'Demo3.1 = box'
        self.demo = 3
        self.demolabel = 'smaller box'
        self.T = 630
        
        x0 = 0.
        y0 = 0.
        sigma = 0.25
        px0 = 0.
        py0 = 0.
        
        V0pot = 10**5.
        
        #Add potential
        
        self.pot.add_function(potentials.box,[V0pot,x0,y0])
        
        self.background()
        self.update_texture()
        
        #Add wave function
        
        self.wav.add_function(wavef.InitWavef.Gauss,[x0,y0,sigma,px0,py0])
            
        self.background_main()
        self.update_texture_main()
        
        #Update time evolution from file
    
        file = open('Demo3.1','rb')
        probevol = pkl.load(file)

        self.probability = probevol
        
    
    def demo4(self):
        'Demo4 = double barrier / double well'
        self.demo = 4
        self.demolabel = 'double well in x'
        self.T = 630
        
        x0 = 0.
        y0 = 0.
        w = 5.

        wx = 5.
        wy = 2.

        #Add potential
        
        self.pot.add_function(potentials.osc_nosym,[x0,y0,wx,wy])
        
        self.background()
        self.update_texture()
        
        #Add wave function
        
        self.wav.add_function(wavef.InitWavef.OsciEigen,[x0,y0,w,0,0])
            
        self.background_main()
        self.update_texture_main()
        
        #Update time evolution from file
    
        file = open('Demo4','rb')
        probevol = pkl.load(file)

        self.probability = probevol
        
    
    def demo5(self):
        'Demo5 = single slit'
        self.demo = 4
        self.demolabel = 'single slit'
        self.T = 400
        
        x0 = -1.
        y0 = 0.
        sigma = 0.25
        px0 = 1.
        py0 = 0.
        
        V0pot = 1000.

        #Add potential
        
        self.pot.add_function(potentials.singleslit,[V0pot,x0,y0])
        
        self.background()
        self.update_texture()
        
        #Add wave function
        
        self.wav.add_function(wavef.InitWavef.Gauss,[x0,y0,sigma,px0,py0])
            
        self.background_main()
        self.update_texture_main()
        
        #Update time evolution from file
    
        file = open('Demo5','rb')
        probevol = pkl.load(file)

        self.probability = probevol

    def demo51(self):
        'Demo5.1 = single slit'
        self.demo = 4
        self.demolabel = 'single slit'
        self.T = 400
        
        x0 = -1.
        y0 = 0.
        sigma = 0.25
        px0 = 1.
        py0 = 0.
        
        V0pot = 1000.

        #Add potential
        
        self.pot.add_function(potentials.doubleslit,[V0pot,x0,y0])
        
        self.background()
        self.update_texture()
        
        #Add wave function
        
        self.wav.add_function(wavef.InitWavef.Gauss,[x0,y0,sigma,px0,py0])
            
        self.background_main()
        self.update_texture_main()
        
        #Update time evolution from file
    
        file = open('Demo5.1','rb')
        probevol = pkl.load(file)

        self.probability = probevol
        
    def pop(self):
        if (self.demo == 1):
            pop = Popup(title='Energy', content=Image(source='Energy1.png'),
                    size_hint=(None, None), size=(700, 1000))
        elif (self.demo == 2):
            pop = Popup(title='Energy', content=Image(source='Energy2.png'),
                    size_hint=(None, None), size=(700, 1000))
        elif (self.demo == 3):
            pop = Popup(title='Energy', content=Image(source='Energy3.png'),
                    size_hint=(None, None), size=(700, 1000))
        elif (self.demo == 4):
            pop = Popup(title='Energy', content=Image(source='Energy4.png'),
                    size_hint=(None, None), size=(700, 1000))
        
        pop.open()

    def compute(self):        
        self.initwave = wavef.Wave(self.pot.val(self.xx,self.yy),self.wav.val(self.xx,self.yy),self.dt,self.T)
        self.probability = self.initwave.ProbEvolution() #wavef.Wave.ProbEvolution(self.pot.val,self.wav.val)
            
        with self.statuslabel.canvas:
            Color(0,1,0)
            Rectangle(pos=self.statuslabel.pos,size=self.statuslabel.size)
        
        norm0 = wavef.Wave.Norm(self,self.probability[0,:,:])
        norm1 = wavef.Wave.Norm(self,self.probability[self.T-1,:,:])
        
        print('Norma t_ini: ', norm0)
        print('Norma t_final: ', norm1)
        
        
            
    def play(self):
        self.timer = Clock.schedule_interval(self.animate,0.04) #0.04=interval
        self.running = True
    
    def pause(self):
        if(self.running==True):
            self.timer.cancel()
        else:
            pass


    def stop(self):
        self.pause()
        self.time = 0
        
        self.reset_pot_list()
        self.reset_wave_list()
        self.init_conds = []
      
    ''' El tiempo de la animacion!!!!!! '''
 
    def animate(self,interval):
        cx = self.wavebox.pos[0]
        cy = self.wavebox.pos[1]
        w = self.wavebox.size[0]
        h = self.wavebox.size[1]
        b = min(w,h)
        
        dt=self.dt
        
        t = int(self.time/dt) # T*dt=Tmax
        if t >= self.T :
            t = 0.
        
        self.im = self.probability[t,:,:]
        
        self.im = self.im + np.abs(self.im.min())
        self.im = np.uint8(255.*(self.im/self.im.max()))
            
        with self.wavebox.canvas:
            self.plotwave_texture.blit_buffer(self.im.reshape(self.im.size),colorfmt='luminance')
            Color(1.0,1.0,1.0)
            Rectangle(texture = self.plotwave_texture, pos = (cx,cy),size = (b,b))
            
        self.wavebox.canvas.clear()
        self.update_texture_main()
       
        self.time += interval*self.speed
        if(self.time >= self.T*dt):
            self.time = 0.
            
            
class BoxApp(App):

    def build(self):
        return main()


if __name__ == '__main__':
    BoxApp().run()
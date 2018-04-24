#misc imports
import numpy as np
import threading
import random


#cranknicolson imports
from dsexperiment import DSexperiment
from dsexperiment import create_experiment_from_files

#kivy imports
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty, NumericProperty
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.graphics import Color
from kivy.clock import Clock

class DoubleSlitScreen(BoxLayout):
    #Objects binded from .kv file
    p_rectangle = ObjectProperty()
    playpause_button = ObjectProperty()
    frame_slider = ObjectProperty()
    speed_slider = ObjectProperty()
    normalize_switch = ObjectProperty()
    screen_pos_slider = ObjectProperty()
    screen_width_slider = ObjectProperty()
    compute_button = ObjectProperty()
    progress_bar = ObjectProperty()

    slider_d = ObjectProperty()
    slider_sx = ObjectProperty()
    slider_sy = ObjectProperty()

    #Objects created here
    frame = NumericProperty(0) #current frame
    frames = NumericProperty(0) #total number of frames
    texture = ObjectProperty() #texture object (initialized at create_texture)
    zoom = NumericProperty(1) #this value autoadjusts when calling blit_P

    #Drawing parameters
    #size and position of he heatmap
    wh = 0
    hh = 0
    xh = 0
    yh = 0

    #Playback status and settings
    speed = NumericProperty(4)
    playing = False
    normalize_each_frame = False

    #Simulation results
    computing = False
    computed = False
    Pt = None
    times = None
    maxP = 0

    def __init__(self, *args, **kwargs):
        super(DoubleSlitScreen, self).__init__(*args, **kwargs)

        print("Trying to load last simulation")
        try:
            self.experiment = create_experiment_from_files("lastsim")
            print("Last simulation loaded correctly")
            self.computation_done(save = False)
        except FileNotFoundError:
            print("Could not find last simulation, creating new one...")
            self.experiment = DSexperiment()
            self.experiment.set_gaussian_psi0(p0x = 100/self.experiment.Lx)


        self.slider_sx.value = self.experiment.sx
        self.slider_sy.value = self.experiment.sy
        self.slider_d.value = self.experiment.d

        self.create_texture()

    #Drawing functions
    def create_texture(self):
        self.texture = Texture.create(size = self.experiment.Pt[0].shape[::-1], colorfmt = "luminance", bufferfmt = "uint")

    def blit_P(self, P):
        """
        This function draws the heatmap for P centered at
        P is a 2d numpy array
        """

        #Basically if white should represent the maximum value of P at each frame
        #or should represent the maximum value of all frames
        if self.normalize_each_frame:
            max = np.max(P)
        else:
            max = self.maxP

        #Stores the P matrix in the texture object
        #this texture is created in the method creature_texture and already has the size
        #It's a gray-scale texture so value must go from 0 to 255 (P/self.maxP)*255
        #It must be an array of unsigned 8bit integers. And also it has to be flattened

        self.texture.blit_buffer( ( (P/max)*255 ).astype(np.uint8).reshape(P.size), colorfmt = "luminance")

        #Draws the box walls and the
        with self.p_rectangle.canvas:
            #Determines the size of the box:
            #Full height
            self.zoom = self.p_rectangle.height/P.shape[0]
            #If full height implies cutting by the sides, it uses full width
            if P.shape[1]*self.zoom > self.p_rectangle.width:
                #Full width
                self.zoom = self.p_rectangle.width/P.shape[1]

            self.wh = P.shape[1]*self.zoom
            self.hh = P.shape[0]*self.zoom

            self.xh = self.p_rectangle.pos[0] + self.p_rectangle.width/2 - self.wh/2
            self.yh = self.p_rectangle.pos[1] + self.p_rectangle.height/2 - self.hh/2

            Color(1., 0, 0) #Red
            #box wall
            Rectangle(pos = (self.xh-5, self.yh-5), size = (self.wh+10, self.hh+10))

            #Heatmap
            Color(1., 1., 1.) #White
            Rectangle(texture = self.texture, pos = (self.xh, self.yh), size = (self.wh, self.hh))


    def draw_walls(self):
        with self.p_rectangle.canvas:
            scale = self.zoom/self.experiment.dx

            sx = self.experiment.sx
            sy = self.experiment.sy
            d = self.experiment.d

            #Slits
            Color(1., 0, 0)
            #top wall
            Rectangle(pos = (self.xh + self.wh/2 - (sx*scale)/2, self.yh + self.hh/2 + ((sy+d)*scale)/2), size = (sx*scale, self.hh/2 - d*scale/2 - sy*scale/2))
            #middle wall
            Rectangle(pos = (self.xh + self.wh/2 - (sx*scale)/2, self.yh + self.hh/2  - (d/2-sy/2)*scale), size = (sx*scale, (d-sy)*scale) )
            #bottom wall
            Rectangle(pos = (self.xh + self.wh/2 - (sx*scale)/2, self.yh), size = (sx*scale, self.hh/2 -(d+sy)*scale/2 ))

    def draw_measures(self):

        with self.p_rectangle.canvas:
            scale = self.zoom/self.experiment.dx

            #Measuring screen
            Color(0, 1., 0, 0.25)
            Rectangle(pos = (self.xh + self.experiment.mp*self.zoom, self.yh), size = (self.experiment.mw*self.zoom, self.hh))

            Color(0, 1., 0)
            for measure in self.experiment.measurements:
                Rectangle(pos = (self.xh + measure[1]*self.zoom, self.yh + measure[0]*self.zoom), size = (self.zoom, self.zoom))

    #Crank-Nicolson functions
    def computation_update(self, msg, x):
        self.progress_bar.value = 100*x

    def computation_done(self, save = True):
        self.computed = True
        self.computing = False
        self.compute_button.disabled = False

        self.frames = self.experiment.Pt.shape[0]
        self.frame_slider.max = self.frames - 1
        self.maxP = np.max(self.experiment.Pt)

        self.create_texture()

        if save:
            self.experiment.save_to_files("lastsim")

    def compute(self):
        """
        This is called when the compute button is pressed
        """
        if not self.computing:
            self.experiment.compute_evolution(update_callback = self.computation_update, done_callback = self.computation_done)

            self.playing = False
            self.computed = False
            self.computing = True
            self.compute_button.disabled = True

            self.frame = 0

    #Playback functions
    def playpause(self):
        self.playing = not self.playing

    def change_frame(self):
        self.playing = False
        self.frame = int(self.frame_slider.value)

    def measure(self, N = 1):
        self.experiment.measure(N)

    def remove_measurements(self):
        self.experiment.clear_measurements()

    def update(self, dt):
        self.p_rectangle.canvas.clear()

        self.playpause_button.disabled = not self.computed

        if self.playing:
            self.playpause_button.text = "Pause"
        else:
            self.playpause_button.text = "Play"

        self.frame_slider.disabled = not self.computed

        self.normalize_each_frame = self.normalize_switch.active

        self.speed = int(self.speed_slider.value)

        self.experiment.mp = int(self.experiment.Pt[0].shape[1]*self.screen_pos_slider.value)
        self.experiment.mw = self.screen_width_slider.value

        self.experiment.update_slits(sx = self.slider_sx.value, sy = self.slider_sy.value, d = self.slider_d.value)

        if self.playing:
            self.blit_P(self.experiment.Pt[self.frame])
            self.draw_walls()
            self.draw_measures()
            self.frame = (self.frame+self.speed)%self.frames
            self.frame_slider.value = self.frame

        else:
            if not self.experiment.Pt is None:
                self.blit_P(self.experiment.Pt[self.frame])
                self.draw_walls()
                self.draw_measures()



class DoubleSlitApp(App):
    def build(self):
        random.seed()
        screen = DoubleSlitScreen()
        Clock.schedule_interval(screen.update, 1.0 / 30.0)
        return screen

if __name__ == "__main__":
    DoubleSlitApp().run()

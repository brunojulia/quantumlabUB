#misc imports
import numpy as np
import threading

#cranknicolson imports
from cranknicolson.cn2d import crank_nicolson2D

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

class CNThread(threading.Thread):
    def __init__(self, threadID, threadName, app):
        threading.Thread.__init__(self)
        self.app = app

    def run(self):
        self.app.cnupdate(0)
        Lx = self.app.Lx
        Ly = self.app.Ly
        dx = self.app.dx

        x, y = np.meshgrid(np.arange(-Lx, Lx, dx), np.arange(-Ly, Ly, dx))

        psi = self.app.psi0(x, y)
        V = self.app.V(x, y)

        psit, self.app.times = crank_nicolson2D(x, y, psi, V, tmax = 2, callback = self.app.cnupdate)

        self.app.Pt = np.absolute(psit)**2
        self.app.maxP = np.max(self.app.Pt)
        self.app.frames = self.app.Pt.shape[0]

        self.app.computing = False
        self.app.create_texture()


class DoubleSlitScreen(BoxLayout):
    #Objects binded from .kv file
    p_rectangle = ObjectProperty()
    playpause_button = ObjectProperty()
    progress_bar = ObjectProperty()

    slider_d = ObjectProperty()
    slider_sx = ObjectProperty()
    slider_sy = ObjectProperty()

    #Objects created here
    frame = NumericProperty(0)
    frames = 0
    texture = ObjectProperty()
    zoom = NumericProperty(2)

    #Drawing parameters
    #position and size of he heatmap
    wh = 0
    hh = 0
    xh = 0
    yh = 0

    #Playback status
    playing = False

    #Simulation parameters
    ##grid
    Lx = 10.0
    Ny = 300
    Nx = 300
    dx = 2*Lx/Nx
    Ly = Ny*dx/2

    ##psi0
    x0 = 5
    y0 = 0
    s = 2
    p0x = 100.0/Lx
    p0y = 0.0/Lx
    ##V
    Vo = 200
    sx = 0.25
    sy = 2
    d = 4

    #Simulation results
    computing = False
    Pt = None
    times = None
    maxP = 0

    def __init__(self, *args, **kwargs):
        super(DoubleSlitScreen, self).__init__(*args, **kwargs)
        x, y = np.meshgrid(np.arange(-self.Lx, self.Lx, self.dx), np.arange(-self.Ly, self.Ly, self.dx))
        self.Pt = [np.absolute(self.psi0(x, y))**2]
        self.maxP = np.max(self.Pt)
        self.create_texture()

        self.slider_d.value = self.d
        self.slider_sx.value = self.sx
        self.slider_sy.value = self.sy

    #Drawing functions
    def create_texture(self):
        self.texture = Texture.create(size = self.Pt[0].shape, colorfmt = "luminance", bufferfmt = "uint")

    def blit_P(self, P):
        """
        This function draws the heatmap for P centered at
        P is a 2d numpy array
        """
        #Draws to the Texture
        self.maxP = np.max(P)
        self.texture.blit_buffer( ( (P/self.maxP)*255 ).astype(np.uint8).reshape(P.size), colorfmt = "luminance")

        #Draws rectangle to the canvas
        with self.p_rectangle.canvas:
            self.wh = P.shape[0]*self.zoom
            self.hh = P.shape[1]*self.zoom

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
            scale = self.zoom/self.dx
            #Slits
            Color(1., 0, 0)
            #top wall
            Rectangle(pos = (self.xh + self.wh/2 - (self.sx*scale)/2, self.yh + self.hh/2 + ((self.sy+self.d)*scale)/2), size = (self.sx*scale, self.hh/2 -self.d*scale/2 - self.sy*scale/2))
            #middle wall
            Rectangle(pos = (self.xh + self.wh/2 - (self.sx*scale)/2, self.yh + self.hh/2  - (self.d/2-self.sy/2)*scale), size = (self.sx*scale, (self.d-self.sy)*scale) )
            #bottom wall
            Rectangle(pos = (self.xh + self.wh/2 - (self.sx*scale)/2, self.yh), size = (self.sx*scale, self.hh/2 -(self.d+self.sy)*scale/2 ))

    #Crank-Nicolson functions
    def cnupdate(self, x):
        self.progress_bar.value = 100*x

    def compute(self):
        if not self.computing:
            cnThread = CNThread(1, "CN_Thread", self)
            cnThread.start()
            self.computing = True

    def psi0(self, x, y):
        """
        Wave function at t = 0
        """
        r2 = (x-self.x0)**2 + (y-self.y0)**2
        return np.exp(-1j*(self.p0x*x + self.p0y*y))*np.exp(-r2/(4*self.s**2))/(2*self.s**2*np.pi)**(.5)

    def Vslits(self, x, y):
        if np.abs(x) < self.sx/2:
            if np.abs(y) < (self.d/2 - self.sy/2) or np.abs(y) > (self.d/2 + self.sy/2):
                return self.Vo
            else:
                return 0
        else:
            return 0

    def V(self, x, y):
        return np.vectorize(self.Vslits)(x, y)

    #Playback functions
    def playpause(self):
        self.playing = not self.playing

    def update(self, dt):
        self.p_rectangle.canvas.clear()
        self.playpause_button.disabled = self.Pt is None

        self.d = self.slider_d.value
        self.sx = self.slider_sx.value
        self.sy = self.slider_sy.value

        if self.playing:
            self.blit_P(self.Pt[self.frame])
            self.draw_walls()
            self.frame = (self.frame+1)%self.frames
        else:
            if not self.Pt is None:
                self.blit_P(self.Pt[0])
                self.draw_walls()


class DoubleSlitApp(App):
    def build(self):
        screen = DoubleSlitScreen()
        Clock.schedule_interval(screen.update, 1.0 / 30.0)
        return screen

if __name__ == "__main__":
    DoubleSlitApp().run()

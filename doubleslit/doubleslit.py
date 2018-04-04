from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.properties import ObjectProperty, NumericProperty
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
import numpy as np
from numpy.random import random, randint
from kivy.clock import Clock

from cranknicolson.cn2d import crank_nicolson2D, VbarreraDiscret, psi0

psit = np.load("cranknicolson/psit2d.npy")
maxP = np.max(np.absolute(psit)**2)
frames = psit.shape[0]

class DoubleSlitScreen(BoxLayout):
    p_rectangle = ObjectProperty()
    frame = NumericProperty(0)
    texture = ObjectProperty()
    playing = False

    #cranknicolson parameters
    psit = None
    times = None


    def create_texture(self):
        self.texture = Texture.create(size = psit[0].shape, colorfmt = "luminance", bufferfmt = "uint")

    def blit_P(self, P):
        """
        P is a 2d numpy array
        """
        # create a 64x64 texture, defaults to rgba / ubyte
        self.texture.blit_buffer( ( (P/maxP)*255 ).astype(np.uint8).reshape(P.size), colorfmt = "luminance")

        # that's all ! you can use it in your graphics now :)
        # if self is a widget, you can do this
        with self.p_rectangle.canvas:
            x = self.p_rectangle.pos[0] + self.p_rectangle.width/2 - P.shape[0]/2
            y = self.p_rectangle.pos[1] + self.p_rectangle.height/2 - P.shape[1]/2
            Rectangle(texture = self.texture, pos = (x, y), size = P.shape)

    def cnupdate(self, x):
        print(x)

    def compute(self):
        Lx = 10.0
        Ny = 300
        Nx = 300
        dx = 2*Lx/Nx
        Ly = Ny*dx/2

        x, y = np.meshgrid(np.arange(-Lx, Lx, dx), np.arange(-Ly, Ly, dx))

        self.psit, self.times = crank_nicolson2D(x, y, psi0, VbarreraDiscret, tmax = 2, callback = self.cnupdate)

    def playpause(self):
        self.playing = not self.playing

    def update(self, dt):
        if self.playing:
            self.blit_P(np.absolute(self.psit[self.frame])**2)
            self.frame = (self.frame+1)%frames

class DoubleSlitApp(App):
    def build(self):
        screen = DoubleSlitScreen()
        screen.create_texture()
        Clock.schedule_interval(screen.update, 1.0 / 30.0)
        return screen

if __name__ == "__main__":
    DoubleSlitApp().run()

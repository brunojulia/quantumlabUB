import numpy as np
from matplotlib import pyplot as plt
#from scipy import integrate as inte
from scipy.integrate import solve_ivp
from kivy.app import App
from kivy.core.window import Window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.screenmanager import FadeTransition, SlideTransition
#from kivy.uix.popup import Popup
from functools import partial
from kivy.graphics.vertex_instructions import Line,Rectangle,Ellipse
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty,ReferenceListProperty,\
    NumericProperty,StringProperty,ListProperty,BooleanProperty
from kivy.config import Config
Config.set('graphics', 'width', '900')
Config.set('graphics', 'height', '600')
Window.clearcolor = (1, 1, 1, 1)


class tun_v5App(App):
    def build(self):
        self.title='Spin Tunneling'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)
        self.get_screen('Tunneling').stpseudo_init()
       # self.get_screen("Game").gpseudo_init()
    

class StartingScreen(Screen):
    def __init__(self,**kwargs):
        super(StartingScreen,self).__init__(**kwargs)

    def transition_ST(self):
        """Transicio del starting a l Spin Tunneling"""

        stscreen = self.manager.get_screen('Tunneling')
        self.manager.transition = FadeTransition()
        self.manager.current = 'Tunneling'

    def transition_Game(self):
        """Transicio del starting a l Spin Tunneling"""

        stscreen = self.manager.get_screen('Game')
        self.manager.transition = FadeTransition()
        self.manager.current = 'Game'

class TunnelingScreen(Screen):
    H_type= NumericProperty(0)
    s= NumericProperty(0)


    def __init__(self,**kwargs):
        super(TunnelingScreen,self).__init__(**kwargs)

    def stpseudo_init(self):
        self.plot, self.axs =plt.subplots(1)
        self.axs.set(xlim=[-50,50],ylim=[0,1])


        self.plot = FigureCanvasKivyAgg(self.plot)
        self.box1.add_widget(self.plot)

    def transition_SScreen(self, *largs):
        self.manager.current = 'starting'


    def send_but(self):

        plt.clf()
        self.mlt = int(2 * self.s + 1)

        # Definicio de parametres generals per descriure l,Hamiltonia per un spin s
        self.Sx = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.Sy = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.Sz = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.m = np.zeros(shape=(self.mlt, self.mlt), dtype=int)

        for i in range(self.mlt):
            self.Sz[i, i] = (i - self.s) * (-1)
            self.m[i, i] = 1
            for k in range(self.mlt):
                if k == (i + 1):
                    self.Sx[i, k] = 0.5 * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
                    self.Sy[i, k] = -0.5j * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))

                if k == (i - 1):
                    self.Sx[i, k] = 0.5 * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
                    self.Sy[i, k] = 0.5j * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
        self.Sxy2 = np.dot(self.Sx, self.Sx) - np.dot(self.Sy, self.Sy)
        self.Sz2 = np.dot(self.Sz, self.Sz)

        self.y0 = []
        for i in range(self.mlt):
            if i != (self.mlt - 1):
                self.y0.append(0 + 0j)
            else:
                self.y0.append(1 + 0j)

        t0 = -50.
        tf = 50.
        self.alpha=0.1
        # self.alpha = float(self.ids.alpha_text.text)
        self.a=0.01
        self.ti = [t0, tf]

        self.sol = solve_ivp(self.dam, self.ti, self.y0,rtol=1e-4)

        am = self.sol.y
        mod2 = abs(am) * abs(am)

        for i in range(self.mlt - 1):
            if i == 0:
                norma = (list(map(sum, zip(mod2[i], mod2[i + 1]))))
            else:
                norma = (list(map(sum, zip(norma, mod2[i + 1]))))

        temps=self.sol.t

        plt.plot(temps, norma, label='Norma')
        plt.axhline(y=1, xmin=t0, xmax=tf, ls='dashed')
        for i in range(self.mlt):
            # plt.plot(t,mod2[i],label='|'+str(int(i-s))+'>')
            plt.plot(temps, mod2[i], label='|' + str(int(i - self.s)) + '>')
        plt.legend()

        #A VEURE SI PUC FER EL MALEÏT DIBUIX


        self.box1.remove_widget(self.plot)
        self.plot = FigureCanvasKivyAgg(plt.gcf())
        self.box1.add_widget(self.plot)

        # self.plot.draw()
        print('ACABAT')

    def H(self,t):
        if self.H_type==1:
            Ham = -self.Sz2 + self.a * t * self.Sz + self.alpha * (self.Sxy2)
        if self.H_type==2:
            Ham= -self.Sz2+self.a * t * self.Sz + self.alpha * self.Sx
        return Ham

    def dam(self,t, y):
        dadt = []
        for i in range(self.mlt):
            sum_H = 0.
            mi = self.m[i]

            for k in range(self.mlt):
                mk = self.m[k]
                mk = np.transpose(mk)
                sum_H = sum_H + y[k] * np.dot(mi, (np.dot(self.H(t), mk)))

            dam_res = -1j * sum_H
            dadt.append(dam_res)
            dam_res = 0

        return dadt

class GameScreen(Screen):
    H_type= NumericProperty(0)
    s= NumericProperty(0)
    rm=NumericProperty(0)
    estat=NumericProperty(0)
    t0=NumericProperty(-50)
    tf=NumericProperty(50)
    ft=0     #Primera vegada que executa el dibuix
    gtime=None
    gpos=None
    gene=None
    Comptador=None
    Counter=None

    def __init__(self,**kwargs):
        super(GameScreen,self).__init__(**kwargs)
        # CLOCK


    def gamepseudo_init(self):
        pass

    def transition_SScreen(self, *largs):
        self.manager.current = 'starting'

    def send_but(self):
        self.mlt = int(2 * self.s + 1)

        # Definicio de parametres generals per descriure l,Hamiltonia per un spin s
        self.Sx = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.Sy = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.Sz = np.zeros(shape=(self.mlt, self.mlt), dtype=complex)
        self.m = np.zeros(shape=(self.mlt, self.mlt), dtype=int)

        for i in range(self.mlt):
            self.Sz[i, i] = (i - self.s) * (-1)
            self.m[i, i] = 1
            for k in range(self.mlt):
                if k == (i + 1):
                    self.Sx[i, k] = 0.5 * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
                    self.Sy[i, k] = -0.5j * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))

                if k == (i - 1):
                    self.Sx[i, k] = 0.5 * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
                    self.Sy[i, k] = 0.5j * np.sqrt(self.s * (self.s + 1) - (i - self.s) * (k - self.s))
        self.Sxy2 = np.dot(self.Sx, self.Sx) - np.dot(self.Sy, self.Sy)
        self.Sz2 = np.dot(self.Sz, self.Sz)

        self.y0 = []
        for i in range(self.mlt):
            if i != (self.mlt - 1):
                self.y0.append(0 + 0j)
            else:
                self.y0.append(1 + 0j)

        
        self.alpha=0.1
        self.a=0.01
        self.ti = [self.t0, self.tf]

        self.sol = solve_ivp(self.dam, self.ti, self.y0,rtol=1e-4)

        am = self.sol.y
        self.mod2 = abs(am) * abs(am)

        for i in range(self.mlt - 1):
            if i == 0:
                norma = (list(map(sum, zip(self.mod2[i], self.mod2[i + 1]))))
            else:
                norma = (list(map(sum, zip(norma, self.mod2[i + 1]))))

        self.temps=self.sol.t
        self.energia=self.energy()



        # self.plot.draw()
        print('ACABAT')
    def H(self,t):
        if self.H_type==1:
            Ham = -self.Sz2 + self.a * t * self.Sz + self.alpha * (self.Sxy2)
        if self.H_type==2:
            Ham= -self.Sz2+self.a * t * self.Sz + self.alpha * self.Sx
        return Ham

    def dam(self,t, y):
        dadt = []
        for i in range(self.mlt):
            sum_H = 0.
            mi = self.m[i]

            for k in range(self.mlt):
                mk = self.m[k]
                mk = np.transpose(mk)
                sum_H = sum_H + y[k] * np.dot(mi, (np.dot(self.H(t), mk)))

            dam_res = -1j * sum_H
            dadt.append(dam_res)
            # dam_res = 0

        return dadt

    # Càlcul de energia no pertorbada
    def energy(self):
        ene=[]
        for j in range(len(self.temps)):
            t_j = self.temps[j]
            ene_i = []
            for i in range(self.mlt):
                mi = self.m[i]
                mi_t = np.transpose(mi)
                e=np.dot(mi, (np.dot(self.H(t_j), mi_t)))
                ene_i.append(abs(e))
            ene.append(ene_i)
        return ene


    def drawcall(self):
        Background.figures(self,self.s,self.temps,self.mod2,self.energia)



# Dibuixos de l'energia etc
class Background(Widget):
    # temps=ListProperty([])
    comptador = -1
    mlt=None
    s=None
    prob=None
    t=None
    prob_space=None
    posi_t=None
    # estat = 0
    k=0
    def __init__(self,**kwargs):

        super().__init__(**kwargs)
        self.timer=-50
        self.counter=0
        Clock.schedule_interval(self.freshrate, 1/36)


    def figures(self,s,t,y,e):
        self.spin = s
        self.temps=list(t)
        self.prob=list(y)
        self.mlt=2*s+1
        GameScreen.gtime=list(t)
        GameScreen.gpos=list(y)
        GameScreen.mlt=self.mlt
        GameScreen.gene=list(e)

        GameScreen.estat=1
        GameScreen.Comptador=0
        GameScreen.Counter=0
        GameScreen.ft=GameScreen.ft+1


        #print(GameScreen.gene)

    def freshrate(self,dt):
        # self.comptador=GameScreen.Comptador
        if GameScreen.estat==1:
            if GameScreen.Counter==0:
                #print(GameScreen.Comptador)
                self.mlt=GameScreen.mlt
                self.s=int((self.mlt-1)/2)
                self.prob_space = 350. / self.mlt
                self.prob=[]
                self.vel=[]
                self.evel=[]
                self.t = GameScreen.gtime
                self.ene = GameScreen.gene
                GameScreen.gpos = [l.tolist() for l in GameScreen.gpos]

                for i in range(len(GameScreen.gpos[0])):
                    x=[el[i] for el in GameScreen.gpos]
                    self.prob.append(x)

                for i in range(len(self.prob)-1):
                    x=[]
                    y=[]
                    for j in range(self.mlt):
                        x.append((self.prob[i+1][j]-self.prob[i][j])/(self.t[i+1]-self.t[i]))
                        y.append((self.ene[i+1][j]-self.ene[i][j])/(self.t[i+1]-self.t[i]))
                    self.vel.append(x)
                    self.evel.append(y)
                #print(self.evel)
                with self.canvas:
                    for i in range(self.mlt):
                        exec(f"self.tag_{i} = Label(pos=(30+i*self.prob_space+self.prob_space/2.-15,125),"
                             f" size=(30, 20),text='|' + str(i-self.s) + '>')")
                        exec(f'self.rec_{i} = Rectangle(pos=(30+i*self.prob_space,150), '
                             f'size=(self.prob_space-10, 250*self.prob[0][i]))')

                        exec(f"self.tag_{i} = Label(pos=(495+i*self.prob_space-15,125),"
                             f" size=(30, 20),text='|' + str(i-self.s) + '>')")
                        exec(f'self.erec_{i} = Rectangle(pos=(500+i*self.prob_space-self.prob_space/2.,'
                             f'300+self.ene[0][i]*100), '
                             f'size=(self.prob_space-10, 10))')

                        # exec(f'self.rec_{i} = Rectangle(pos=(55+i*self.prob_space,150), '
                        #      f'size=(self.prob_space-10, 250*self.prob[0][i]))')
                # print(len(vel),len(prob))
                # print(self.s)
                GameScreen.Counter = 1


            ti = self.t[GameScreen.Comptador]
            # if self.timer >= self.t[len(self.t) - 1]:
            #     GameScreen.estat = 0
            if ti < self.timer:
                GameScreen.Comptador = GameScreen.Comptador + 1
                if GameScreen.Comptador >= len(self.t) - 2:
                    GameScreen.estat=0

            # for i in range(self.mlt):
            x0, y0 = self.rec_0.size
            m, n = self.erec_0.pos
            self.rec_0.size = (x0, y0 + self.vel[GameScreen.Comptador][0] * 5*250 * dt)
            self.erec_0.pos = (m, n + self.evel[GameScreen.Comptador][0] * 5 * 100 * dt)

            x1, y1 = self.rec_1.size
            m, n = self.erec_1.pos
            self.rec_1.size = (x1, y1 + self.vel[GameScreen.Comptador][1] * 5 * 250*dt)
            self.erec_1.pos = (m, n + self.evel[GameScreen.Comptador][1] * 5 * 100 * dt)

            x2, y2 = self.rec_2.size
            m, n = self.erec_2.pos
            self.rec_2.size = (x2, y2 + self.vel[GameScreen.Comptador][2] * 5 * 250*dt)
            self.erec_2.pos = (m, n + self.evel[GameScreen.Comptador][2] * 5 * 100 * dt)

            if self.mlt>=5:
                x3, y3 = self.rec_3.size
                m, n = self.erec_0.pos
                self.rec_3.size = (x3, y3 + self.vel[GameScreen.Comptador][3] * 5 * 250*dt)
                self.erec_3.pos = (m, n + self.evel[GameScreen.Comptador][3] * 5 * 10 * dt)

                x4, y4 = self.rec_4.size
                m, n = self.erec_0.pos
                self.rec_4.size = (x4, y4 + self.vel[GameScreen.Comptador][4] * 5 * 250 * dt)
                self.erec_4.pos = (m, n + self.evel[GameScreen.Comptador][4] * 5 * 10 * dt)

            if self.mlt>=7:
                x5, y5 = self.rec_5.size
                m, n = self.erec_0.pos
                self.rec_5.size = (x5, y5 + self.vel[GameScreen.Comptador][5] * 5 * 250 * dt)
                self.erec_5.pos = (m, n + self.evel[GameScreen.Comptador][5] * 5 * 10 * dt)

                x6, y6 = self.rec_6.size
                m, n = self.erec_0.pos
                self.rec_6.size = (x6, y6 + self.vel[GameScreen.Comptador][6] * 5 * 250 * dt)
                self.erec_6.pos = (m, n + self.evel[GameScreen.Comptador][6] * 5 * 10 * dt)

            if self.mlt>=9:
                x7, y7 = self.rec_7.size
                m, n = self.erec_0.pos
                self.rec_7.size = (x7, y7 + self.vel[GameScreen.Comptador][7] * 5 *250* dt)
                self.erec_7.pos = (m, n + self.evel[GameScreen.Comptador][7] * 5 * 10 * dt)

                x8, y8 = self.rec_8.size
                m, n = self.erec_0.pos
                self.rec_8.size = (x8, y8 + self.vel[GameScreen.Comptador][8] * 5 * 250 * dt)
                self.erec_8.pos = (m, n + self.evel[GameScreen.Comptador][8] * 5 * 10 * dt)


            self.timer = self.timer + 5*dt

        else:
            pass
if __name__ == '__main__':
    tun_v5App().run()
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
from kivy.graphics import Color
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '900')
Config.set('graphics', 'height', '600')

Window.clearcolor = (0,0.08,0.31,1)


class tun_v7App(App):
    def build(self):
        self.title='Spin Tunneling'
        return MyScreenManager()

class MyScreenManager(ScreenManager):
    def  __init__(self, **kwargs):
        super(MyScreenManager, self).__init__(**kwargs)

        # with self.canvas:
        #     self.bg = Rectangle(source='Title.png', pos=(200,362), size=(400,176))

        self.get_screen('Tunneling').stpseudo_init()
        # self.get_screen("Game").gpseudo_init()



class StartingScreen(Screen):
    def __init__(self,**kwargs):
        super(StartingScreen,self).__init__(**kwargs)
        with self.canvas:
            self.bg = Rectangle(source='Title.png', pos=(220,370), size=(1196*0.3,579*0.3))
            # self.bg = Rectangle(source='Title.png', pos=(450-1196 * 0.3/1.6, 370), size=(1196 * 0.3, 579 * 0.3))

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

    H_type= NumericProperty(1)
    s= NumericProperty(1)
    a=NumericProperty(0.01)
    alpha=NumericProperty(0.1)


    def __init__(self,**kwargs):
        super(TunnelingScreen,self).__init__(**kwargs)
        # Window.clearcolor = (0, 0.08, 0.31, 1)

    def stpseudo_init(self):
        self.plot, self.axs =plt.subplots(1)
        self.axs.set(xlim=[-50,50],ylim=[0,1])


        self.plot = FigureCanvasKivyAgg(self.plot)
        self.box1.add_widget(self.plot)

    def transition_SScreen(self, *largs):
        self.manager.current = 'starting'


    def send_but(self):

        #print(self.a)
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
            if i != 0:
                self.y0.append(0 + 0j)
            else:
                self.y0.append(1 + 0j)

        t0 = -50.
        tf = 50.
        # self.alpha=0.1
        # self.alpha = float(self.ids.alpha_text.text)
        # self.a=0.01
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
        plt.xlabel("t'")
        plt.ylabel('Probability')
        plt.plot(temps, norma, label='Norma')
        plt.axhline(y=1, xmin=t0, xmax=tf, ls='dashed')
        for i in range(self.mlt):
            # plt.plot(t,mod2[i],label='|'+str(int(i-s))+'>')
            plt.plot(temps, mod2[i], label='|' + str(int(i - self.s)*(-1)) + '>')
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
    a=NumericProperty(0.01)
    alpha=NumericProperty(0.1)
    gtime=None
    gpos=None
    gene=None
    Comptador=None
    Counter=None
    # with canvas:
    #     en_t = Rectangle(source='energia.png', pos=(550, 450), size=(744 * 0.15, 200 * 0.15))
    #     pro_t = Rectangle(source='prob.png', pos=(100, 450), size=(1301 * 0.15, 200 * 0.15))

    def __init__(self,**kwargs):
        super(GameScreen,self).__init__(**kwargs)
        # CLOCK


    def gamepseudo_init(self):
        pass

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
            if i != 0:
                self.y0.append(0 + 0j)
            else:
                self.y0.append(1 + 0j)

        #print(self.a)
        # self.alpha=0.1
        # self.a=0.01
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
                ene_i.append(e.real)
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

        with self.canvas:
            self.en_t = Rectangle(source='energy.png', pos=(565, 450),size=(711*0.15,200*0.15))
            self.pro_t = Rectangle(source='prob2.png', pos=(100, 450),size=(1203*0.15, 200*0.15))
            # self.en_t = Rectangle(source='energy.png', pos=(220, 370), size=(1196 * 0.3, 579 * 0.3))
            # self.pro_t = Rectangle(source='prob.png', pos=(220, 370), size=(1196 * 0.3, 579 * 0.3))

        # print(GameScreen.gene)
        #for i in GameScreen.gene:
            #print(i)
    def freshrate(self,dt):
        # print(GameScreen.Counter)
        if GameScreen.estat==1:
            if GameScreen.Counter==0:
                # print(GameScreen.Comptador)
                self.mlt=int(GameScreen.mlt)
                self.s=(self.mlt-1)/2.
                self.prob_space = 350. / self.mlt
                self.prob=[]
                pos_i=(-1000,-1000)
                self.vel=[]
                self.evel=[]
                self.t = GameScreen.gtime
                self.ene = GameScreen.gene
                GameScreen.gpos = [l.tolist() for l in GameScreen.gpos]
                self.timer=-50

                if GameScreen.ft == 1:
                    with self.canvas:
                        self.tag_1 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(-1) + '>')
                        self.tag_2 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(-2) + '>')
                        self.tag0 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(0) + '>')
                        self.tag1 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(1) + '>')
                        self.tag2 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(2) + '>')
                        self.tag_52 = Label(pos=(-200, -200), size=(30, 20), text='|-5/2>')
                        self.tag_32 = Label(pos=(-200, -200), size=(30, 20), text='|-3/2>')
                        self.tag_12 = Label(pos=(-200, -200), size=(30, 20), text='|-1/2>')
                        self.tag12 = Label(pos=(-200, -200), size=(30, 20), text='|1/2>')
                        self.tag32 = Label(pos=(-200, -200), size=(30, 20), text='|3/2>')
                        self.tag52 = Label(pos=(-200, -200), size=(30, 20), text='|5/2>')

                        Color(255, 255, 255, 1, mode='rgba')
                        self.rec_2 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec_1 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec0 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec1 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec2 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec_52 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec_32 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec_12 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec12 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec32 = Rectangle(pos=(-200, -200), size=(10, 10))
                        self.rec52 = Rectangle(pos=(-200, -200), size=(10, 10))

                        self.etag_2 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(-2) + '>')
                        self.etag_1 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(-1) + '>')
                        self.etag0 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(0) + '>')
                        self.etag1 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(1) + '>')
                        self.etag2 = Label(pos=(-200, -200), size=(30, 20), text='|' + str(2) + '>')
                        self.etag_52 = Label(pos=(-200, -200), size=(30, 20), text='|-5/2>')
                        self.etag_32 = Label(pos=(-200, -200), size=(30, 20), text='|-3/2>')
                        self.etag_12 = Label(pos=(-200, -200), size=(30, 20), text='|-1/2>')
                        self.etag12 = Label(pos=(-200, -200), size=(30, 20), text='|1/2>')
                        self.etag32 = Label(pos=(-200, -200), size=(30, 20), text='|3/2>')
                        self.etag52 = Label(pos=(-200, -200), size=(30, 20), text='|5/2>')

                        Color(255, 255, 255, 1, mode='rgba')
                        self.erec_2 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec_1 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec0 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec1 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec2 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec_52 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec_32 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec_12 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec12 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec32 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))
                        self.erec52 = Rectangle(pos=(-200, -200), size=(self.prob_space-10, 10))

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


                # print(self.evel)

                if self.mlt==3:
                    #Movem aquells que volem que es mostrin
                    self.rec1.pos=(30,150)
                    self.rec0.pos = (30 + 1 * self.prob_space, 150)
                    self.rec_1.pos=(30+2*self.prob_space,150)

                    self.erec1.pos = (480 + 0 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][0] * 100)
                    self.erec0.pos = (480 + 1 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][1] * 100)
                    self.erec_1.pos = (480 + 2 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][2] * 100)


                    self.tag1.pos=(30+0*self.prob_space+self.prob_space/2.-15,125)
                    self.tag0.pos=(30+1*self.prob_space+self.prob_space/2.-15,125)
                    self.tag_1.pos=(30+2*self.prob_space+self.prob_space/2.-15,125)

                    self.etag1.pos = (475+0*self.prob_space-15,125)
                    self.etag0.pos = (475+1*self.prob_space-15,125)
                    self.etag_1.pos = (475+2*self.prob_space-15,125)


                    self.rec1.size = (self.prob_space-10, 250*self.prob[0][0])
                    self.rec0.size = (self.prob_space-10, 250*self.prob[0][1])
                    self.rec_1.size = (self.prob_space-10, 250*self.prob[0][2])

                    self.erec1.size = (self.prob_space-10, 10)
                    self.erec0.size = (self.prob_space-10, 10)
                    self.erec_1.size = (self.prob_space-10, 10)


                    #Apartem aquells que no volem que es mostrin
                    self.rec2.pos = pos_i
                    self.rec2.pos = pos_i
                    self.rec_12.pos = pos_i
                    self.rec12.pos = pos_i
                    self.rec_32.pos = pos_i
                    self.rec32.pos = pos_i
                    self.rec_52.pos = pos_i
                    self.rec52.pos = pos_i

                    self.erec_2.pos = pos_i
                    self.erec2.pos = pos_i
                    self.erec_12.pos = pos_i
                    self.erec12.pos = pos_i
                    self.erec_32.pos = pos_i
                    self.erec32.pos = pos_i
                    self.erec_52.pos = pos_i
                    self.erec52.pos = pos_i


                    self.tag_2.pos = pos_i
                    self.tag2.pos = pos_i
                    self.tag12.pos = pos_i
                    self.tag_12.pos = pos_i
                    self.tag32.pos = pos_i
                    self.tag_32.pos = pos_i
                    self.tag52.pos = pos_i
                    self.tag_52.pos = pos_i

                    self.etag_2.pos = pos_i
                    self.etag2.pos = pos_i
                    self.etag12.pos = pos_i
                    self.etag_12.pos = pos_i
                    self.etag32.pos = pos_i
                    self.etag_32.pos = pos_i
                    self.etag52.pos = pos_i
                    self.etag_52.pos = pos_i

                elif self.mlt == 5:
                    # Movem aquells que volem que es mostrin
                    self.rec2.pos = (30 , 150)
                    self.rec1.pos = (30 + 1 * self.prob_space, 150)
                    self.rec0.pos = (30 + 2 * self.prob_space, 150)
                    self.rec_1.pos = (30 + 3 * self.prob_space, 150)
                    self.rec_2.pos = (30 + 4 * self.prob_space, 150)

                    self.erec2.pos = (480 + 0 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][0] * 100)
                    self.erec1.pos = (480 + 1 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][1] * 100)
                    self.erec0.pos = (480 + 2 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][2] * 100)
                    self.erec_1.pos = (480 + 3 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][3] * 100)
                    self.erec_2.pos = (480 + 4 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][4] * 100)


                    self.tag2.pos = (30 + 0 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag1.pos = (30 + 1 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag0.pos = (30 + 2 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag_1.pos = (30 + 3 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag_2.pos = (30 + 4 * self.prob_space + self.prob_space / 2. - 15, 125)

                    self.etag2.pos = (475 + 0 * self.prob_space - 15, 125)
                    self.etag1.pos = (475 + 1 * self.prob_space - 15, 125)
                    self.etag0.pos = (475 + 2 * self.prob_space - 15, 125)
                    self.etag_1.pos = (475 + 3 * self.prob_space - 15, 125)
                    self.etag_2.pos = (475 + 4 * self.prob_space - 15, 125)


                    self.rec2.size = (self.prob_space - 10, 250 * self.prob[0][0])
                    self.rec1.size = (self.prob_space - 10, 250 * self.prob[0][1])
                    self.rec0.size = (self.prob_space - 10, 250 * self.prob[0][2])
                    self.rec_1.size = (self.prob_space - 10, 250 * self.prob[0][3])
                    self.rec_2.size = (self.prob_space - 10, 250 * self.prob[0][4])

                    self.erec2.size = (self.prob_space-10, 10)
                    self.erec1.size = (self.prob_space-10, 10)
                    self.erec0.size = (self.prob_space-10, 10)
                    self.erec_1.size = (self.prob_space-10, 10)
                    self.erec_2.size = (self.prob_space-10, 10)


                    # Apartem aquells que no volem que es mostrin

                    self.rec_12.pos = pos_i
                    self.rec12.pos = pos_i
                    self.rec_32.pos = pos_i
                    self.rec32.pos = pos_i
                    self.rec_52.pos = pos_i
                    self.rec52.pos = pos_i

                    self.erec_12.pos = pos_i
                    self.erec12.pos = pos_i
                    self.erec_32.pos = pos_i
                    self.erec32.pos = pos_i
                    self.erec_52.pos = pos_i
                    self.erec52.pos = pos_i


                    self.tag12.pos = pos_i
                    self.tag_12.pos = pos_i
                    self.tag32.pos = pos_i
                    self.tag_32.pos = pos_i
                    self.tag52.pos = pos_i
                    self.tag_52.pos = pos_i


                    self.etag12.pos = pos_i
                    self.etag_12.pos = pos_i
                    self.etag32.pos = pos_i
                    self.etag_32.pos = pos_i
                    self.etag52.pos = pos_i
                    self.etag_52.pos = pos_i

                elif self.mlt == 4:
                    # Movem aquells que volem que es mostrin
                    self.rec32.pos = (30, 150)
                    self.rec12.pos = (30 + 1 * self.prob_space, 150)
                    self.rec_12.pos = (30 + 2 * self.prob_space, 150)
                    self.rec_32.pos = (30 + 3 * self.prob_space, 150)

                    self.erec32.pos = (480 + 0 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][0] * 100)
                    self.erec12.pos = (480 + 1 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][1] * 100)
                    self.erec_12.pos = (480 + 2 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][2] * 100)
                    self.erec_32.pos = (480 + 3 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][3] * 100)


                    self.tag32.pos = (30 + 0 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag12.pos = (30 + 1 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag_12.pos = (30 + 2 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag_32.pos = (30 + 3 * self.prob_space + self.prob_space / 2. - 15, 125)

                    self.etag32.pos = (475 + 0 * self.prob_space - 15, 125)
                    self.etag12.pos = (475 + 1 * self.prob_space - 15, 125)
                    self.etag_12.pos = (475 + 2 * self.prob_space - 15, 125)
                    self.etag_32.pos = (475 + 3 * self.prob_space - 15, 125)


                    self.rec32.size = (self.prob_space - 10, 250 * self.prob[0][0])
                    self.rec12.size = (self.prob_space - 10, 250 * self.prob[0][1])
                    self.rec_12.size = (self.prob_space - 10, 250 * self.prob[0][2])
                    self.rec_32.size = (self.prob_space - 10, 250 * self.prob[0][3])

                    self.erec32.size = (self.prob_space-10, 10)
                    self.erec12.size = (self.prob_space-10, 10)
                    self.erec_12.size = (self.prob_space-10, 10)
                    self.erec_32.size = (self.prob_space-10, 10)

                    # Apartem aquells que no volem que es mostrin
                    self.rec_2.pos = pos_i
                    self.rec_1.pos = pos_i
                    self.rec0.pos = pos_i
                    self.rec1.pos = pos_i
                    self.rec2.pos = pos_i
                    self.rec_52.pos = pos_i
                    self.rec52.pos = pos_i

                    self.erec_2.pos = pos_i
                    self.erec_1.pos = pos_i
                    self.erec0.pos = pos_i
                    self.erec1.pos = pos_i
                    self.erec2.pos = pos_i
                    self.erec_52.pos = pos_i
                    self.erec52.pos = pos_i

                    self.tag_2.pos = pos_i
                    self.tag_1.pos = pos_i
                    self.tag0.pos = pos_i
                    self.tag1.pos = pos_i
                    self.tag2.pos = pos_i
                    self.tag_52.pos = pos_i
                    self.tag52.pos = pos_i

                    self.etag_2.pos = pos_i
                    self.etag_1.pos = pos_i
                    self.etag0.pos = pos_i
                    self.etag1.pos = pos_i
                    self.etag2.pos = pos_i
                    self.etag_52.pos = pos_i
                    self.etag52.pos = pos_i

                elif self.mlt == 6:
                    # Movem aquells que volem que es mostrin
                    self.rec52.pos = (30 + 0 * self.prob_space, 150)
                    self.rec32.pos = (30 + 1 * self.prob_space, 150)
                    self.rec12.pos = (30 + 2 * self.prob_space, 150)
                    self.rec_12.pos = (30 + 3 * self.prob_space, 150)
                    self.rec_32.pos = (30 + 4 * self.prob_space, 150)
                    self.rec_52.pos = (30 + 5 * self.prob_space, 150)

                    self.erec52.pos = (480 + 0 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][0] * 100)
                    self.erec32.pos = (480 + 1 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][1] * 100)
                    self.erec12.pos = (480 + 2 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][2] * 100)
                    self.erec_12.pos = (480 + 3 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][3] * 100)
                    self.erec_32.pos = (480 + 4 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][4] * 100)
                    self.erec_52.pos = (480 + 5 * self.prob_space - self.prob_space / 2., 370 + self.ene[0][5] * 100)

                    self.tag52.pos = (30 + 0 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag32.pos = (30 + 1 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag12.pos = (30 + 2 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag_12.pos = (30 + 3 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag_32.pos = (30 + 4 * self.prob_space + self.prob_space / 2. - 15, 125)
                    self.tag_52.pos = (30 + 5 * self.prob_space + self.prob_space / 2. - 15, 125)

                    self.etag52.pos = (475 + 0 * self.prob_space - 15, 125)
                    self.etag32.pos = (475 + 1 * self.prob_space - 15, 125)
                    self.etag12.pos = (475 + 2 * self.prob_space - 15, 125)
                    self.etag_12.pos = (475 + 3 * self.prob_space - 15, 125)
                    self.etag_32.pos = (475 + 4 * self.prob_space - 15, 125)
                    self.etag_52.pos = (475 + 5 * self.prob_space - 15, 125)

                    self.rec52.size = (self.prob_space - 10, 250 * self.prob[0][0])
                    self.rec32.size = (self.prob_space - 10, 250 * self.prob[0][1])
                    self.rec12.size = (self.prob_space - 10, 250 * self.prob[0][2])
                    self.rec_12.size = (self.prob_space - 10, 250 * self.prob[0][3])
                    self.rec_32.size = (self.prob_space - 10, 250 * self.prob[0][4])
                    self.rec_52.size = (self.prob_space - 10, 250 * self.prob[0][5])

                    self.erec52.size = (self.prob_space-10, 10)
                    self.erec32.size = (self.prob_space-10, 10)
                    self.erec12.size = (self.prob_space-10, 10)
                    self.erec_12.size = (self.prob_space-10, 10)
                    self.erec_32.size = (self.prob_space-10, 10)
                    self.erec_52.size = (self.prob_space-10, 10)


                    # Apartem aquells que no volem que es mostrin
                    self.rec_2.pos = pos_i
                    self.rec_1.pos = pos_i
                    self.rec0.pos = pos_i
                    self.rec1.pos = pos_i
                    self.rec2.pos = pos_i


                    self.erec_2.pos = pos_i
                    self.erec_1.pos = pos_i
                    self.erec0.pos = pos_i
                    self.erec1.pos = pos_i
                    self.erec2.pos = pos_i


                    self.tag_2.pos = pos_i
                    self.tag_1.pos = pos_i
                    self.tag0.pos = pos_i
                    self.tag1.pos = pos_i
                    self.tag2.pos = pos_i


                    self.etag_2.pos = pos_i
                    self.etag_1.pos = pos_i
                    self.etag0.pos = pos_i
                    self.etag1.pos = pos_i
                    self.etag2.pos = pos_i

                GameScreen.Counter = 1

                # with self.canvas:
                #     self.tag_1 = Label(pos=(-200,-200),size=(30, 20),text='|' + str(-1) + '>')")
                #     self.tag_2 = Label(pos=(-200, -200),size=(30, 20), text='|' + str(-2) + '>')
                #     self.tag0 = Label(pos=(-200, -200),size=(30, 20), text='|' + str(0) + '>')
                #     self.tag1 = Label(pos=(-200, -200),size=(30, 20), text='|' + str(1) + '>')
                #     self.tag2 = Label(pos=(-200, -200),size=(30, 20), text='|' + str(2) + '>')
                #     self.tag_32 = Label(pos=(-200, -200),size=(30, 20), text='|-3/2>')
                #     self.tag_12 = Label(pos=(-200, -200),size=(30, 20), text='|-1/2>')
                #     self.tag12 = Label(pos=(-200, -200),size=(30, 20), text='|1/2>')
                #     self.tag32 = Label(pos=(-200, -200),size=(30, 20), text='|3/2>')
                #
                #     Color(0, 21, 79, 0.8, mode='rgba')
                #     self.rec_2 = Rectangle(pos=(-200,-200), size=(10, 10))
                #     self.rec_1 = Rectangle(pos=(-200, -200), size=(10, 10))
                #     self.rec0 = Rectangle(pos=(-200, -200), size=(10, 10))
                #     self.rec1 = Rectangle(pos=(-200, -200), size=(10, 10))
                #     self.rec2 = Rectangle(pos=(-200, -200), size=(10, 10))
                #     self.rec_32 = Rectangle(pos=(-200, -200), size=(10, 10))
                #     self.rec_12 = Rectangle(pos=(-200, -200), size=(10, 10))
                #     self.rec12 = Rectangle(pos=(-200, -200), size=(10, 10))
                #     self.rec32 = Rectangle(pos=(-200, -200), size=(10, 10))



                # with self.canvas:
                #
                #     for i in range(self.mlt):
                #
                #         exec(f"self.tag_{i} = Label(pos=(30+i*self.prob_space+self.prob_space/2.-15,125),"
                #              f" size=(30, 20),text='|' + str(i-self.s) + '>')")
                #         Color(0,21,79,0.8, mode='rgba')
                #         exec(f'self.rec_{i} = Rectangle(pos=(30+i*self.prob_space,150), '
                #              f'size=(self.prob_space-10, 250*self.prob[0][i]))')
                #
                #         exec(f"self.tag_{i} = Label(pos=(475+i*self.prob_space-15,125),"
                #              f" size=(30, 20),text='|' + str(i-self.s) + '>')")
                #         exec(f'self.erec_{i} = Rectangle(pos=(480+i*self.prob_space-self.prob_space/2.,'
                #              f'370+self.ene[0][i]*100), '
                #              f'size=(self.prob_space-10, 10))')

                        # exec(f'self.rec_{i} = Rectangle(pos=(55+i*self.prob_space,150), '
                        #      f'size=(self.prob_space-10, 250*self.prob[0][i]))')
                # print(len(vel),len(prob))
                # print(self.s)




            ti = self.t[GameScreen.Comptador]
            # if self.timer >= self.t[len(self.t) - 1]:
            #     GameScreen.estat = 0
            # print(ti)
            if ti < self.timer:
                GameScreen.Comptador = GameScreen.Comptador + 1
                if GameScreen.Comptador >= len(self.t) - 2:
                    GameScreen.estat=0

            if self.mlt==3:
                x0, y0 = self.rec1.size
                m, n = self.erec1.pos
                self.rec1.size = (x0, y0 + self.vel[GameScreen.Comptador][0] * 5 * 250 * dt)
                self.erec1.pos = (m, n + self.evel[GameScreen.Comptador][0] * 5 * 100 * dt)

                x1, y1 = self.rec0.size
                m, n = self.erec0.pos
                self.rec0.size = (x1, y1 + self.vel[GameScreen.Comptador][1] * 5 * 250 * dt)
                self.erec0.pos = (m, n + self.evel[GameScreen.Comptador][1] * 5 * 100 * dt)

                x2, y2 = self.rec_1.size
                m, n = self.erec_1.pos
                self.rec_1.size = (x2, y2 + self.vel[GameScreen.Comptador][2] * 5 * 250 * dt)
                self.erec_1.pos = (m, n + self.evel[GameScreen.Comptador][2] * 5 * 100 * dt)




            if self.mlt==5:
                x0, y0 = self.rec2.size
                m, n = self.erec2.pos
                self.rec2.size = (x0, y0 + self.vel[GameScreen.Comptador][0] * 5 * 250 * dt)
                self.erec2.pos = (m, n + self.evel[GameScreen.Comptador][0] * 5 * 100 * dt)

                x1, y1 = self.rec1.size
                m, n = self.erec1.pos
                self.rec1.size = (x1, y1 + self.vel[GameScreen.Comptador][1] * 5 * 250 * dt)
                self.erec1.pos = (m, n + self.evel[GameScreen.Comptador][1] * 5 * 100 * dt)

                x2, y2 = self.rec0.size
                m, n = self.erec0.pos
                self.rec0.size = (x2, y2 + self.vel[GameScreen.Comptador][2] * 5 * 250 * dt)
                self.erec0.pos = (m, n + self.evel[GameScreen.Comptador][2] * 5 * 100 * dt)

                x3, y3 = self.rec_1.size
                m, n = self.erec_1.pos
                self.rec_1.size = (x3, y3 + self.vel[GameScreen.Comptador][3] * 5 * 250*dt)
                self.erec_1.pos = (m, n + self.evel[GameScreen.Comptador][3] * 5 * 10 * dt)

                x4, y4 = self.rec_2.size
                m, n = self.erec_2.pos
                self.rec_2.size = (x4, y4 + self.vel[GameScreen.Comptador][4] * 5 * 250 * dt)
                self.erec_2.pos = (m, n + self.evel[GameScreen.Comptador][4] * 5 * 10 * dt)

            if self.mlt==4:
                x0, y0 = self.rec32.size
                m, n = self.erec32.pos
                self.rec32.size = (x0, y0 + self.vel[GameScreen.Comptador][0] * 5 * 250 * dt)
                self.erec32.pos = (m, n + self.evel[GameScreen.Comptador][0] * 5 * 100 * dt)

                x1, y1 = self.rec12.size
                m, n = self.erec12.pos
                self.rec12.size = (x1, y1 + self.vel[GameScreen.Comptador][1] * 5 * 250 * dt)
                self.erec12.pos = (m, n + self.evel[GameScreen.Comptador][1] * 5 * 100 * dt)

                x2, y2 = self.rec_12.size
                m, n = self.erec_12.pos
                self.rec_12.size = (x2, y2 + self.vel[GameScreen.Comptador][2] * 5 * 250 * dt)
                self.erec_12.pos = (m, n + self.evel[GameScreen.Comptador][2] * 5 * 100 * dt)

                x3, y3 = self.rec_32.size
                m, n = self.erec_32.pos
                self.rec_32.size = (x3, y3 + self.vel[GameScreen.Comptador][3] * 5 * 250 * dt)
                self.erec_32.pos = (m, n + self.evel[GameScreen.Comptador][3] * 5 * 10 * dt)

            if self.mlt==6:
                x0, y0 = self.rec52.size
                m, n = self.erec52.pos
                self.rec52.size = (x0, y0 + self.vel[GameScreen.Comptador][0] * 5 * 250 * dt)
                self.erec52.pos = (m, n + self.evel[GameScreen.Comptador][0] * 5 * 100 * dt)

                x1, y1 = self.rec32.size
                m, n = self.erec32.pos
                self.rec32.size = (x1, y1 + self.vel[GameScreen.Comptador][1] * 5 * 250 * dt)
                self.erec32.pos = (m, n + self.evel[GameScreen.Comptador][1] * 5 * 100 * dt)

                x2, y2 = self.rec12.size
                m, n = self.erec12.pos
                self.rec12.size = (x2, y2 + self.vel[GameScreen.Comptador][2] * 5 * 250 * dt)
                self.erec12.pos = (m, n + self.evel[GameScreen.Comptador][2] * 5 * 100 * dt)

                x3, y3 = self.rec_12.size
                m, n = self.erec_12.pos
                self.rec_12.size = (x3, y3 + self.vel[GameScreen.Comptador][3] * 5 * 250 * dt)
                self.erec_12.pos = (m, n + self.evel[GameScreen.Comptador][3] * 5 * 10 * dt)

                x4, y4 = self.rec_32.size
                m, n = self.erec_32.pos
                self.rec_32.size = (x4, y4 + self.vel[GameScreen.Comptador][4] * 5 *250* dt)
                self.erec_32.pos = (m, n + self.evel[GameScreen.Comptador][4] * 5 * 10 * dt)

                x5, y5 = self.rec_52.size
                m, n = self.erec_32.pos
                self.rec_52.size = (x5, y5 + self.vel[GameScreen.Comptador][5] * 5 * 250 * dt)
                self.erec_52.pos = (m, n + self.evel[GameScreen.Comptador][5] * 5 * 10 * dt)

            print(self.timer, ti)
            #self.timer = self.timer + 5*dt


        else:
            pass
if __name__ == '__main__':
    tun_v7App().run()
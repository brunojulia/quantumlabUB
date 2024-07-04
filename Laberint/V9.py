import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics import Color
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView

import numpy as np
import math
import random
from PIL import Image





class LaberintApp(App):
    def build(self):
        return Inici()





class Inici(BoxLayout):
    """Primera pantalla del joc: aquí es poden llegir les instruccions o començar a jugar"""


    def __init__(self, **kwargs):
        super(Inici, self).__init__(**kwargs)

        self.orientation = "vertical"

        intruccions = Button(text='Instruccions', size_hint=(0.5, 0.25), pos_hint={'center_x': 0.5})
        intruccions.bind(on_press=self.mostrar_instruccions)
        self.add_widget(intruccions)

        horta = Button(text='Laberint d\'Horta, Barcelona 1808', size_hint=(0.5, 0.25), pos_hint={'center_x': 0.5})
        horta.bind(on_press=self.joc_horta)
        self.add_widget(horta)

        wakefield = Button(text='Laberint de Wakefield, Yorkshire 2013', size_hint=(0.5, 0.25), pos_hint={'center_x': 0.5})
        wakefield.bind(on_press=self.joc_wakefield)
        self.add_widget(wakefield)
    
    def joc_horta(self, instance):
        self.joc = 1
        self.finestra_jugar(instance)
    
    def joc_wakefield(self, instance):
        self.joc = 2
        self.finestra_jugar(instance)


    def mostrar_instruccions(self, instance):
        instruccions = Label(text='L\'objectiu del joc és sortir del laberint.\n'
                 'Utilitza les fletxes per a moure la partícula.\n'
                 'Però vigila! La partícula és quàntica i per tant la seva funció d\'ona patirà deformacions a mesura que passi el temps.\n'
                 'El que veuràs a la pantalla és el mòdul al quadrat de la funció d\'ona, que et donarà la probabilitat de trobar la partícula en aquella posició.\n'
                 'Prem "Mesura" per a mesurar la posició exacte de la partícula.\n'
                 )
        instruccions.bind(size=instruccions.setter('text_size'))
        scrollview = ScrollView()
        scrollview.add_widget(instruccions)
        popup = Popup(title='Instruccions',
              content=scrollview,
              size_hint=(0.5, 0.5))
        popup.open()

    def finestra_jugar(self, instance):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Finestra(self.joc))


class Finestra(BoxLayout):
    """Interfase de joc"""
    def __init__(self, joc, **kwargs):
        super(Finestra, self).__init__(**kwargs)
        tauler = Tauler(joc)
        self.joc = joc
        self.add_widget(Fletxes(tauler))
        self.add_widget(tauler)



class Fletxes(BoxLayout):

    """ Fletxes per a la partícula """
    
    def __init__(self, tauler, **kwargs):
        super().__init__(**kwargs)

        self.orientation = "vertical"
        self.tauler = tauler

        esquerra = Button(text='<-')
        esquerra.bind(on_press=self.tauler.on_esquerra)
        dreta = Button(text='->')
        dreta.bind(on_press=self.tauler.on_dreta)
        amunt = Button(text='^')
        amunt.bind(on_press=self.tauler.on_amunt)
        avall = Button(text='')
        avall.bind(on_press=self.tauler.on_avall)

        amuntavall = BoxLayout(orientation='vertical')
        amuntavall.add_widget(amunt)
        amuntavall.add_widget(avall)

        fletxes = BoxLayout(orientation='horizontal')
        fletxes.add_widget(esquerra)
        fletxes.add_widget(amuntavall)
        fletxes.add_widget(dreta)
        self.add_widget(fletxes)

        mesura = Button(text='Mesura')
        mesura.bind(on_press=self.tauler.mesura)
        self.add_widget(mesura)

        temps = Button(text='Temps')
        temps.bind(on_press=self.tauler.temps2)
        self.add_widget(temps)

        inici = Button(text='Inici')
        inici.bind(on_press=self.inici)
        self.add_widget(inici)

        self.size_hint=(.25,.5)
        self.pos_hint={'center_y':0.5}


    def inici(self, instance):
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Inici())



class Tauler(RelativeLayout):

    def __init__(self, joc, **kwargs):
        super().__init__(**kwargs)

        self.nivell = 80
        self.mida = 20
        self.funcona = []
        self.joc = joc

        if self.joc == 1:
            imagen = Image.open("horta.png")
        if self.joc == 2:
            imagen = Image.open("wakefield.jpg")
        imagen = imagen.resize((self.nivell,self.nivell))
        imagen = imagen.convert("L")
        pixeles = np.array(imagen)
        self.V = np.ones((self.nivell, self.nivell))
        self.V[pixeles > 128 ] = 0

        self.func = np.zeros((self.nivell, self.nivell), dtype=np.complex)
        self.func[40, 40] = 1
        self.func2 = np.zeros((self.nivell, self.nivell), dtype=np.complex)
        self.func2[40, 40] = 1
        self.prob = np.zeros((self.nivell, self.nivell))
        self.prob[40, 40] = 1

        self.particula()

    def laberint(self, **kwargs):
        self.parets = []
        self.final = []
        with self.canvas:
            Color(0.6, 0.4, 0.2)
            self.fons=Rectangle(pos=(0, 0), size=(self.nivell*self.mida, self.nivell*self.mida))
            for j in range(self.nivell):
                for i in range(self.nivell):
                    if not self.V[j][i] == 0:
                        Color(0, 0.3, 0)
                        self.parets.append([Rectangle(pos=(i*self.mida, j*self.mida), size=(self.mida,self.mida)), i, j])
                    Color(1, 0, 0)
                    if self.joc == 1:
                        if j>self.nivell-5 and i>self.nivell-5:
                            self.final.append([Rectangle(pos=(i*self.mida, j*self.mida), size=(self.mida,self.mida)), i, j])
                    if self.joc == 2:
                        if (j>65 and j<70) and (i>35 and i<40):
                            self.final.append([Rectangle(pos=(i*self.mida, j*self.mida), size=(self.mida,self.mida)), i, j])
        self.on_size()

    def particula(self):
        self.canvas.clear()
        self.laberint()
        ppi_x = self.width/2 - self.nivell*self.mida/2
        ppi_y = self.height/2 - self.nivell*self.mida/2
        max_prob = 0
        for j in range(self.nivell):
            for i in range(self.nivell):
                if self.V[j][i] == 0:
                    if self.prob[j][i] > max_prob:
                        max_prob = self.prob[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                if not self.prob[j][i] == 0:
                    if self.V[j][i] == 0:
                        with self.canvas:
                            Color(1, 1, 1, (1-0.03)/(max_prob)*self.prob[j][i]+0.03)
                            self.funcona.append([Rectangle(pos=(i*self.mida+ppi_x, j*self.mida+ppi_y), size=(self.mida,self.mida)), i, j])

    def on_size(self, *args):
        self.mida = min(self.width/self.nivell, self.height/self.nivell)
        ppi_x = self.width/2 - self.nivell*self.mida/2
        ppi_y = self.height/2 - self.nivell*self.mida/2
        for i in self.parets:
            i[0].size = (self.mida, self.mida)
            i[0].pos = (i[1]*self.mida+ppi_x, i[2]*self.mida+ppi_y)
        for i in self.funcona:
            i[0].size = (self.mida, self.mida)
            i[0].pos = (i[1]*self.mida+ppi_x, i[2]*self.mida+ppi_y)
        self.fons.size = (self.nivell*self.mida, self.nivell*self.mida)
        self.fons.pos = (ppi_x, ppi_y)
        for i in self.final:
            i[0].size = (self.mida, self.mida)
            i[0].pos = (i[1]*self.mida+ppi_x, i[2]*self.mida+ppi_y)


    def on_dreta(self, instance):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if i < self.nivell-1:
                    if self.V[j][i+1] == 0:
                        self.func2[j][i+1] += self.func[j][i]
                    else:
                        self.func2[j][i] += self.func[j][i]
                if i == self.nivell-1:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i]=0
        self.temps()

    def on_esquerra(self, instance):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if i > 0:
                    if self.V[j][i-1] == 0:
                        self.func2[j][i-1] += self.func[j][i]
                    else:  
                        self.func2[j][i] += self.func[j][i]
                if i == 0:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i]=0
        self.temps()

    def on_amunt(self, instance):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if j < self.nivell-1:
                    if self.V[j+1][i] == 0:
                        self.func2[j+1][i] += self.func[j][i]
                    else:
                        self.func2[j][i] += self.func[j][i]
                if j == self.nivell-1:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i] = 0
        self.temps()
    
    def on_avall(self, temps):
        for j in range(self.nivell):
            for i in range(self.nivell):
                if j > 0:
                    if self.V[j-1][i] == 0:
                        self.func2[j-1][i] += self.func[j][i]
                    else:    
                        self.func2[j][i] += self.func[j][i]
                if j == 0:
                    self.func2[j][i] += self.func[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.func2[j][i]=0
        self.temps()
    




    def temps(self):

        hbar = 1
        m = hbar**2/(2*61.04212604) # massa de l'electró
        t = 100
        delta = self.mida

        coef = (t*hbar)/(1j*2*m*delta**2)

        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func2[j][i] = self.func[j][i] * ( 1 + self.V[j][i]*t/(1j*hbar) + (2*t*hbar)/(1j*m*delta**2) ) 
                if j<self.nivell-1:
                    self.func2[j][i] -= coef * self.func[j+1][i]
                if j>0:
                    self.func2[j][i] -= coef * self.func[j-1][i]
                if i<self.nivell-1:
                    self.func2[j][i] -= coef * self.func[j][i+1]
                if i>0:
                    self.func2[j][i] -= coef * self.func[j][i-1]

        suma=0
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.func[j][i] = self.func2[j][i]
                self.prob[j][i] = (abs(self.func2[j][i]))**2
                suma += self.prob[j][i]
        for j in range(self.nivell):
            for i in range(self.nivell):
                self.prob[j][i] = self.prob[j][i] / suma
                self.func[j][i] = self.func[j][i] / suma
                self.func2[j][i] = 0

        self.particula()

    def temps2(self, instance):
        self.temps()
    
    def mesura(self, instance):
        # Aplanar las matrices lab y prob en listas unidimensionales, rastreando las posiciones
        func_flat = [(j, i) for j, sublist in enumerate(self.func) for i, item in enumerate(sublist)]
        prob_flat = [item for sublist in self.prob for item in sublist]
        # Seleccionar una posición de lab con la probabilidad correspondiente en prob
        selected_position = random.choices(func_flat, weights=prob_flat, k=1)[0]
        while self.V[selected_position[0]][selected_position[1]] != 0:
            selected_position = random.choices(func_flat, weights=prob_flat, k=1)[0]
        # Actualitzar la funció d'ona
        self.func = [[0 for j in range(self.nivell)] for i in range(self.nivell)]
        self.func[selected_position[0]][selected_position[1]] = 1
        self.prob = [[0 for j in range(self.nivell)] for i in range(self.nivell)]
        self.prob[selected_position[0]][selected_position[1]] = 1
        self.particula()
        #Mirar si ha acabat el joc
        for i in self.final:
            if i[1] == selected_position[1] and i[2] == selected_position[0]:
                self.final_joc()
    
    def final_joc(self):
        popup = Popup(title='Felicitats!',
              content=Label(text='Has aconseguit sortir del laberint!'),
              size_hint=(0.5, 0.5))
        popup.open()
        App.get_running_app().root.clear_widgets()
        App.get_running_app().root.add_widget(Inici())





LaberintApp().run()

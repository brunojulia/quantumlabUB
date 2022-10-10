import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, NumericProperty, ListProperty
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout


from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.providers.aer import AerSimulator

import math as m
import matplotlib.pyplot as plt
import numpy as np

from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

Builder.load_file("1.kv")

#--------------------------------------------------------------------------------------
#FUNCIONES PARA CALCULAR EVOLUCIÓN TEMPORAL:
def qft(qc,n):
    '''performs quantum fourier transform on circuit qc
    n: number of qubits'''
    #iterates through each qubit starting at the bottom
    for j in range(n-1, -1, -1): 
    # h gate on each qubit    
        qc.h(j)
    #iterates through each qubit above current one
    #adds controlled P gate, phase pi/2**()
        for i in range(j-1, -1, -1):
            qc.cp(m.pi/2**(j-i), i,j)
    #creates barrier before next H gate
        qc.barrier()
    #changes order of qubits    
    for i in range(int(n/2)):
        qc.swap(i, n-i-1)

def inverse_qft(qc,n):
    '''performs inverse quantum fourier transform on circuit qc
    same as fourier, reversed order and phases are negative
    n: number of qubits'''
    #changes order of qubits    
    for i in range(int(n/2)):
        qc.swap(i, n-i-1)
    #iterates through each qubit starting at the bottom
    for j in range(0, n): 
    # h gate on each qubit    
        qc.h(j)
    #iterates through each qubit above current one
    #adds controlled P gate, phase -pi/2**()
        for i in range(j+1, n):
            qc.cp(-m.pi/2**(i-j), i,j)
    #creates barrier before next H gate
        qc.barrier()

def crz(qc, theta, control, target):
#controlled rz
# https://qiskit.org/textbook/ch-gates/more-circuit-identities.html
  qc.rz(theta/2,target)
  qc.cx(control,target)
  qc.rz(theta/2,target)
  qc.cx(control,target)
 

def tunnel(qc, t, n, h, well):
   '''
   Creates circuit qc needed to simulate schrodinger equation for 2 qubits

   qc: quantum circuit
   t: time increment
   n: number of time steps
   h: height of potential wells
   well: determines type of potential, if = 0, step potential, if = 1, double well
  
   '''
   for i in range(n):
     #fourier transf:
     qft(qc, 2)

     #kinetic operator  
     qc.rz(-t*m.pi**2,1)
     qc.rz(-t*0.25*m.pi**2,0)
     crz(qc,t*m.pi**2,0,1)

     #inverse fourier transform
     inverse_qft(qc, 2)

     #potential:
     #rotation around z axis
     if well == 0 or well == 1:
       qc.rz(2*h*t, 1-well)
     if well == 2:
       qc.rz(2*h*t, 0)
       qc.rz(2*h*t, 1)


    
   qc.measure_all()

#--------------------------------------------------------------------------------------
#CREA GRAFICA INICIAL
#potencial escalon
x_step = [0,1,2,2,3,4]
y_step = [1,1,1,0,0,0]
        
plt.xlim(0,4)
plt.ylim(-0.05,1.01)
plt.plot(x_step, y_step, "k--")
plt.subplots_adjust(left=0.06, right=0.95, top=0.98, bottom=0.11)    
plt.xticks([i for i in range(4)],[""])   
plt.tick_params(axis='both', which='major', labelsize=18)       
plt.tick_params(axis='x',which='both',bottom=False, top=False,labelbottom=False) 

class MyFigure(FigureCanvasKivyAgg):
    def __init__(self, **kwargs):
        super().__init__(plt.gcf(), **kwargs)

           
#--------------------------------------------------------------------------------------
class Game(Widget):
    #potential height (0-50)
    V_height = ObjectProperty(0) 

    #type of potential
    # 0 -> step
    # 1 -> single well
    # 2 -> double well
    V_type = ObjectProperty(0)
    
    # contains probability for each position and time 
    r = ListProperty()
    
    #counts time iteration
    counter = NumericProperty(0)
    
    def press(self, V):
        #sets type of potential, updates potential graph
        #tied to potential buttons
        ids_v = [self.step, self.double, self.single]
        x_step = [0,1,2,2,3,4]
        y_step = [1,1,1,0,0,0]
        
        x_double = [0,1,1,2,3,3,4]
        y_double = [1,1,0,0,0,1,1] 
        
        x_single = [0,1,1,2,2,3,3,4]
        y_single = [1,1,0,0,1,1,0,0]
        
        #set type of potential
        self.V_type = V
        
        #clear previous potential graph
        plt.cla()
        plt.xlim(0,4)
        plt.ylim(-0.05,1.01)
        
        #plot new graph
        if self.V_type == 0:
            plt.plot(x_step, y_step, "k--")
            
        elif self.V_type == 1:
            plt.plot(x_single, y_single, "k--")
            
        elif self.V_type == 2:
            plt.plot(x_double, y_double, "k--")
            
        #changes padding around graph 
        plt.subplots_adjust(left=0.06, right=0.95, top=0.98, bottom=0.11)    
        
        #removes ticks on x axis
        plt.tick_params(axis='x',which='both',bottom=False, top=False,labelbottom=False) 
        plt.xticks([i for i in range(4)],[""]) 
        
        #adds plot to layout
        canvas_plot=FigureCanvasKivyAgg(plt.gcf())
        self.axes.clear_widgets()
        self.axes.add_widget(canvas_plot)	
        
        #changes colors on potential buttons        
        for i in ids_v:
            i.background_color = [0.3, 0.3 , 0.3 ,1]
        ids_v[V].background_color = [1.0, 0.0, 0.0, 1.0]
        
        
        
        
    def V_slider(self, *args):
        #sets value for potential height
        #tied to potential slider
        self.V_height = args[1]
        self.potential_text.text = "Potential height: " + str(int(args[1]))
    
    def position_slider(self, *args, slider):
        #set initial state
        #tied to sliders
        ids_text = [self.text_00, self.text_01, self.text_10, self.text_11]
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        
        ids_text[slider].text = str(round(args[1], 2))
        if args[1] == 0:
            ids_sliders[slider].value_track = False
        else: 
            ids_sliders[slider].value_track = True
                
    def normalize(self):
        #normalizes position sliders, locks positions
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        norm = sum([i.value for i in ids_sliders])
        for i in ids_sliders:
            i.disabled = True
            if norm != 0:
                i.value = i.value/norm
            else: 
                i.value = 0.25
        self.calculate_button.disabled = False  
        self.initial_button.disabled = False    
        self.normalize_button.disabled = True  
                
            
    def release(self):
        # unlocks position sliders, potential buttons
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        self.step.disabled = False
        self.single.disabled = False
        self.double.disabled = False
        self.potential_slide.disabled = False
        
        self.normalize_button.disabled = False
        self.play_button.disabled = True
        
        for i in ids_sliders:
            i.disabled = False
    
    def calculate(self):
        #calculates probabilites for each position, time interval
        
        self.normalize_button.disabled = True
        self.initial_button.disabled = True
        self.calculate_button.disabled = True
        
        self.step.disabled = True
        self.single.disabled = True
        self.double.disabled = True
        self.potential_slide.disabled = True
        
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        qc = QuantumCircuit(2)
        sim = Aer.get_backend('aer_simulator') 
        delt = 0.1
        state = ["00", "01", "10", "11"]
        
        self.r = [[0 for i in range(50)] for i in range(4)]
        
        for i in range(50):
            qc = QuantumCircuit(2)
            qc.initialize([i.value**0.5 for i in ids_sliders])
            tunnel(qc, delt, i, self.V_height, self.V_type)
            result = sim.run(qc, shots=2**13).result()
            counts = result.get_counts()
            
            for j in range(len(state)):
                if state[j] in counts.keys():
                    self.r[j][i] = counts[state[j]]/(2**13)          
        self.play_button.disabled = False
                
    def update(self, dt):
        #sets position sliders according to calculated probabilites
        
        ids_sliders = [self.slider_00, self.slider_01, self.slider_10, self.slider_11]
        
        #stops updating sliders after 49 iterations:
        if self.counter > 49:
            self.play_button.disabled = False
            self.initial_button.disabled = False
            self.time.text = " "
            return False
        else: 
            self.time.text = str(round(0.1*self.counter, 1)) + " s"
            for i in range(4):
                ids_sliders[i].value =  self.r[i][self.counter]
            self.counter += 1

    def play(self):
        #calls update function every 0.1 sec 
        self.counter = 0
        Clock.schedule_interval(self.update, 0.1)  
        self.play_button.disabled = True
         
        
            
class sliide(App):
    def build(self):
        return Game()
    
if __name__ == "__main__":
    sliide().run()
    
    
    
#add boxes around sections
#images for buttons?


#3 qubits
    
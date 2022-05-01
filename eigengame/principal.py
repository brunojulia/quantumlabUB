import kivy
from kivy.config import Config
Config.set('input', 'mouse', 'mouse,multitouch_on_demand') #deactivate left button
from kivy.app import App 
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.uix.floatlayout import FloatLayout 
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.properties import ObjectProperty
from kivy.animation import Animation 
from kivy.uix.image import Image
import matplotlib.pyplot as plt 
import math as math
import numpy as np  
import scipy
import scipy.special 
import random 
from numba import jit
import time




     
class GameWindow(Screen):
        float_plot=ObjectProperty(None)
        measure_layout=ObjectProperty(None)
        value_n=0#if no button is pressed
        hooke_constant=5 #if it isn't slide it
        x0_harmonic=0 #if there is no sliding
        center_b=0 
        width_b=1
        Vb=20 #if the slides for the barrier aren't activated
        left_x_w=-0.4 
        right_x_w=0.4 
        Vw=20 #if the slides for the water aren't activated
        potential_type1="Free particle" #if no potential button is pressed
        potential_type2="Free particle"  
        e_position=[None]*100  #we put it here so its global, maximum level is 100
        #We create the list of values x outside so its faster(no need to do it every time)
        n_steps=250 #number of steps taken to integrate the wavefunction 
        L=10.0 #lenght of the box 
        x0=-L/2  
        x_list_values=[x0]
        dx=L/n_steps  
        for i in range(1,n_steps+1):
                x_value=x0+dx*i
                x_list_values.append(x_value)

        #we make a list of values x for plotting the potential as a continuous function 
        potential_steps=500
        potential_dx=L/potential_steps
        x_potential_values=[x0]
        for i in range(1,potential_steps+1):
                potential_x=x0+potential_dx*i
                x_potential_values.append(potential_x)
        


        #TARGET
        target_position=[None]*10 #maximum level 10
        target_position[0]=random.random() #we calculate first target_position 
        target_epsilon=0.2 #target width
        #we check that all the target is in the screen 
        while (target_position[0]-target_epsilon)<0 or (target_position[0]+target_epsilon)>1: 
                target_position[0]=random.random() #we generate a new target position 
        
        yellow=Image(source="graphs/blue_target.png",allow_stretch=True,keep_ratio=False)

        #we create the first grid target 
        grid_target=[None]*15 #maximum level 15 
        grid_heart_recover=None
        grid_target[0]=GridLayout(rows=1,cols=1)
        grid_target[0].pos_hint={"center_x":target_position[0],"center_y":0.5}
        grid_target[0].size_hint_x=target_epsilon*2
        grid_target[0].size_hint_y=1
        grid_target[0].add_widget(yellow) 

        first_target=True
        
        position=None

        #HEARTS
        heart=[None]*5
        for i in range(0,5):
                heart[i]=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=True)
        lives_counter=5 #counter of how many lives we have left. Initially we have 5 lives

        #To control no measure is done without a plot 
        is_plot=False #initially equals to False because no plot is showed 

        #to control if there is the possibility to recover a live 
        is_live_recover=False

        #plots left 
        plots_left=3 #Initially we have 3 plots left 

        #score 
        score=0 
        record=0 


        level=1
        start=0
        end=0
        lang= "ENG" #by default

        #controling wheter we are in tutorial mode or in game mode
        tutorial=True 
        tutorial_step=1 

        #MENU FUNCTIONALITY 
        
        #SELECTING LANGUAGE 
        def catalan(self): 
                '''Sets language to catalan''' 
                self.lang="CAT" 
                self.button_cat.background_color=(0,0,1,0.8)
                self.button_esp.background_color=(1,1,1,1)
                self.button_eng.background_color=(1,1,1,1)
                #MENU
                self.front_text.text="Benvingut a EIGENgame.\n Per aprendre a jugar premi TUTORIAL."
                self.game_button.text="JOC"
                #GAME 
                self.center_label.text="CENTRE"
                self.width_label.text="AMPLADA"
                self.right_label.text="DRETA"
                self.left_label.text="ESQUERRA"
                self.score_label.text=" PUNTS = 0"
                self.level_label.text="NIVELL = 1"

                self.tutorial_label.text= "Primer, crea  el teu potencial \n Per començar crea un potencial harmònic "

        def esp(self): 
                '''Sets language to spanish''' 
                self.lang="ESP"
                self.button_cat.background_color=(1,1,1,1)
                self.button_esp.background_color=(0,0,1,0.8)
                self.button_eng.background_color=(1,1,1,1)
                #MENU
                self.front_text.text="Bienvenido a EIGENgame.\n Para aprender a jugar pulse TUTORIAL."
                self.game_button.text="JUEGO"
                #GAME 
                self.center_label.text="CENTRO"
                self.width_label.text="ANCHO"
                self.right_label.text="DERECHA"
                self.left_label.text="IZQUIERDA"
                self.score_label.text="PUNTOS = 0"
                self.level_label.text="NIVEL = 1"

                self.tutorial_label.text= "Primero, crea tu potencial \n Para empezar crea un potencial harmónico "
                
        def english(self): 
                '''Sets language to english''' 
                self.lang="ENG"
                self.button_cat.background_color=(1,1,1,1)
                self.button_esp.background_color=(1,1,1,1)
                self.button_eng.background_color=(0,0,1,0.8)
                #MENU
                self.front_text.text="Welcome to EIGENgame.\n To learn how to play press on TUTORIAL."
                self.game_button.text="GAME"
                #GAME 
                self.center_label.text="CENTER"
                self.width_label.text="WIDTH"
                self.right_label.text="RIGHT"
                self.left_label.text="LEFT"
                self.score_label.text=" SCORE = 0"
                self.level_label.text="LEVEL = 1"
                self.tutorial_label.text= "First, create your potential \n You can start by creating an harmonic potential "


        def game_transition(self): 
                '''Goes to the game'''
                self.game_over_layout.size_hint_x=0 
                self.game_over_layout.size_hint_y=0
                self.game_over_layout.pos_hint={"x":-1,"y":-1}

                self.menu_layout.size_hint_y=0
                self.menu_layout.size_hint_x=0
                self.menu_layout.pos_hint={"x":-1,"y":-1}

                self.bug_layout.size_hint_x=0.15725  
                self.bug_layout.size_hint_y=0.16
                self.bug_layout.pos_hint={"x":0,"y":0}

                self.game_layout.size_hint_x=1 
                self.game_layout.size_hint_y=1 
                self.game_layout.pos_hint={"x":0,"y":0}

                self.tutorial=False


        def tutorial_transition(self): 
                '''Goes to the tutorial'''
                self.game_over_layout.size_hint_x=0 
                self.game_over_layout.size_hint_y=0
                self.game_over_layout.pos_hint={"x":-1,"y":-1}

                self.menu_layout.size_hint_y=0
                self.menu_layout.size_hint_x=0
                self.menu_layout.pos_hint={"x":-1,"y":-1}

                self.bug_layout.size_hint_x=0.15725  
                self.bug_layout.size_hint_y=0.16
                self.bug_layout.pos_hint={"x":0,"y":0}

                self.game_layout.size_hint_x=1 
                self.game_layout.size_hint_y=1 
                self.game_layout.pos_hint={"x":0,"y":0}

                self.tutorial=True

        def menu_transition(self):
                '''Goes to the tutorial'''
                self.game_over_layout.size_hint_x=0 
                self.game_over_layout.size_hint_y=0
                self.game_over_layout.pos_hint={"x":-1,"y":-1}

                self.menu_layout.size_hint_y=1
                self.menu_layout.size_hint_x=1
                self.menu_layout.pos_hint={"x":0,"y":0}

                self.bug_layout.size_hint_x=0  
                self.bug_layout.size_hint_y=0
                self.bug_layout.pos_hint={"x":-1,"y":-1}

                self.game_layout.size_hint_x=0 
                self.game_layout.size_hint_y=0 
                self.game_layout.pos_hint={"x":-1,"y":-1}

                

        def game_over_transition(self):
                '''Goes to game_over'''
                self.game_over_layout.size_hint_x=1
                self.game_over_layout.size_hint_y=1
                self.game_over_layout.pos_hint={"x":0,"y":0}

                self.menu_layout.size_hint_y=0
                self.menu_layout.size_hint_x=0
                self.menu_layout.pos_hint={"x":-1,"y":-1}

                self.bug_layout.size_hint_x=0  
                self.bug_layout.size_hint_y=0
                self.bug_layout.pos_hint={"x":-1,"y":-1}

                self.game_layout.size_hint_x=0 
                self.game_layout.size_hint_y=0 
                self.game_layout.pos_hint={"x":-1,"y":-1} 


        #Choosing energies 
        def value_0(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=0 
                self.button0.background_color=(0,0,1,0.8)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(1,1,1,1)
                self.plot_energy() #changes the color of every button to remark the one pressed

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==3: #tutorial step one 
                                self.tutorial_forth_step()        

        def value_1(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=1 
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(0,0,1,0.8)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(1,1,1,1)
                self.plot_energy()

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==3: #tutorial step one 
                                self.tutorial_forth_step()   
        

        def value_2(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=2
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(0,0,1,0.8)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(1,1,1,1)
                self.plot_energy()

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==3: #tutorial step one 
                                self.tutorial_forth_step()   
               

        def value_3(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=3
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(0,0,1,0.8)
                self.button4.background_color=(1,1,1,1)
                self.plot_energy()

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==3: #tutorial step one 
                                self.tutorial_forth_step()   
              

        def value_4(self): 
                ''' Changes values o value_n when the button is pressed'''
        
                self.value_n=4
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(0,0,1,0.8)
                self.plot_energy()

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==3: #tutorial step one 
                                self.tutorial_forth_step()   
        
        #choosing potentials 

        def harmonic_potential(self): 
                type1_copy=None
                if self.potential_type1=="Harmonic": #there is a harmonic 
                        pass 
                elif self.potential_type2=="Harmonic": #there is a harmonic 
                        pass 
                else: #There is no harmonic
                        if self.potential_type1!="Free particle" and self.potential_type2!="Free particle": #two potential chosen 
                                self.potential_type1="Free particle"
                                self.potential_type2="Free particle" #we reinicialise the potentials to free 

                        if self.potential_type1=="Free particle" or self.potential_type1=="Harmonic": #No other potential chosen 
                                type1_copy="Harmonic" 
                                self.button_harmonic.background_color=(0,0,1,0.8) #only one button pressed 
                                self.button_free.background_color=(1,1,1,1)
                                self.button_barrier.background_color=(1,1,1,1)
                                self.button_water.background_color=(1,1,1,1)
                        else: #ANOTHER POTENTIAL CHOSEN   
                                self.potential_type2="Harmonic"
                                self.button_harmonic.background_color=(0,0,1,0.8) #we press harmonic one too 

                        if type1_copy=="Harmonic": self.potential_type1="Harmonic"


                self.plot_potential()

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==1: #tutorial step one 
                                self.tutorial_second_step()
               

        def slider_value_k(self,*args):
                ''' Takes the hooke constant value from a slider''' 
                self.hooke_constant=args[1]
                self.harmonic_potential()  

        def slider_x0(self,*args):
                ''' Takes the hooke constant value from a slider''' 
                self.x0_harmonic=args[1]
                self.harmonic_potential()
                

        def free_particle(self): 
                self.potential_type1="Free particle"
                self.potential_type2="Free particle"
                self.button_harmonic.background_color=(1,1,1,1)
                self.button_free.background_color=(0,0,1,0.8)
                self.button_barrier.background_color=(1,1,1,1)
                self.button_water.background_color=(1,1,1,1)
                self.plot_potential()
                


        def center_barrier(self,*args): 
                ''' Takes the x position left value of the barrier from a slider''' 
                self.center_b=args[1]
                self.barrier()
                
        def width_barrier(self,*args): 
                ''' Takes the x position right value of the barrier from a slider''' 
                self.width_b=args[1]
                self.barrier()
                
        def V_barrier(self,*args): 
                ''' Takes the V value of the barrier from a slider''' 
                self.Vb=args[1]
                self.barrier()
                
        def barrier(self): 
                b_copy=None
                if self.potential_type1=="Barrier": #there is a barrier 
                        pass 
                elif self.potential_type2=="Barrier": #there is a barrier
                        pass 
                else: #There is no Barrier 
                        if self.potential_type1!="Free particle" and self.potential_type2!="Free particle": #two potential chosen 
                                self.potential_type1="Free particle"
                                self.potential_type2="Free particle" #we reinicialise the potentials to free 
                        if self.potential_type1=="Free particle": #No other potential chosen
                                b_copy="Barrier"
                                self.button_harmonic.background_color=(1,1,1,1)
                                self.button_free.background_color=(1,1,1,1)
                                self.button_barrier.background_color=(0,0,1,0.8)
                                self.button_water.background_color=(1,1,1,1)
                        else: #ANOTHER POTENTIAL CHOSEN   
                                self.potential_type2="Barrier"
                                self.button_barrier.background_color=(0,0,1,0.8) #we press barrier one too 
                        if b_copy=="Barrier": self.potential_type1="Barrier"
                self.plot_potential()

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==2: #tutorial step 2 
                                self.tutorial_third_step()
               


        def left_water(self,*args): 
                ''' Takes the x position left value of the barrier from a slider''' 
                self.left_x_w=args[1]
                self.water_well()

        def right_water(self,*args): 
                ''' Takes the x position right value of the barrier from a slider''' 
                self.right_x_w=args[1]
                self.water_well()

        def V_water(self,*args): 
                ''' Takes the V value of the barrier from a slider''' 
                self.Vw=args[1]
                self.water_well()

        def water_well(self): 
                w_copy=None
                if self.potential_type1=="Well": #there is a well 
                        pass 
                elif self.potential_type2=="Well": #there is a well 
                        pass 
                else: #There is no well 
                        if self.potential_type1!="Free particle" and self.potential_type2!="Free particle": #two potential chosen 
                                self.potential_type1="Free particle"
                                self.potential_type2="Free particle" #we reinicialise the potentials to free 
                        if self.potential_type1=="Free particle": #No other potential chosen
                                w_copy="Well"
                                self.button_harmonic.background_color=(1,1,1,1)
                                self.button_free.background_color=(1,1,1,1)
                                self.button_barrier.background_color=(1,1,1,1)
                                self.button_water.background_color=(0,0,1,0.8)
                        else: #ANOTHER POTENTIAL CHOSEN   
                                self.potential_type2="Well"
                                self.button_water.background_color=(0,0,1,0.8) #we press barrier one too 
                        if w_copy=="Well": self.potential_type1="Well"
                self.plot_potential()
                        


        def potential(self,x): 
                '''Gives the potential'''
                if self.potential_type1=="Harmonic":
                        V1=(1/2)*self.hooke_constant*(x-self.x0_harmonic)**2
                                 
                if self.potential_type1=="Free particle": #if no potential button is pressed we have a free particle
                        V1=0 
                if self.potential_type1=="Barrier":
                        V1=0
                        left_x_b=self.center_b-self.width_b/2 
                        right_x_b=self.center_b+self.width_b/2  
                        if x<left_x_b: V1=0 
                        if x>left_x_b and x <right_x_b: V1=self.Vb  
                        if x>right_x_b: V1=0  
                if self.potential_type1=="Well":
                        V1=0 #if the condition is not satisfied V=0
                        #we check that the left position is smaller than the right one 
                        if self.left_x_w>self.right_x_w:
                                self.right_x_w=self.left_x_w+0.6
                        if self.left_x_w<self.right_x_w: 
                                if x<self.left_x_w: V1=self.Vw 
                                if x>self.left_x_w and x <self.right_x_w: V1=0  
                                if x>self.right_x_w: V1=self.Vw 

                #Now potential 2 

                if self.potential_type2=="Harmonic":
                        V2=(1/2)*self.hooke_constant*(x-self.x0_harmonic)**2
                                 
                if self.potential_type2=="Free particle": #if no potential button is pressed we have a free particle
                        V2=0 
                if self.potential_type2=="Barrier":
                        V2=0
                        left_x_b=self.center_b-self.width_b/2 
                        right_x_b=self.center_b+self.width_b/2  
                        if x<left_x_b: V2=0 
                        if x>left_x_b and x <right_x_b: V2=self.Vb  
                        if x>right_x_b: V2=0  
                if self.potential_type2=="Well":
                        V2=0 #if the condition is not satisfied V=0
                        #we check that the left position is smaller than the right one 
                        if self.left_x_w>self.right_x_w:
                                self.right_x_w=self.left_x_w+0.6
                        if self.left_x_w<self.right_x_w: 
                                if x<self.left_x_w: V2=self.Vw 
                                if x>self.left_x_w and x <self.right_x_w: V2=0  
                                if x>self.right_x_w: V2=self.Vw 


                V=max(V1,V2) #we get the maximum value as the actual potential 

                #for any potential the partcile is inside the box so: 
                if x<-(self.L/2-self.dx/2) : V=10**10 #infinite potential
                if x>(self.L/2-self.dx/2): V=10**10 

                return V 


        #@jit(nopython=False) #NOT  A DECREASE IN RUNTIME  
        def matrix(self): 
                '''This function creates the numpy array we will work with. '''
                #define parameters 
                h2_me=7.6199  # (eV*A) h**2/me

                #we create the matrix with all zeros 
                #H_matrix=np.zeros((self.n_steps+1,self.n_steps+1)) #matrix 501x501 
                diagonal=np.zeros(self.n_steps+1) 
                off_diagonal=np.zeros(self.n_steps)
                for i in range(0,self.n_steps): #rows 
                        diagonal[i]=h2_me/(self.dx**2)+self.potential(self.x_list_values[i])
                        off_diagonal[i]=-h2_me/(2*self.dx**2)
                        
                        #NOT TRIDIAGONAL: (slower) 

                        #if i==0: #first row
                                #ADVANCED TWO POINTS
                                #H_matrix[i,0]= -h2_me/(2*self.dx**2)+self.potential(self.x_list_values[i])
                                #H_matrix[i,1]= h2_me/(self.dx**2)
                                #H_matrix[i,2]= -h2_me/(2*self.dx**2)

                        
                        #elif i==self.n_steps: #last row 
                                #TWO POINTS BACK 
                                #H_matrix[i,self.n_steps]=-h2_me/(2*self.dx**2)+self.potential(self.x_list_values[i])
                                #H_matrix[i,self.n_steps-1]=h2_me/(self.dx**2)
                                #H_matrix[i,self.n_steps-2]=-h2_me/(2*self.dx**2)
                
                        #else: #not first nor last row 
                                #H_matrix[i,i]= h2_me/(self.dx**2)+self.potential(self.x_list_values[i])
                                #H_matrix[i,i-1]=-h2_me/(2*self.dx**2)
                                #H_matrix[i,i+1]=-h2_me/(2*self.dx**2)

                diagonal[self.n_steps]=h2_me/(self.dx**2)+self.potential(self.x_list_values[self.n_steps]) #not covered by loop
                
                return diagonal,off_diagonal 


        #@jit(nopython=False)
        def wave_function(self):
                '''This function finds the eigenvalues and eigenvectors and normalises the eigenvector we want to plot 
                Returns the wave function squared and the energy used'''
                diag,off_diag=self.matrix()
                #eigenval,eigenvec=np.linalg.eig(A) #we compute eigenvalues and eigenvectors 
                #we sort the eigenvalues and eigenvectors 
                #index=np.argsort(eigenval)
                #eigenval=eigenval[index] #list of eigenvalues sorted
                #eigenvec=eigenvec[:,index] #list of eigenvectors sorted accordingly 

                #we compute only the eigenvalue and eigenvector we use: THEORICALLY FASTER
                eigenval,eigenvec=scipy.linalg.eigh_tridiagonal(diag,off_diag,select='i',select_range=(self.value_n,self.value_n))
                E=eigenval[0]
                
                #we choose the energy 
                #E=eigenval[self.value_n] 
                wave=eigenvec[:,0]
                wave=wave.tolist() #we make it a list 
                
                #Now we normalise 
                integral=0 
                for k in range(0,self.n_steps+1): 
                        if k==0 or k==self.n_steps: #extrem values 
                                integral=integral +(wave[k]**2)/3
                        elif (k % 2) == 0:  #even number
                                integral=integral+2*(wave[k]**2)/3
                        else: #odd number 
                                integral=integral+4*(wave[k]**2)/3
                integral=integral*self.dx 
                #now we have to normalize 
                phi_square=[element**2/integral for element in wave] #and we have the wave function normalised and squared


                return phi_square,E         
        
        def plot_energy(self):
                '''This function plots the energy line for a estatic potential'''
                plt.clf()
                
                fig, ax_phi=plt.subplots()
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                ax_phi.set_ylim((0,1)) #maximum of phi_axis
                ax_phi.xaxis.labelpad = -1
                #we compute the eigen_values 
                diag,off_diag=self.matrix()
                #eigenval,eigenvec=np.linalg.eig(A) #we compute eigenvalues and eigenvectors 
                #we sort the eigenvalues and eigenvectors 
                #index=np.argsort(eigenval)
                #eigenval=eigenval[index] #list of eigenvalues sorted
                #eigenvec=eigenvec[:,index] #lis of eigenvectors sorted accordingly 
                #E=eigenval[self.value_n]  #we assign Energy 
                
                #we compute only the eigenvalue we use 
                eigenval=scipy.linalg.eigh_tridiagonal(diag,off_diag,eigvals_only=True,select='i',select_range=(self.value_n,self.value_n))
                E=eigenval[0]
                

                y2=0
                V_plot=[self.potential(s) for s in self.x_potential_values] 
                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_potential_values,V_plot, label="V(x)" , color='tab:blue') 
                ax_V.axhline(y=E, color='g', linestyle='-',label="E") #we plot the Energy value too 
                ax_V.fill_between(self.x_potential_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                if E<40: ax_V.set_ylim((0,41)) #limit in potential axis
                else: ax_V.set_ylim((0,E+5)) 
                ax_V.legend(loc="upper right")

                #MAKING THE TITLE OF THE GRAPH
                #if self.potential_type=="Harmonic":
                        #string_var="    $K(eV/\AA^2)$="+str(self.hooke_constant)
                #if self.potential_type=="Free particle": string_var=""
                #if self.potential_type=="Barrier":
                        #string_var="    $V(eV)$="+str(self.Vb)
                #if self.potential_type=="Well": 
                        #string_var="    $V(eV)$="+str(self.Vw)

                #plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4],loc="right")#"   V(x)= "+str(self.potential_type)+string_var
                        
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()
                self.float_plot.add_widget(canvas_plot)
                self.is_plot=False #no wave function plotted
                self.measure_button.disabled=True

        def plot_potential(self): 
                '''This function plots the potential
                This function won't plot energy in order to not having to compute the eigenvalues
                every time the potential is changing'''
                #clear previous plot 
                plt.clf()

                fig, ax_phi=plt.subplots()
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                ax_phi.set_ylim((0,1)) #maximum of phi_axis
                ax_phi.xaxis.labelpad = -1

                y2=0
                V_plot=[self.potential(s) for s in self.x_potential_values] 
                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_potential_values,V_plot, label="V(x)" , color='tab:blue') 
                ax_V.fill_between(self.x_potential_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                ax_V.set_ylim((0,41)) 
                ax_V.legend(loc="upper right")

                #MAKING THE TITLE OF THE GRAPH
                #if self.potential_type=="Harmonic":
                        #string_var="    $K(eV/\AA^2)$="+str(self.hooke_constant)[0:4]
                #if self.potential_type=="Free particle": string_var=""
                #if self.potential_type=="Barrier":
                        #string_var="    $V(eV)$="+str(self.Vb)
                #if self.potential_type=="Well": 
                        #string_var="    $V(eV)$="+str(self.Vw)

         
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()
                self.float_plot.add_widget(canvas_plot)
                self.is_plot=False #no wave function plotted
                self.measure_button.disabled=True 

        def plot_wave_function(self,*args):
                '''This function plots into the window the wave function.
                It also plots the potential. Previous to plotting it computes the eigenvalues
                and computes the eigenvector. '''
                plt.clf()
                self.plots_left-=1  
                self.plots_label.text=str(self.plots_left) #we have to update the plots left
                #we animate the plotsleft 
                label_animation1=Animation(font_size=25,duration=0.5)
                label_animation1+=Animation(font_size=20,duration=0.5)
                label_animation1.start(self.plots_label)
        
                phi_square, E=self.wave_function()#we compute the eigen_values  and assign E 
                 
                y2=0
                #let's plot:
                V_plot=[self.potential(k) for k in self.x_potential_values] #compute potential for every x_position
                fig, ax_phi=plt.subplots() 
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                ax_phi.xaxis.labelpad = -1
                        
                ax_phi.set_ylim((0,max(phi_square)+0.2)) #maximum of phi_axis= maxim of probability +0.2
                        

                #we plot the potential
                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_potential_values,V_plot, label="V(x)" , color='tab:blue')
                ax_V.fill_between(self.x_potential_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                ax_phi.plot(self.x_list_values,phi_square,label=r"$ \phi^2(x)$" , color='purple')
                ax_phi.fill_between(self.x_list_values, phi_square,0, facecolor='purple', alpha=0.7) #paint wave_function
                ax_V.axhline(y=E, color='g', linestyle='-',label="E") #we plot the Energy value too 
                        
                if E<35: ax_V.set_ylim((0,42))
                else: ax_V.set_ylim((0,E+5)) 
                ax_V.legend(loc="upper right")
                ax_phi.legend(loc="upper left")

                #plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4],loc="right")#"   V(x)= "+str(self.potential_type)+string_var
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()
                self.float_plot.add_widget(canvas_plot)
                self.is_plot=True #there is a wave function plotted
                #now we can measure a particle 
                self.measure_button.disabled=False 
                #we are out of plots 
                if self.plots_left==0: 
                        self.disable_plotting()
                        self.button_measure_anim()

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==4: #tutorial step one 
                                self.tutorial_fifth_step()

                             
        def disable_plotting(self): 
                '''Disables all buttons related to plotting so we can't plot nothing'''
                #disabling ploting
                self.plot_button.disabled=True 
                #disbaling energies
                self.button0.disabled=True 
                self.button1.disabled=True
                self.button2.disabled=True
                self.button3.disabled=True
                self.button4.disabled=True
                #disabling potential buttons 
                self.button_free.disabled=True 
                self.button_harmonic.disabled=True 
                self.button_barrier.disabled=True 
                self.button_water.disabled=True 
                self.rubber_button.disabled=True 
                #disabling sliders
                self.hooke_slider.disabled=True
                self.x0_slider.disabled=True
                self.center_slider.disabled=True
                self.width_slider.disabled=True
                self.vb_slider.disabled=True
                self.left_slider.disabled=True
                self.right_slider.disabled=True
                self.vw_slider.disabled=True


        def able_plotting(self):
                '''Ables all buttons related to plotting so we can't plot nothing'''
                #abling ploting
                self.plot_button.disabled=False 
                #abling energies
                self.button0.disabled=False 
                self.button1.disabled=False 
                self.button2.disabled=False 
                self.button3.disabled=False 
                self.button4.disabled=False 
                #abling potential buttons 
                self.button_free.disabled=False  
                self.button_harmonic.disabled=False  
                self.button_barrier.disabled=False  
                self.button_water.disabled=False  
                self.rubber_button.disabled=False  
                #abling sliders
                self.hooke_slider.disabled=False 
                self.x0_slider.disabled=False 
                self.center_slider.disabled=False 
                self.width_slider.disabled=False 
                self.vb_slider.disabled=False 
                self.left_slider.disabled=False 
                self.right_slider.disabled=False 
                self.vw_slider.disabled=False 

        def button_measure_anim(self): 
                '''Animates the measure button when there's no plots left'''
                button_animation=Animation(size_hint_x=0.085,size_hint_y=0.095,duration=0.5)
                button_animation+=Animation(size_hint_x=0.075,size_hint_y=0.08,duration=0.5)
                button_animation+=Animation(size_hint_x=0.085,size_hint_y=0.095,duration=0.5)
                button_animation+=Animation(size_hint_x=0.075,size_hint_y=0.08,duration=0.5)
                button_animation.start(self.measure_button)

                #we animate the plotsleft 
                label_animation=Animation(font_size=30,duration=0.5)
                label_animation+=Animation(font_size=20,duration=0.5)
                label_animation+=Animation(font_size=30,duration=0.5)
                label_animation+=Animation(font_size=20,duration=0.5)
                label_animation.start(self.plots_label)

        def loading_anim(self): 
                '''Triggers loading animation'''
                loading_animation=Animation(color=(0,0,0,1),duration=0.000005)
                loading_animation+=Animation(color=(0,0,0,1),duration=0.000005)
                loading_animation+=Animation(color=(0,0,0,0),duration=0.00005)
                loading_animation.start(self.loading_label) 
                loading_animation.bind(on_complete=self.plot_wave_function)

        def appear_e(self): 
                '''Manages the live recovery, when the process of live recovery is finished we call teh functions that checks 
                the targets'''
                self.able_plotting()
                self.plots_left=3  
                self.plots_label.text=str(self.plots_left)

                probabilities,E=self.wave_function() #compute the probabilities and energy 
                n_e=self.level*2-1 
                if self.level==1: n_e=1  #number of electrons we want to plot (level number)
                self.position=random.choices(self.x_list_values,weights=probabilities,k=n_e) #computes the position accordingly to WF
                #we create a lists of the gridslayouts with are working with 
                e_grid=[None]*n_e #list of n_e elements
                n_greens=0 #number of targets in green
                targets_achieved=[]
                for i in range(0,n_e): #we add the electrons to the measure layout 
                        e_graphs=Image(source="graphs/electron.png",allow_stretch=True,keep_ratio=False) #we import image
                        self.e_position[i]=(self.position[i]+5)/10
                        e_grid[i]=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                        e_grid[i].pos_hint={"center_x":self.e_position[i],"center_y":0.5} #we put the gridlayout in the position we want
                        e_grid[i].size_hint_x=0.06
                        e_grid[i].size_hint_y=0.25 #size of gridlayout
                        e_grid[i].add_widget(e_graphs) #add the image to the gridLayout
                        self.measure_layout.add_widget(e_grid[i]) #we add the electron grid layout to the float layout
                
                #checking whether we have touched the live 
                if self.is_live_recover==True: 
                        heart_position=self.target_position[0]
                        is_heart_achived=False 
                        for i in range(0,n_e): 
                                self.e_position[i]=(self.position[i]+5)/10
                                if self.e_position[i]>(heart_position-self.target_epsilon) \
                                and self.e_position[i]<(heart_position+self.target_epsilon):  
                                        #green target: live recovered  
                                        self.live_recovered()
                                        is_heart_achived=True 
                                        break
                        if is_heart_achived==False: self.electron_check() #no live recovered 


                else: self.electron_check() #no live recovered 

        def electron_check(self,*args): 
                '''Plots the electron after pressing the measure button'''
                self.plots_left=3  
                self.plots_label.text=str(self.plots_left)
                n_e=self.level*2-1 
                if self.level==1: n_e=1  #number of electrons we want to plot (level number) 
                e_grid=[None]*n_e #list of n_e elements
                n_greens=0 #number of targets in green
                targets_achieved=[]


                for i in range(0,n_e): #we add the electrons to the measure layout 

                        for j in range(0,self.level): #we check the targets  
                                if self.e_position[i]>(self.target_position[j]-self.target_epsilon) \
                                and self.e_position[i]<(self.target_position[j]+self.target_epsilon):  #green target 
                                        if j not in targets_achieved: 
                                                targets_achieved.append(j) #add green target
                                                n_greens+=1 
                                                #we have a list of the green targets achieved   
                
                if n_greens<self.level: #not all targets in green

                        for j in range(0,self.level): 
                                if self.first_target==False: #not the first target: 
                                        self.grid_target[j].clear_widgets() #erase previous target
                                        if j in targets_achieved: #we are on a green target 
                                                first_green=Image(source="graphs/green_target.png",allow_stretch=True,keep_ratio=False)
                                                self.grid_target[j].add_widget(first_green)#we turn into green the rectangle
                                                target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                                target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                                target_anim.start(self.grid_target[j])  
                                        else: #j no in targets_achieved   
                                                new_red= Image(source="graphs/red_target.png",allow_stretch=True,keep_ratio=False)
                                                self.grid_target[j].add_widget(new_red)#we turn into green the rectangle
                                                target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                                target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                                target_anim.start(self.grid_target[j])  
                                        if j==(self.level-1): #the last one 
                                                target_anim.bind(on_complete=self.same_target)
                                else: #the first target 
                                        self.first_grid_target.clear_widgets()
                                        first_red=Image(source="graphs/red_target.png",allow_stretch=True,keep_ratio=False)
                                        self.first_grid_target.add_widget(first_red)#we turn into red the rectangle
                                        target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                        target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                        target_anim.start(self.first_grid_target)  
                                        target_anim.bind(on_complete=self.same_target)


                else: #all greens 
                        for j in range(0,self.level): 
                                if self.first_target==False: #not the first target: 
                                        self.grid_target[j].clear_widgets() #erase previous target
                                        new_green=Image(source="graphs/green_target.png",allow_stretch=True,keep_ratio=False)
                                        self.grid_target[j].add_widget(new_green)#we turn into green the rectangle
                                        target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                        target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                        target_anim.start(self.grid_target[j]) 
                                        if j==(self.level-1): #the last one 
                                                target_anim.bind(on_complete=self.new_target)
                                                #when the animation is done we generate a new rectangle  
                                else: #the first target 
                                        self.first_grid_target.clear_widgets()
                                        first_green=Image(source="graphs/green_target.png",allow_stretch=True,keep_ratio=False)
                                        self.first_grid_target.add_widget(first_green)#we turn into green the rectangle
                                        target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                        target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                        target_anim.start(self.first_grid_target)  
                                        target_anim.bind(on_complete=self.new_target)


        def new_target(self,*args): #GREEN
                '''Generates a new target in another position after having acomplished a green'''
                self.measure_layout.clear_widgets()
                
                if self.first_target==True:  self.first_grid_target.clear_widgets()
                else:
                        for j in range(0,self.level):
                                self.grid_target[j].clear_widgets() #erase previous target
                
                self.first_target=False 

                if self.level==1: #we check if it needs to be smaller in level 1 
                        #we generate a new epsilon 
                        if self.target_epsilon>0.08:  #if it's bigger than 0.08
                                self.target_epsilon=self.target_epsilon-0.05
                        else: #it's smaller 
                                self.level=2 #plus one in level
                                #once we know for sure the new level 
                                level_animation1=Animation(color=(233/255, 179/255, 7/255, 1),duration=0.5)
                                level_animation1+=Animation(color=(0,0,0,1),duration=0.5)
                                level_animation1.start(self.level_label) 

                                if self.lang=="ENG": self.level_label.text="LEVEL = "+str(self.level)
                                elif self.lang=="ESP": self.level_label.text="NIVEL = "+str(self.level)
                                else: self.level_label.text="NIVELL = "+str(self.level) 
                                self.target_epsilon=0.1 #resize, note that now is smaller than at the beggining of level 1 
                else: #we increase level inmidiatelly 
                        self.level+=1 
                        level_animation1=Animation(color=(233/255, 179/255, 7/255, 1),duration=0.5)
                        level_animation1+=Animation(color=(0,0,0,1),duration=0.5)
                        level_animation1.start(self.level_label) 
                        if self.lang=="ENG": self.level_label.text="LEVEL = "+str(self.level)
                        elif self.lang=="ESP": self.level_label.text="NIVEL = "+str(self.level)
                        else: self.level_label.text="NIVELL = "+str(self.level) 
                        if self.level<=3: 
                                self.target_epsilon=0.1
                        elif self.level==4: 
                                self.target_epsilon=0.075
                        elif self.level<=7: 
                                self.target_epsilon=0.05
                        else: self.target_epsilon=0.025

                overlaping_zone=[]
                j=0 
                while j<self.level: #new targets + avoiding overlapping 
                        actual_j=j
                        self.target_position[j]=random.random() #new position     
                        actual_max=self.target_position[j]+self.target_epsilon
                        actual_min=self.target_position[j]-self.target_epsilon
                        if j!=0: 
                                for p in range(0,j): 
                                        previous_min=overlaping_zone[p][0]
                                        previous_max=overlaping_zone[p][1]
                                        if (actual_min<=previous_max and actual_min>=previous_min) or \
                                        (actual_max<=previous_max and actual_max>=previous_min) or (actual_min)<0 or (actual_max)>1:
                                        #the target position is bad 
                                                j=actual_j 
                                                break #we go back to another self.targetposition with same j 
                                        else: #the target position is good 
                                                if p==(j-1): #we have arrived to the end of the overlaping check
                                                        overlaping_zone.append([actual_min,actual_max]) 
                                                        self.grid_target[j]=GridLayout(rows=1,cols=1) #new target
                                                        self.grid_target[j].pos_hint={"center_x":self.target_position[j],"center_y":0.5}
                                                        self.grid_target[j].size_hint_x=self.target_epsilon*2
                                                        self.grid_target[j].size_hint_y=1
                                                        yellow_image=Image(source="graphs/blue_target.png",allow_stretch=True,keep_ratio=False)
                                                        self.grid_target[j].add_widget(yellow_image) 
                                                        self.measure_layout.add_widget(self.grid_target[j])
                                                        j=j+1 #going to the next iteration         
                                                #else we continue iterating along p

                                                
                        if j==0: #only one target 
                                while (self.target_position[j]-self.target_epsilon)<0 or (self.target_position[j]+self.target_epsilon)>1: 
                                        self.target_position[j]=random.random() #we generate a new target position

                                actual_max=self.target_position[j]+self.target_epsilon
                                actual_min=self.target_position[j]-self.target_epsilon
                                overlaping_zone.append([actual_min,actual_max])

                                self.grid_target[j]=GridLayout(rows=1,cols=1) #new target
                                self.grid_target[j].pos_hint={"center_x":self.target_position[j],"center_y":0.5}
                                self.grid_target[j].size_hint_x=self.target_epsilon*2
                                self.grid_target[j].size_hint_y=1
                                yellow_image=Image(source="graphs/blue_target.png",allow_stretch=True,keep_ratio=False)
                                self.grid_target[j].add_widget(yellow_image) 
                                self.measure_layout.add_widget(self.grid_target[j])
                                j=j+1 #going to the next iteration

                if self.lives_counter<5: #NOT ALL LIVES : it is possible to recover a live  
                        if random.randint(1,3)==2 and self.level==1: #one in trhee times  
                                self.is_live_recover=True 
                                heart_position=self.target_position[0] #live appears in first target generated 
                                self.grid_heart_recover=GridLayout(rows=1,cols=1)
                                self.grid_heart_recover.pos_hint={"center_x":heart_position,"center_y":0.5}
                                self.grid_heart_recover.size_hint_x=0.035
                                self.grid_heart_recover.size_hint_y=0.2
                                heart_image=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=False)
                                self.grid_heart_recover.add_widget(heart_image)
                                self.measure_layout.add_widget(self.grid_heart_recover)
                        elif random.randint(1,2)==2 and self.level>1:  #one in two times
                                self.is_live_recover=True 
                                heart_position=self.target_position[0] #live appears in first target generated 
                                self.grid_heart_recover=GridLayout(rows=1,cols=1)
                                self.grid_heart_recover.pos_hint={"center_x":heart_position,"center_y":0.5}
                                self.grid_heart_recover.size_hint_x=0.035
                                self.grid_heart_recover.size_hint_y=0.2
                                heart_image=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=False)
                                self.grid_heart_recover.add_widget(heart_image)
                                self.measure_layout.add_widget(self.grid_heart_recover)
                        else: 
                                self.is_live_recover=False

                else: self.is_live_recover=False

                                
            

                #UPDATING THE SCORE + ELECTRON ANIMATION TO THE SCORE
                if self.level==1: 
                        n_e=1 #we had one electron 
                elif self.level==2:  
                        n_e=1 #we had the number of the previous level 
                else: 
                        n_e=(self.level-1)*2-1

                e_grid=[None]*(n_e) #list of n_e elements
                for i in range(0,n_e):
                        e_graphs=Image(source="graphs/electron.png",allow_stretch=True,keep_ratio=False) #we import image
                        e_grid[i]=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                        e_grid[i].pos_hint={"center_x":self.e_position[i],"center_y":0.5} #we put the gridlayout in the position we want
                        e_grid[i].size_hint_x=0.06
                        e_grid[i].size_hint_y=0.25 #size of gridlayout
                        e_grid[i].add_widget(e_graphs) #add the image to the gridLayout
                        self.measure_layout.add_widget(e_grid[i])
                        e_animation=Animation(pos_hint={"center_x":self.e_position[i],"center_y":0.5,},duration=0.005)
                        e_animation+=Animation(pos_hint={"center_x":-0.15,"center_y":4.35,},duration=1.5) #GOES TO THE SCORE
                        e_animation+=Animation(size_hint_x=0,size_hint_y=0,duration=0.0005) #THEN DISAPPEARS
                        e_animation.start(e_grid[i])
                        e_animation.bind(on_complete=self.score_update)

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==5: #tutorial step one 
                                self.tutorial_six_step_green()
                        elif self.tutorial_step==6: #tutorial step one 
                                if self.lang=="ENG": self.tutorial_label.text="TUTORIAL COMPLETED!" 
                                elif self.lang=="ESP": self.tutorial_label.text='¡TUTORIAL COMPLETADO!' 
                                else: self.tutorial_label.text="TUTORIAL COMPLETAT"
                                tutorial_animation=Animation(font_size=45,duration=0.2)
                                tutorial_animation+=Animation(font_size=35,duration=0.2)
                                tutorial_animation+=Animation(font_size=45,duration=0.2)
                                tutorial_animation+=Animation(font_size=35,duration=0.2)
                                tutorial_animation.start(self.tutorial_label)
                                tutorial_animation.bind(on_complete=self.tutorial_final_step)

       
        def same_target(self,*args): #RED
                '''Generates same target in the same position after having missed'''
                self.measure_layout.clear_widgets()
                if self.first_target==True: #the same target 
                        self.first_grid_target.clear_widgets()
                        self.grid_target[0]=GridLayout(rows=1,cols=1) #new target
                        self.grid_target[0].pos_hint={"center_x":self.target_position[0],"center_y":0.5}
                        self.grid_target[0].size_hint_x=self.target_epsilon*2
                        self.grid_target[0].size_hint_y=1
                        first_yellow=Image(source="graphs/blue_target.png",allow_stretch=True,keep_ratio=False)
                        self.grid_target[0].add_widget(first_yellow) 
                        self.measure_layout.add_widget(self.grid_target[0])
                else: 
                        for j in range(0,self.level):
                                self.grid_target[j].clear_widgets() #erase previous target 
                                yellow_image=Image(source="graphs/blue_target.png",allow_stretch=True,keep_ratio=False)
                                self.grid_target[j].add_widget(yellow_image)#we turn into yellow
                                self.grid_target[j].size_hint_x=self.target_epsilon*2 #resize
                                self.grid_target[j].size_hint_y=1 
                                self.measure_layout.add_widget(self.grid_target[j])

                self.first_target=False

                #we have lost a live: we erase one heart 
                heart_loss_anim=Animation(size_hint_y=1,size_hint_x=0.05) 
                heart_loss_anim+=Animation(size_hint_y=1.1,size_hint_x=0.07,duration=0.2) #we make it bigger
                heart_loss_anim+=Animation(size_hint_y=1,size_hint_x=0.05,duration=0.1)
                
                
                if self.lives_counter==5: #we have 5 lives
                        heart_loss_anim.start(self.heart5_grid)
                        heart_loss_anim.bind(on_complete=self.heart5_disappear)
                if self.lives_counter==4: #we have 4 lives
                        heart_loss_anim.start(self.heart4_grid)
                        heart_loss_anim.bind(on_complete=self.heart4_disappear)
                if self.lives_counter==3: #we have 3 lives
                        heart_loss_anim.start(self.heart3_grid)
                        heart_loss_anim.bind(on_complete=self.heart3_disappear)
                if self.lives_counter==2: #we have 2 lives
                        heart_loss_anim.start(self.heart2_grid)
                        heart_loss_anim.bind(on_complete=self.heart2_disappear)
                if self.lives_counter==1: #we have 1 lives
                        pass 
                        

                #after erasing the heart we update the counter
                self.lives_counter-=1  
                if self.lives_counter==0: #if we are out of lives we show the Game Over image
                         self.game_over_transition()
                         if self.lang=="ENG": self.final_score_label.text="FINAL SCORE = "+ str(self.score)
                         elif self.lang=="ESP": self.final_score_label.text="PUNTUACIÓN FINAL = "+ str(self.score)
                         else: self.final_score_label.text="PUNTUACIÓ FINAL = "+ str(self.score)
                         self.final_score_label.color=(233/255, 179/255, 7/255, 1)
                         self.is_plot=False
                         #we check if there's is a new record: 
                         new_record=False
                         if self.score>self.record: #we have outreached the previous record
                                self.record=self.score 

                                if self.lang=="ENG": self.record_label_over.text=" NEW RECORD = "+ str(self.record)
                                elif self.lang=="ESP": self.record_label_over.text=" NUEVO RÉCORD = "+ str(self.record)
                                else: self.record_label_over.text=" NOU RÈCORD = "+ str(self.record) 

                                self.record_label_over.color=(233/255, 179/255, 7/255, 1)
                                label_animation=Animation(font_size=40,duration=0.5)
                                label_animation+=Animation(font_size=35,duration=0.5)
                                label_animation.start(self.record_label_over)
                         else: #not a new record 
                                if self.lang=="ENG": self.record_label_over.text=" RECORD = "+ str(self.record)
                                elif self.lang=="ESP": self.record_label_over.text=" RÉCORD = "+ str(self.record)
                                else: self.record_label_over.text=" RÈCORD = "+ str(self.record)   
                                self.record_label_over.color=(233/255, 179/255, 7/255, 1)

                elif self.lives_counter<5: #NOT ALL LIVES : it is possible to recover a live 
                        if random.randint(1,3)==2 and self.level==1: #one in three times  
                                self.is_live_recover=True 
                                heart_position=self.target_position[0] #live appears in first target generated 
                                self.grid_heart_recover=GridLayout(rows=1,cols=1)
                                self.grid_heart_recover.pos_hint={"center_x":heart_position,"center_y":0.5}
                                self.grid_heart_recover.size_hint_x=0.035
                                self.grid_heart_recover.size_hint_y=0.2
                                heart_image=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=False)
                                self.grid_heart_recover.add_widget(heart_image)
                                self.measure_layout.add_widget(self.grid_heart_recover)
                        elif random.randint(1,2)==2 and self.level>1:  #one in two times
                                self.is_live_recover=True 
                                heart_position=self.target_position[0] #live appears in first target generated 
                                self.grid_heart_recover=GridLayout(rows=1,cols=1)
                                self.grid_heart_recover.pos_hint={"center_x":heart_position,"center_y":0.5}
                                self.grid_heart_recover.size_hint_x=0.035
                                self.grid_heart_recover.size_hint_y=0.2
                                heart_image=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=False)
                                self.grid_heart_recover.add_widget(heart_image)
                                self.measure_layout.add_widget(self.grid_heart_recover)
                        else: 
                                self.is_live_recover=False

                else: self.is_live_recover=False 

                if self.tutorial==True: #we are on tutorial 
                        if self.tutorial_step==5 or self.tutorial_step==6 : #tutorial step one 
                                self.tutorial_six_step_red()



        def live_recovered(self): #we have recovered a live
                '''Updates the live counter and live icons with animation'''
                heart_position=self.target_position[0]
                self.grid_heart_recover.clear_widgets()#delete previous heart 
                self.grid_heart_recover=GridLayout(rows=1,cols=1)
                self.grid_heart_recover.pos_hint={"center_x":heart_position,"center_y":0.5}
                self.grid_heart_recover.size_hint_x=0.035
                self.grid_heart_recover.size_hint_y=0.2
                heart_image=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=False)
                self.grid_heart_recover.add_widget(heart_image)
                self.measure_layout.add_widget(self.grid_heart_recover)
                heart__recover_animation=Animation(pos_hint={"center_x":heart_position,"center_y":0.5,},duration=0.005)
                heart__recover_animation+=Animation(pos_hint={"center_x":-0.15,"center_y":4.5,},duration=1.5) #GOES TO THE LIVES 
                heart__recover_animation+=Animation(size_hint_x=0,size_hint_y=0,duration=0.0005) #THEN DISAPPEARS
                heart__recover_animation.start(self.grid_heart_recover)
                heart__recover_animation.bind(on_complete=self.lives_counter_add1)


        def heart5_disappear(self,*args): 
                self.heart5_grid.clear_widgets()
        def heart4_disappear(self,*args): 
                self.heart4_grid.clear_widgets()
        def heart3_disappear(self,*args): 
                self.heart3_grid.clear_widgets()
        def heart2_disappear(self,*args): 
                self.heart2_grid.clear_widgets()

        def heart1_disappear(self,*args): 
                self.heart1_grid.clear_widgets()

        def lives_counter_add1(self,*args):
                self.lives_counter+=1 
                heart_image=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=True)   
                if self.lives_counter==5:
                        self.heart5_grid.add_widget(heart_image)
                if self.lives_counter==4:
                        self.heart4_grid.add_widget(heart_image)
                if self.lives_counter==3:
                        self.heart3_grid.add_widget(heart_image)
                if self.lives_counter==2:
                        self.heart2_grid.add_widget(heart_image)
                if self.lives_counter==1:
                        self.heart1_grid.add_widget(heart_image)
                self.electron_check() #calls electron check 


        def score_update(self,*args): #UPTADES THE SCORE
                '''Updates the score, called by new target''' 
                self.score+=1 #we add one point 
                if self.lang=="ENG": self.score_label.text=" SCORE = " +str(self.score)
                elif self.lang=="ESP": self.score_label.text="PUNTOS = " +str(self.score)
                else: self.score_label.text=" PUNTS = " +str(self.score) 
                label_animation1=Animation(color=(233/255, 179/255, 7/255, 1),duration=0.5)
                label_animation1+=Animation(color=(0,0,0,1),duration=0.5)
                #e_animation.bind(on_complete=
                label_animation1.start(self.score_label)
                if self.score>=self.record: 
                        self.record_label.text= " MAX = " +str(self.score)
                        label_animation1=Animation(color=(233/255, 179/255, 7/255, 1),duration=0.5)
                        label_animation1+=Animation(color=(0,0,0,1),duration=0.5)
                        label_animation1.start(self.record_label)

        def reboot(self): 
                '''This function reboots the trainer so when you enter the game it's all new and fresh. 
                It's activated when you press the button '''

                #we delete the gameover layout again 
                self.game_over_layout.size_hint_x=0 
                self.game_over_layout.size_hint_y=0 

                #we reactivate all lives 
                self.lives_counter=5 
                heart_image=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=True) 
                self.heart1_grid.clear_widgets() #in case the reboot option was activated by pressing menu 
                self.heart1_grid.add_widget(self.heart[0])
                self.heart2_grid.clear_widgets()
                self.heart2_grid.add_widget(self.heart[1])
                self.heart3_grid.clear_widgets()
                self.heart3_grid.add_widget(self.heart[2])
                self.heart4_grid.clear_widgets()
                self.heart4_grid.add_widget(self.heart[3])
                self.heart5_grid.clear_widgets()
                self.heart5_grid.add_widget(self.heart[4])

                #we reactivate plots 
                self.plots_left=3  
                self.able_plotting()
                self.plots_label.text=str(self.plots_left)

                self.free_particle() #deletes other potential drawn in screen 

                #score 
                self.score=0 #we put the score to zero 
                if self.lang=="ENG": self.score_label.text=" SCORE = " +str(self.score)
                elif self.lang=="ESP": self.score_label.text="PUNTOS = " +str(self.score)
                else: self.score_label.text=" PUNTS = " +str(self.score)


                for j in range(0,self.level): 
                        self.grid_target[j].clear_widgets() #erase previous targets

                if self.first_target==True:  self.first_grid_target.clear_widgets()
                self.measure_layout.clear_widgets()
                self.first_target=False 
                self.is_live_recover=False
                self.level=1
                if self.lang=="ENG": self.level_label.text="LEVEL = "+str(self.level)
                elif self.lang=="ESP": self.level_label.text="NIVEL = "+str(self.level)
                else: self.level_label.text="NIVELL = "+str(self.level) 

               
                 
                self.target_position[0]=random.random() #new position 
                self.target_epsilon=0.2
                while (self.target_position[0]-self.target_epsilon)<0 or (self.target_position[0]+self.target_epsilon)>1: 
                        self.target_position[0]=random.random() #we generate a new target position 
                self.grid_target[0]=GridLayout(rows=1,cols=1) #new target
                self.grid_target[0].pos_hint={"center_x":self.target_position[j],"center_y":0.5}
                self.grid_target[0].size_hint_x=self.target_epsilon*2
                self.grid_target[0].size_hint_y=1
                yellow_image=Image(source="graphs/blue_target.png",allow_stretch=True,keep_ratio=False)
                self.grid_target[0].add_widget(yellow_image) 
                self.measure_layout.add_widget(self.grid_target[0])
        
                

                #we clear the plot 
                self.float_plot.clear_widgets()
                self.is_plot=False #no wave function plotted
                self.measure_button.disabled=True 

                #we overpaint again the bug layout 
                self.bug_layout.size_hint_x= 0.15725
                self.bug_layout.size_hint_y=0.16

        #TUTORIAL 

        def tutorial_first_step(self): 
                '''First step consisting in choosing a potential''' 
                self.tutorial_layout.size_hint_y=1
                self.tutorial_layout.size_hint_x=1
                self.tutorial_layout.pos_hint={'x': 0, 'y':0}

                self.tutorial_step=1 

                #disabling ploting
                self.plot_button.disabled=True 
                #disbaling energies
                self.button0.disabled=True 
                self.button1.disabled=True
                self.button2.disabled=True
                self.button3.disabled=True
                self.button4.disabled=True
                #disabling potential buttons 
                self.button_free.disabled=True
                self.button_harmonic.disabled=False 
                self.button_barrier.disabled=True 
                self.button_water.disabled=True
                self.rubber_button.disabled=False
                #disabling sliders
                self.hooke_slider.disabled=False
                self.x0_slider.disabled=False
                self.center_slider.disabled=True
                self.width_slider.disabled=True
                self.vb_slider.disabled=True
                self.left_slider.disabled=True
                self.right_slider.disabled=True 
                self.vw_slider.disabled=True 

        def tutorial_second_step(self): 
                '''Second step consisting in choosing a potential''' 
                self.tutorial_layout.size_hint_y=1
                self.tutorial_layout.size_hint_x=1
                self.tutorial_layout.pos_hint={'x': 0, 'y':0}
                if self.lang=="ENG": self.tutorial_label.text='Now, try adding a barrier to your potential' 
                elif self.lang=="ESP": self.tutorial_label.text='Ahora, añadele una barrera a tu potencial' 
                else: self.tutorial_label.text='Ara, afegeix-li una barrera al teu potencial'

                self.tutorial_step=2

                #disabling ploting
                self.plot_button.disabled=True 
                #disbaling energies
                self.button0.disabled=True 
                self.button1.disabled=True
                self.button2.disabled=True
                self.button3.disabled=True
                self.button4.disabled=True
                #disabling potential buttons 
                self.button_free.disabled=True
                self.button_harmonic.disabled=False 
                self.button_barrier.disabled=False 
                self.button_water.disabled=True
                self.rubber_button.disabled=False
                #disabling sliders
                self.hooke_slider.disabled=False
                self.x0_slider.disabled=False
                self.center_slider.disabled=False
                self.width_slider.disabled=False
                self.vb_slider.disabled=False
                self.left_slider.disabled=True
                self.right_slider.disabled=True 
                self.vw_slider.disabled=True 

        def tutorial_third_step(self): 
                '''Third step consisting in choosing a energy ''' 
                self.tutorial_layout.size_hint_y=1
                self.tutorial_layout.size_hint_x=1
                self.tutorial_layout.pos_hint={'x': 0, 'y':0}
                if self.lang=="ENG": self.tutorial_label.text="Once we have the potential, \n let's choose an energy level" 
                elif self.lang=="ESP": self.tutorial_label.text='Ya tenemos el potencial, \n toca escoger un nivel energético' 
                else: self.tutorial_label.text='Ja tenim el potencial, \n toca escollir un nivell energètic'

                self.tutorial_step=3

                #disabling ploting
                self.plot_button.disabled=True 
                #disbaling energies
                self.button0.disabled=False 
                self.button1.disabled=False
                self.button2.disabled=False
                self.button3.disabled=False
                self.button4.disabled=False
                #disabling potential buttons 
                self.button_free.disabled=True
                self.button_harmonic.disabled=True 
                self.button_barrier.disabled=True 
                self.button_water.disabled=True
                self.rubber_button.disabled=True
                #disabling sliders
                self.hooke_slider.disabled=True
                self.x0_slider.disabled=True
                self.center_slider.disabled=True
                self.width_slider.disabled=True
                self.vb_slider.disabled=True
                self.left_slider.disabled=True
                self.right_slider.disabled=True 
                self.vw_slider.disabled=True 
                
        def tutorial_forth_step(self): 
                '''Forth step consisting in ploting the wave function ''' 
                self.tutorial_layout.size_hint_y=1
                self.tutorial_layout.size_hint_x=1
                self.tutorial_layout.pos_hint={'x': 0, 'y':0}
                if self.lang=="ENG": self.tutorial_label.text="Let's plot the wave function" 
                elif self.lang=="ESP": self.tutorial_label.text='Ahora a dibujar la función de onda' 
                else: self.tutorial_label.text="És el moment de dibuixar la funció d'ona"

                self.tutorial_step=4

                #disabling ploting
                self.plot_button.disabled=False
                #disbaling energies
                self.button0.disabled=True 
                self.button1.disabled=True
                self.button2.disabled=True
                self.button3.disabled=True
                self.button4.disabled=True
                #disabling potential buttons 
                self.button_free.disabled=True
                self.button_harmonic.disabled=True 
                self.button_barrier.disabled=True 
                self.button_water.disabled=True
                self.rubber_button.disabled=True
                #disabling sliders
                self.hooke_slider.disabled=True
                self.x0_slider.disabled=True
                self.center_slider.disabled=True
                self.width_slider.disabled=True
                self.vb_slider.disabled=True
                self.left_slider.disabled=True
                self.right_slider.disabled=True 
                self.vw_slider.disabled=True 

        def tutorial_fifth_step(self): 
                '''Fifth step consisting in measuring the electron ''' 
                self.tutorial_layout.size_hint_y=1
                self.tutorial_layout.size_hint_x=1
                self.tutorial_layout.pos_hint={'x': 0, 'y':0}
                if self.lang=="ENG": self.tutorial_label.text="Let's measure the position of the electron" 
                elif self.lang=="ESP": self.tutorial_label.text='A mesurar la posición del electron' 
                else: self.tutorial_label.text="Mesura la posició de l'electró"

                self.tutorial_step=5

                #disabling ploting
                self.plot_button.disabled=True
                #disbaling energies
                self.button0.disabled=True 
                self.button1.disabled=True
                self.button2.disabled=True
                self.button3.disabled=True
                self.button4.disabled=True
                #disabling potential buttons 
                self.button_free.disabled=True
                self.button_harmonic.disabled=True 
                self.button_barrier.disabled=True 
                self.button_water.disabled=True
                self.rubber_button.disabled=True
                #disabling sliders
                self.hooke_slider.disabled=True
                self.x0_slider.disabled=True
                self.center_slider.disabled=True
                self.width_slider.disabled=True
                self.vb_slider.disabled=True
                self.left_slider.disabled=True
                self.right_slider.disabled=True 
                self.vw_slider.disabled=True 

        def tutorial_six_step_green(self): 
                '''six step  green consisting in repating all the porcedura again ''' 
                self.tutorial_layout.size_hint_y=1
                self.tutorial_layout.size_hint_x=1
                self.tutorial_layout.pos_hint={'x': 0, 'y':0}
                if self.lang=="ENG": self.tutorial_label.text="You made it! \n Now repeat the procedure and make another measure" 
                elif self.lang=="ESP": self.tutorial_label.text='Lo lograste! \n Ahora repite el procedimiento y haz otra medida ' 
                else: self.tutorial_label.text="Ho has aconseguit! \n Ara repeteix el procediment is fes una altra mesura"

                self.tutorial_step=6

                #disabling ploting
                self.plot_button.disabled=False
                #disbaling energies
                self.button0.disabled=False
                self.button1.disabled=False
                self.button2.disabled=False
                self.button3.disabled=False
                self.button4.disabled=False
                #disabling potential buttons 
                self.button_free.disabled=False
                self.button_harmonic.disabled=False 
                self.button_barrier.disabled=False
                self.button_water.disabled=False
                self.rubber_button.disabled=False
                #disabling sliders
                self.hooke_slider.disabled=False
                self.x0_slider.disabled=False
                self.center_slider.disabled=False
                self.width_slider.disabled=False
                self.vb_slider.disabled=False
                self.left_slider.disabled=False
                self.right_slider.disabled=False 
                self.vw_slider.disabled=False

        def tutorial_six_step_red(self): 
                '''six step  red consisting in repating all the porcedura again ''' 
                self.tutorial_layout.size_hint_y=1
                self.tutorial_layout.size_hint_x=1
                self.tutorial_layout.pos_hint={'x': 0, 'y':0}
                if self.lang=="ENG": self.tutorial_label.text="You missed! \n Now repeat the procedure and make another measure" 
                elif self.lang=="ESP": self.tutorial_label.text='Fallaste! \n Ahora repite el procedimiento y haz otra medida ' 
                else: self.tutorial_label.text="Has fallat! \n Ara repeteix el procediment is fes una altra mesura"

                self.tutorial_step=6
                
                #disabling ploting
                self.plot_button.disabled=False
                #disbaling energies
                self.button0.disabled=False
                self.button1.disabled=False
                self.button2.disabled=False
                self.button3.disabled=False
                self.button4.disabled=False
                #disabling potential buttons 
                self.button_free.disabled=False
                self.button_harmonic.disabled=False 
                self.button_barrier.disabled=False
                self.button_water.disabled=False
                self.rubber_button.disabled=False
                #disabling sliders
                self.hooke_slider.disabled=False
                self.x0_slider.disabled=False
                self.center_slider.disabled=False
                self.width_slider.disabled=False
                self.vb_slider.disabled=False
                self.left_slider.disabled=False
                self.right_slider.disabled=False 
                self.vw_slider.disabled=False

        def tutorial_final_step(self,*args): 
                '''Tutorial completed''' 
                self.reboot()
                self.menu_transition()
                self.game_button.disabled=False
                self.tutorial_layout.size_hint_y=0
                self.tutorial_layout.size_hint_x=0
                self.tutorial_layout.pos_hint={'x': -1, 'y':-1}



          

     
Builder.load_file("principal_kivy.kv") #d'aquesta manera li podem dir al fitxer kv com volguem

class MygameApp(App): #inherits from app (utilitza les pepietats)
    #    def __init__() No fem INIT perquè ja agafa el init de App
    #aqesta classe fa que la grid aparegui a la pantalla
    def build(self): #self argument de sempre no te arguments
        sm=ScreenManager()
        sm.add_widget(GameWindow(name="game"))
        return sm #estem dibuixant la grid, cridem la classe directament de float Layot

if __name__=="__main__":
#no hi ha run defininda a la class... està dins de App!
        Window.maximize() #opens in full screen 
        MygameApp().run() #run method inside App class
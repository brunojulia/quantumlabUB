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
import scipy.special 
import random 




#classes that work with screenmanager 
class MainWindow(Screen):
        pass 
     
class TrainerWindow(Screen):
        float_plot=ObjectProperty(None)
        measure_layout=ObjectProperty(None)
        target_position=ObjectProperty(None)
        value_n=0#if no button is pressed
        hooke_constant=5 #if it isn't slide it
        x0_harmonic=0 #if there is no sliding
        left_x_b=-0.4 
        right_x_b=0.4 
        Vb=20 #if the slides for the barrier aren't activated
        left_x_w=-0.4 
        right_x_w=0.4 
        Vw=20 #if the slides for the water aren't activated
        potential_type="Free particle" #if no potential button is pressed  
        e_position=None #we put it here so its global
        #We create the list of values x outside so its faster(no need to do it every time)
        n_steps=500 #number of steps taken to integrate the wavefunction 
        L=10.0 #lenght of the box 
        x0=-L/2  
        x_list_values=[x0]
        dx=L/n_steps  
        for i in range(1,n_steps+1):
                x_value=x0+dx*i
                x_list_values.append(x_value)

        #TARGET
        target_position=random.random() #we calculate first target_position 
        target_epsilon=0.2 #target width
        #we check that all the target is in the screen 
        while (target_position-target_epsilon)<0 or (target_position+target_epsilon)>1: 
                target_position=random.random() #we generate a new target position 
        yellow=Image(source="graphs/yellow_target.png",allow_stretch=True,keep_ratio=False)
        red=Image(source="graphs/red_target.png",allow_stretch=True,keep_ratio=False)
        green=Image(source="graphs/green_target.png",allow_stretch=True,keep_ratio=False)


        #HEARTS
        heart=[None]*5
        for i in range(0,5):
                heart[i]=Image(source="graphs/heart_icon.png",allow_stretch=True,keep_ratio=False)
        lives_counter=5 #counter of how many lives we have left. Initially we have 5 lives

        #To control no measure is done without a plot 
        is_plot=False #initially equals to False because no plot is showed 

        #plots left 
        plots_left=3 #Initially we have 3 plots left 

        #score 
        score=0 
        level=1


        #Choosing energies 
        def value_0(self): 
                ''' Changes values o value_n when the button is pressed'''
                if self.plots_left>0: #we can do a plot
                        self.value_n=0 
                        self.button0.background_color=(1,0,0,1)
                        self.button1.background_color=(1,1,1,1)
                        self.button2.background_color=(1,1,1,1)
                        self.button3.background_color=(1,1,1,1)
                        self.button4.background_color=(1,1,1,1)
                        self.plot_energy() #changes the color of every button to remark the one pressed
                else: 
                        self.button_plot_anim() #we animate the plot button and label

        def value_1(self): 
                ''' Changes values o value_n when the button is pressed'''
                if self.plots_left>0: #we can do a plot
                        self.value_n=1 
                        self.button0.background_color=(1,1,1,1)
                        self.button1.background_color=(1,0,0,1)
                        self.button2.background_color=(1,1,1,1)
                        self.button3.background_color=(1,1,1,1)
                        self.button4.background_color=(1,1,1,1)
                        self.plot_energy()
                else: 
                        self.button_plot_anim() #we animate the plot button and label

        def value_2(self): 
                ''' Changes values o value_n when the button is pressed'''
                if self.plots_left>0: #we can do a plot
                        self.value_n=2
                        self.button0.background_color=(1,1,1,1)
                        self.button1.background_color=(1,1,1,1)
                        self.button2.background_color=(1,0,0,1)
                        self.button3.background_color=(1,1,1,1)
                        self.button4.background_color=(1,1,1,1)
                        self.plot_energy()
                else: 
                        self.button_plot_anim() #we animate the plot button and label

        def value_3(self): 
                ''' Changes values o value_n when the button is pressed'''
                if self.plots_left>0: #we can do a plot
                        self.value_n=3
                        self.button0.background_color=(1,1,1,1)
                        self.button1.background_color=(1,1,1,1)
                        self.button2.background_color=(1,1,1,1)
                        self.button3.background_color=(1,0,0,1)
                        self.button4.background_color=(1,1,1,1)
                        self.plot_energy()
                else: 
                        self.button_plot_anim() #we animate the plot button and label

        def value_4(self): 
                ''' Changes values o value_n when the button is pressed'''
                if self.plots_left>0: #we can do a plot
                        self.value_n=4
                        self.button0.background_color=(1,1,1,1)
                        self.button1.background_color=(1,1,1,1)
                        self.button2.background_color=(1,1,1,1)
                        self.button3.background_color=(1,1,1,1)
                        self.button4.background_color=(1,0,0,1)
                        self.plot_energy()
                else:  
                        self.button_plot_anim() #we animate the plot button and label
        
        #choosing potentials 

        def harmonic_potential(self): 
                if self.plots_left>0: #we can do a plot
                        self.potential_type="Harmonic"
                        self.button_harmonic.background_color=(1,0,0,1)
                        self.button_free.background_color=(1,1,1,1)
                        self.button_barrier.background_color=(1,1,1,1)
                        self.button_water.background_color=(1,1,1,1)
                        self.plot_potential()
                else: 
                        self.button_plot_anim() #we animate the plot button and label

        def slider_value_k(self,*args):
                ''' Takes the hooke constant value from a slider''' 
                self.hooke_constant=args[1]
                self.harmonic_potential()  

        def slider_x0(self,*args):
                ''' Takes the hooke constant value from a slider''' 
                self.x0_harmonic=args[1]
                self.harmonic_potential()
                

        def free_particle(self): 
                if self.plots_left>0: #we can do a plot
                        self.potential_type="Free particle"
                        self.button_harmonic.background_color=(1,1,1,1)
                        self.button_free.background_color=(1,0,0,1)
                        self.button_barrier.background_color=(1,1,1,1)
                        self.button_water.background_color=(1,1,1,1)
                        self.plot_potential()
                else:
                        self.button_plot_anim() #we animate the plot button and label


        def left_barrier(self,*args): 
                ''' Takes the x position left value of the barrier from a slider''' 
                self.left_x_b=args[1]
                self.barrier()
                
        def right_barrier(self,*args): 
                ''' Takes the x position right value of the barrier from a slider''' 
                self.right_x_b=args[1]
                self.barrier()
                
        def V_barrier(self,*args): 
                ''' Takes the V value of the barrier from a slider''' 
                self.Vb=args[1]
                self.barrier()
                
        def barrier(self): 
                if self.plots_left>0: #we can do a plot
                        self.potential_type="Barrier"
                        self.button_harmonic.background_color=(1,1,1,1)
                        self.button_free.background_color=(1,1,1,1)
                        self.button_barrier.background_color=(1,0,0,1)
                        self.button_water.background_color=(1,1,1,1)
                        self.plot_potential()
                else: 
                        self.button_plot_anim() #we animate the plot button and label



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
                if self.plots_left>0: #we can do a plot
                        self.potential_type="Well"
                        self.button_harmonic.background_color=(1,1,1,1)
                        self.button_free.background_color=(1,1,1,1)
                        self.button_barrier.background_color=(1,1,1,1)
                        self.button_water.background_color=(1,0,0,1)
                        self.plot_potential()
                else: 
                        self.button_plot_anim() #we animate the plot button and label



        def potential(self,x): 
                '''Gives the potential'''
                if self.potential_type=="Harmonic":
                        V=(1/2)*self.hooke_constant*(x-self.x0_harmonic)**2
                                 
                if self.potential_type=="Free particle": #if no potential button is pressed we have a free particle
                        V=0 
                if self.potential_type=="Barrier":
                        V=0 #if the condition is not satisfied V=0
                        #we check that the left position is smaller than the right one 
                        if self.left_x_b>self.right_x_b: #if the left one is bigger
                                self.right_x_b=self.left_x_b+0.6
                        if self.left_x_b<self.right_x_b: 
                                if x<self.left_x_b: V=0 
                                if x>self.left_x_b and x <self.right_x_b: V=self.Vb  
                                if x>self.right_x_b: V=0  
                if self.potential_type=="Well":
                        V=0 #if the condition is not satisfied V=0
                        #we check that the left position is smaller than the right one 
                        if self.left_x_w>self.right_x_w:
                                self.right_x_w=self.left_x_w+0.6
                        if self.left_x_w<self.right_x_w: 
                                if x<self.left_x_w: V=self.Vw 
                                if x>self.left_x_w and x <self.right_x_w: V=0  
                                if x>self.right_x_w: V=self.Vw 
                #for any potential the partcile is inside the box so: 
                if x<-(self.L/2-self.dx/2) : V=10**10 #infinite potential
                if x>(self.L/2-self.dx/2): V=10**10 

                return V 

        def matrix(self): 
                '''This function creates the numpy array we will work with. '''

                #define parameters 
                h2_me=7.6199  # (eV*A) h**2/me

                #we create the matrix with all zeros 
                H_matrix=np.zeros((self.n_steps+1,self.n_steps+1)) #matrix 501x501 
                for i in range(0,self.n_steps+1): #rows 
                        
                        if i==0: #first row
                                H_matrix[i,0]= -h2_me/(2*self.dx**2)+self.potential(self.x_list_values[i])
                                H_matrix[i,1]= h2_me/(self.dx**2)
                                H_matrix[i,2]= -h2_me/(2*self.dx**2)

                        elif i==self.n_steps: #last row 
                                H_matrix[i,self.n_steps]=-h2_me/(2*self.dx**2)+self.potential(self.x_list_values[i])
                                H_matrix[i,self.n_steps-1]=h2_me/(self.dx**2)
                                H_matrix[i,self.n_steps-2]=-h2_me/(2*self.dx**2)
                
                        else: #not first nor last row 
                                H_matrix[i,i]= h2_me/(self.dx**2)+self.potential(self.x_list_values[i])
                                H_matrix[i,i-1]=-h2_me/(2*self.dx**2)
                                H_matrix[i,i+1]=-h2_me/(2*self.dx**2)
        
                return H_matrix 

        def wave_function(self):
                '''This function finds the eigenvalues and eigenvectors and normalises the eigenvector we want to plot 
                Returns the wave function squared and the energy used'''
                A=self.matrix()
                eigenval,eigenvec=np.linalg.eig(A) #we compute eigenvalues and eigenvectors 
                #we sort the eigenvalues and eigenvectors 
                index=np.argsort(eigenval)
                eigenval=eigenval[index] #list of eigenvalues sorted
                eigenvec=eigenvec[:,index] #lis of eigenvectors sorted accordingly 
                
                #we choose the energy 
                E=eigenval[self.value_n] 
                wave=eigenvec[:,self.value_n]
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
                phi_norm=[element/math.sqrt(integral) for element in wave] #and we have to wave function normalised 

                phi_square=[component**2 for component in phi_norm]

                return phi_square,E         
        
        def plot_energy(self):
                '''This function plots the energy line for a estatic potential'''
                plt.clf()
                self.measure_layout.clear_widgets()
                fig, ax_phi=plt.subplots()
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                ax_phi.set_ylim((0,1)) #maximum of phi_axis
                ax_phi.xaxis.labelpad = -1
                #we compute the eigen_values 
                A=self.matrix()
                eigenval,eigenvec=np.linalg.eig(A) #we compute eigenvalues and eigenvectors 
                #we sort the eigenvalues and eigenvectors 
                index=np.argsort(eigenval)
                eigenval=eigenval[index] #list of eigenvalues sorted
                eigenvec=eigenvec[:,index] #lis of eigenvectors sorted accordingly 
                E=eigenval[self.value_n]  #we assign Energy 
                

                y2=0
                V_plot=[self.potential(s) for s in self.x_list_values] 
                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_list_values,V_plot, label="V(x)" , color='tab:blue') 
                ax_V.axhline(y=E, color='g', linestyle='-',label="E") #we plot the Energy value too 
                ax_V.fill_between(self.x_list_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                if E<40: ax_V.set_ylim((0,40)) #limit in potential axis
                else: ax_V.set_ylim((0,E+5)) 
                ax_V.legend(loc="upper right")

                #MAKING THE TITLE OF THE GRAPH
                if self.potential_type=="Harmonic":
                        string_var="    $K(eV/\AA^2)$="+str(self.hooke_constant)
                if self.potential_type=="Free particle": string_var=""
                if self.potential_type=="Barrier":
                        string_var="    $V(eV)$="+str(self.Vb)
                if self.potential_type=="Well": 
                        string_var="    $V(eV)$="+str(self.Vw)

                plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4]+"   V(x)= "+str(self.potential_type)+string_var
                        ,loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()
                self.float_plot.add_widget(canvas_plot)
                self.is_plot=False #no wave function plotted

        def plot_potential(self): 
                '''This function plots the potential
                This function won't plot energy in order to not having to compute the eigenvalues
                every time the potential is changing'''
                #clear previous plot 
                plt.clf()
                self.measure_layout.clear_widgets()


                fig, ax_phi=plt.subplots()
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                ax_phi.set_ylim((0,1)) #maximum of phi_axis
                ax_phi.xaxis.labelpad = -1

                y2=0
                V_plot=[self.potential(s) for s in self.x_list_values] 
                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_list_values,V_plot, label="V(x)" , color='tab:blue') 
                ax_V.fill_between(self.x_list_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                ax_V.set_ylim((0,40)) 
                ax_V.legend(loc="upper right")

                #MAKING THE TITLE OF THE GRAPH
                if self.potential_type=="Harmonic":
                        string_var="    $K(eV/\AA^2)$="+str(self.hooke_constant)
                if self.potential_type=="Free particle": string_var=""
                if self.potential_type=="Barrier":
                        string_var="    $V(eV)$="+str(self.Vb)
                if self.potential_type=="Well": 
                        string_var="    $V(eV)$="+str(self.Vw)

                plt.title("V(x)= "+str(self.potential_type)+string_var
                        ,loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()

                self.float_plot.add_widget(canvas_plot)
                self.is_plot=False #no wave function plotted

        def plot_wave_function(self,*args):
                '''This function plots into the window the wave function.
                It also plots the potential. Previous to plotting it computes the eigenvalues
                and computes the eigenvector. '''
 
                if self.plots_left>0: #we can do a plot
                        self.plots_left-=1  
                        self.plots_label.text=str(self.plots_left) #we have to update the plots left
                        #we animate the plotsleft 
                        label_animation1=Animation(font_size=25,duration=0.5)
                        label_animation1+=Animation(font_size=20,duration=0.5)
                        label_animation1.start(self.plots_label)
                        plt.clf()
                        self.measure_layout.clear_widgets()
                        phi_square, E=self.wave_function()#we compute the eigen_values  and assign E 
                 
                        y2=0
                        #let's plot:
                        V_plot=[self.potential(k) for k in self.x_list_values] #compute potential for every x_position
                        fig, ax_phi=plt.subplots() 
                        ax_phi.set_xlabel(r"$x(\AA)$")
                        ax_phi.set_ylabel(r"$ \phi^2$")
                        ax_phi.xaxis.labelpad = -1

                        #CHECKING
                        #we check free partcile with theory to see if we are right 
                        #x_4= np.linspace(-5,5,401)
                        #y_4=((1/math.sqrt(5))*np.sin(4*math.pi*x_4/10))**2 #plot E4
                        #y_4=((1/math.sqrt(5))*np.cos(5*math.pi*x_4/10))**2 #plot E5
                        #ax_phi.plot(x_4,y_4,label= "Analytic", color='blue')

                        #we check HARMONIC behavoior
                        #h2_me=(7.6199)  # (eV*A) h**2/me
                        #h_planck=(6.582119569*10**(-16)/(2*np.pi))
                        #me_evA=((h_planck**2)/h2_me)
                        #w_freq=np.sqrt(self.hooke_constant/me_evA) #computing w 
                        #constant=np.sqrt(np.sqrt(np.pi*h_planck/np.sqrt(self.hooke_constant*me_evA))*(2**self.value_n)*np.math.factorial(self.value_n))
                        #constant=1/constant
                        #E_harmonica=h_planck*w_freq+(self.value_n+1/2) #computing energy for the oscillator analitically
                        #psi=np.sqrt(me_evA*w_freq/h_planck)*(x_4-self.x0_harmonic) #change of variable
                        #hermite=scipy.special.eval_hermite(self.value_n,psi) #computing the hermite polinomail
                        #phi_harm=constant*np.exp(-me_evA*w_freq*((x_4-self.x0_harmonic)**2)/(2*h_planck))*hermite #computing the wave function
                        #phi_harm=phi_harm**2 
                        #print(self.hooke_constant)

                        #ax_phi.plot(x_4,phi_harm,label= "Analytic", color='blue')

                        #we check integral value
                        #integral_prova=0 
                        #for k in range(0,self.n_steps+1): 
                                #if k==0 or k==self.n_steps: #extrem values 
                                        #integral_prova=integral_prova +(phi_square[k])/3
                                #elif (k % 2) == 0:  #even number
                                        #integral_prova=integral_prova+2*(phi_square[k])/3
                                #else: #odd number 
                                        #integral_prova=integral_prova+4*(phi_square[k])/3
                        #integral_prova=integral_prova*self.dx 
                        #print(integral_prova)


                        ax_phi.plot(self.x_list_values,phi_square,label=r"$ \phi^2(x)$" , color='tab:red')
                        ax_phi.set_ylim((0,max(phi_square)+0.2)) #maximum of phi_axis= maxim of probability +0.2
                        ax_phi.legend(loc="upper left")

                        #we plot the potential
                        ax_V = ax_phi.twinx() #same x_axis
                        ax_V.set_ylabel(r"$V(eV)$")
                        ax_V.plot(self.x_list_values,V_plot, label="V(x)" , color='tab:blue')
                        ax_V.axhline(y=E, color='g', linestyle='-',label="E") #we plot the Energy value too 
                        ax_V.fill_between(self.x_list_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                        if E<35: ax_V.set_ylim((0,40))
                        else: ax_V.set_ylim((0,E+5)) 
                        ax_V.legend(loc="upper right")
                        #MAKING THE TITLE OF THE GRAPH
                        if self.potential_type=="Harmonic":
                                string_var="    $K(eV/\AA^2)$="+str(self.hooke_constant)
                        if self.potential_type=="Free particle": string_var=""
                        if self.potential_type=="Barrier":
                                string_var="    $V(eV)$="+str(self.Vb)
                        if self.potential_type=="Well": 
                                string_var="    $V(eV)$="+str(self.Vw)

                        plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4]+"   V(x)= "+str(self.potential_type)+string_var
                                ,loc="right")
                        
                        canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                        self.float_plot.clear_widgets()
                        self.float_plot.add_widget(canvas_plot)
                        self.is_plot=True #there is a wave function plotted


                
                else: #we animate the measure button
                        self.button_plot_anim() #we animate the plot button and plotsleft label

        def button_plot_anim(self): 
                '''Animates the plot button when there's no plots left'''
                button_animation=Animation(size_hint_x=0.085,size_hint_y=0.095,duration=0.5)
                button_animation+=Animation(size_hint_x=0.075,size_hint_y=0.08,duration=0.5)
                button_animation.start(self.measure_button)

                #we animate the plotsleft 
                label_animation=Animation(font_size=30,duration=0.5)
                label_animation+=Animation(font_size=20,duration=0.5)
                label_animation.start(self.plots_label)

        def loading_anim(self): 
                '''Triggers loading animation'''
                loading_animation=Animation(color=(0,0,0,1),duration=0.000005)
                loading_animation+=Animation(color=(0,0,0,1),duration=1)
                loading_animation+=Animation(color=(0,0,0,0),duration=0.00005)
                loading_animation.start(self.loading_label) 
                loading_animation.bind(on_complete=self.plot_wave_function)

        def appear_e(self,*args): 
                '''Plots the electron after pressing the measure button'''
                if self.is_plot==False: #There is no wave function plotted 
                        #we don't measure and do something 
                        button_animation=Animation(size_hint_x=0.085,size_hint_y=0.095,duration=0.5)
                        button_animation+=Animation(size_hint_x=0.075,size_hint_y=0.08,duration=0.5)
                        button_animation.start(self.plot_button)
                else: #there is a wave function plotted 
                        self.plots_left=3  
                        self.plots_label.text=str(self.plots_left)

                        probabilities,E=self.wave_function() #compute the probabilities and energy 
                        position=random.choices(self.x_list_values,weights=probabilities,k=1) #computes the position accordingly to WF
                        self.e_position=(position[0]+5)/10 #normalising to 0-1 in the box layout
                        e_graphs=Image(source="graphs/electron.png",allow_stretch=True,keep_ratio=False) #we import image
                        #clear image before 
                        e_grid=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                        e_grid.pos_hint={"center_x":self.e_position,"center_y":0.5} #we put the gridlayout in the position we want
                        e_grid.size_hint_x=0.06
                        e_grid.size_hint_y=0.25 #size of gridlayout
                        e_grid.add_widget(e_graphs) #add the image to the gridLayout
                        self.measure_layout.add_widget(e_grid) #we add the electron grid layout to the float layout
                        if self.e_position<(self.target_position-self.target_epsilon): #missed measure
                                self.grid_target.clear_widgets() #erase previous target 
                                self.grid_target.add_widget(self.red)#we turn into red the rectangle
                                target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                target_anim.start(self.grid_target) 
                                target_anim.bind(on_complete=self.same_target)
                                #when the animation is done we change the color of the rectangle
                        elif self.e_position>(self.target_position+self.target_epsilon):  #red
                                self.grid_target.clear_widgets()
                                self.grid_target.add_widget(self.red) #we turn into red the rectangle
                                target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                target_anim.start(self.grid_target) 
                                target_anim.bind(on_complete=self.same_target) #shoot missed

                                #when the animation is done we change the color of the rectangle  
                        else: #green
                                self.grid_target.clear_widgets()
                                self.grid_target.add_widget(self.green)#we turn into green the rectangle
                                target_anim=Animation(size_hint_x=self.target_epsilon*2, size_hint_y=1,duration=1) #we make the target appear
                                target_anim+=Animation(size_hint_x=0, size_hint_y=0,duration=0.005)
                                target_anim.start(self.grid_target) 
                                target_anim.bind(on_complete=self.new_target) #green

                        #when the animation is done we generate a new rectangle  
        
        def appear_multi_e(self,*args): 
                '''This function measures the position of the n electrons and plots the electrons into the screen. 
                acts like self.appear_e but with n_e electrons'''
                n_e=200
                probabilities,E=self.wave_function() #compute the probabilities and energy 
                position=random.choices(self.x_list_values,weights=probabilities,k=n_e) #computes the position accordingly to WF
                #we create a lists of the gridslayouts with are working with 
                gridlayout=[None]*n_e #list of 100 elements
                for i in range(0,n_e): 
                        e_graphs=Image(source="graphs/electron.png",allow_stretch=True,keep_ratio=False) #we import image
                        gridlayout[i]=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                        e_position=(position[i]+5)/10 #normalising to 0-1 in the box layout
                        gridlayout[i].pos_hint={"center_x":e_position,"center_y":0.5} #we put the gridlayout in the position we want
                        gridlayout[i].size_hint_x=0.07/4
                        gridlayout[i].size_hint_y=0.25/4 #size of gridlayout
                        gridlayout[i].add_widget(e_graphs) #add the image to the gridLayout
                        self.measure_layout.add_widget(gridlayout[i]) #we add the grid layout to the float layout

        def new_target(self,*args): #GREEN
                '''Generates a new target in another position after having acomplished a green'''
                self.measure_layout.clear_widgets()
                self.grid_target.clear_widgets() #erase previous target 
                self.grid_target.add_widget(self.yellow)#we turn into yellow 
                #we generate a new position 
                if self.target_epsilon>0.06: #if it's bigger than 0.05
                        self.target_epsilon=self.target_epsilon-0.025
                else: #it's smaller 
                        self.level+=1 #plus one in level 
                        level_animation1=Animation(color=(233/255, 179/255, 7/255, 1),duration=0.5)
                        level_animation1+=Animation(color=(0,0,0,1),duration=0.5)
                        level_animation1.start(self.level_label)
                        
                        self.level_label.text="LEVEL = "+str(self.level)

                        self.target_epsilon=0.2 #resize

                
                self.target_position=random.random()
                #we check that all the target is in the screen 
                while (self.target_position-self.target_epsilon)<0 or (self.target_position+self.target_epsilon)>1: 
                        self.target_position=random.random() #we generate a new target position 

                self.grid_target.pos_hint={"center_x": self.target_position,"center_y":0.5}
                self.grid_target.size_hint_x=self.target_epsilon*2 #resize
                self.grid_target.size_hint_y=1


                #UPDATING THE SCORE + ELECTRON ANIMATION TO THE SCORE
                e_graphs=Image(source="graphs/electron.png",allow_stretch=True,keep_ratio=False) #we import image
                e_grid=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                e_grid.pos_hint={"center_x":self.e_position,"center_y":0.5} #we put the gridlayout in the position we want
                e_grid.size_hint_x=0.06
                e_grid.size_hint_y=0.25 #size of gridlayout
                e_grid.add_widget(e_graphs) #add the image to the gridLayout
                self.measure_layout.add_widget(e_grid)
                e_animation=Animation(pos_hint={"center_x":self.e_position,"center_y":0.5,},duration=0.005)
                e_animation+=Animation(pos_hint={"center_x":-0.15,"center_y":4.35,},duration=1.5) #GOES TO THE SCORE
                e_animation+=Animation(size_hint_x=0,size_hint_y=0,duration=0.0005) #THE DISAPPEARS
                e_animation.start(e_grid)
                e_animation.bind(on_complete=self.score_update)
       
        def same_target(self,*args): #RED
                '''Generates same target in the same position after having missed'''
                self.measure_layout.clear_widgets()
                self.grid_target.clear_widgets() #erase previous target 
                self.grid_target.add_widget(self.yellow)#we turn into yellow
                self.grid_target.size_hint_x=self.target_epsilon*2 #resize
                self.grid_target.size_hint_y=1 

                #we have lost a live: we erase one heart 
                heart_loss_anim=Animation(size_hint_y=1,size_hint_x=0.05) 
                heart_loss_anim+=Animation(size_hint_y=1.1,size_hint_x=0.07,duration=0.3) #we make it bigger
                heart_loss_anim+=Animation(size_hint_y=1,size_hint_x=0.05,duration=0.2)
                
                
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
                        heart_loss_anim.start(self.heart1_grid)
                        heart_loss_anim.bind(on_complete=self.heart1_disappear)

                #after erasing the heart we update the counter
                self.lives_counter-=1  
                if self.lives_counter==0: #if we are out of lives we show the Game Over image
                         self.bug_layout.size_hint_x=0
                         self.bug_layout.size_hint_y=0
                         self.game_over_layout.size_hint_x=1 
                         self.game_over_layout.size_hint_y=1
                         self.final_score_label.text="FINAL SCORE = "+ str(self.score)
                         self.final_score_label.color=(233/255, 179/255, 7/255, 1)

        def heart5_disappear(self,*args): 
                self.heart5_grid.clear_widgets()
        def heart4_disappear(self,*args): 
                self.heart4_grid.clear_widgets()
        def heart3_disappear(self,*args): 
                self.heart3_grid.clear_widgets()
        def heart2_disappear(self,*args): 
                self.heart2_grid.clear_widgets()
        def heart1_disappear(self,*args): 
                self.heart2_grid.clear_widgets()


        def score_update(self,*args): #UPTADES THE SCORE
                '''Updates the score, called by new target''' 
                self.score+=1 #we add one point 
                self.score_label.text=" SCORE = " +str(self.score)
                label_animation1=Animation(color=(233/255, 179/255, 7/255, 1),duration=0.5)
                label_animation1+=Animation(color=(0,0,0,1),duration=0.5)
                #e_animation.bind(on_complete=
                label_animation1.start(self.score_label)

        def reboot(self): 
                '''This function reboots the trainer so when you enter the game it's all new and fresh. 
                It's activated when you press the button '''

                #we delete the gameover layout again 
                self.game_over_layout.size_hint_x=0 
                self.game_over_layout.size_hint_y=0 

                #we reactivate all lives 
                self.lives_counter=5 

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
                self.plots_label.text=str(self.plots_left)

                #score 
                self.score=0 #we add one point 
                self.score_label.text=" SCORE = " +str(self.score)

                self.level=1
                self.level_label.text="LEVEL = "+str(self.level)


                #new target 
                self.measure_layout.clear_widgets()
                self.grid_target.clear_widgets() #erase previous target 
                self.grid_target.add_widget(self.yellow)#we turn into yellow 
                #we generate a new position 
                self.target_position=random.random()
                self.target_epsilon=0.2
                #we check that all the target is in the screen 
                while (self.target_position-self.target_epsilon)<0 or (self.target_position+self.target_epsilon)>1: 
                        self.target_position=random.random() #we generate a new target position 

                self.grid_target.pos_hint={"center_x": self.target_position,"center_y":0.5}
                self.grid_target.size_hint_x=self.target_epsilon*2 #resize
                self.grid_target.size_hint_y=1

                #we clear the plot 
                self.float_plot.clear_widgets()
                self.is_plot=False #no wave function plotted

                #we overpaint again the bug layout 
                self.bug_layout.size_hint_x= 0.15725
                self.bug_layout.size_hint_y=0.16







                     
class GameWindow(Screen):
        pass    

     
Builder.load_file("principal_kivy.kv") #d'aquesta manera li podem dir al fitxer kv com volguem

class MygameApp(App): #inherits from app (utilitza les pepietats)
    #    def __init__() No fem INIT perquè ja agafa el init de App
    #aqesta classe fa que la grid aparegui a la pantalla
    def build(self): #self argument de sempre no te arguments
        sm=ScreenManager()
        sm.add_widget(MainWindow(name="menu"))
        sm.add_widget(TrainerWindow(name="trainer"))
        sm.add_widget(GameWindow(name="game"))
        return sm #estem dibuixant la grid, cridem la classe directament de float Layot

if __name__=="__main__":
#no hi ha run defininda a la class... està dins de App!
        Window.maximize() #opens in full screen 
        MygameApp().run() #run method inside App class
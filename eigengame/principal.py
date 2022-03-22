import kivy
from kivy.app import App 
from kivy.lang import Builder 
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
        hooke_constant=0 #if it isn't slide it
        x0_harmonic=0 #if there is no sliding
        left_x_b=0 
        right_x_b=0 
        Vb=0 #if the slides for the barrier aren't activated
        left_x_w=0 
        right_x_w=0 
        Vw=0 #if the slides for the water aren't activated
        potential_type="Free particle" #if no potential button is pressed  
        #we create the list of values x outside so its faster(no need to do it every time)
        n_steps=500 #number of steps taken to integrate the wavefunction 
        L=10.0 #lenght of the box 
        x0=-L/2  
        x_list_values=[x0]
        dx=L/n_steps  
        for i in range(1,n_steps+1):
                x_value=x0+dx*i
                x_list_values.append(x_value)

        target_position=random.random() #we calculate first target_position 
        target_epsilon=0.05

        yellow=Image(source="graphs/yellow_target.PNG",allow_stretch=True,keep_ratio=False)
        red=Image(source="graphs/red_target.PNG",allow_stretch=True,keep_ratio=False)
        green=Image(source="graphs/green_target.PNG",allow_stretch=True,keep_ratio=False)
        


        #Choosing energies 
        def value_0(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=0 
                self.button0.background_color=(1,0,0,1)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(1,1,1,1) #changes the color of every button to remark the one pressed

        def value_1(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=1 
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(1,0,0,1)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(1,1,1,1)

        def value_2(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=2
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(1,0,0,1)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(1,1,1,1)

        def value_3(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=3
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(1,0,0,1)
                self.button4.background_color=(1,1,1,1)

        def value_4(self): 
                ''' Changes values o value_n when the button is pressed'''
                self.value_n=4
                self.button0.background_color=(1,1,1,1)
                self.button1.background_color=(1,1,1,1)
                self.button2.background_color=(1,1,1,1)
                self.button3.background_color=(1,1,1,1)
                self.button4.background_color=(1,0,0,1)
        
        #choosing potentials 

        def harmonic_potential(self): 
                self.potential_type="Harmonic"
                self.button_harmonic.background_color=(1,0,0,1)
                self.button_free.background_color=(1,1,1,1)
                self.button_barrier.background_color=(1,1,1,1)
                self.button_water.background_color=(1,1,1,1)
        def slider_value_k(self,*args):
                ''' Takes the hooke constant value from a slider''' 
                self.hooke_constant=args[1]
                self.harmonic_potential()
                self.plot_potential()
        def slider_x0(self,*args):
                ''' Takes the hooke constant value from a slider''' 
                self.x0_harmonic=args[1]
                self.harmonic_potential()
                self.plot_potential()

        def free_particle(self): 
                self.potential_type="Free particle"
                self.button_harmonic.background_color=(1,1,1,1)
                self.button_free.background_color=(1,0,0,1)
                self.button_barrier.background_color=(1,1,1,1)
                self.button_water.background_color=(1,1,1,1)


        def left_barrier(self,*args): 
                ''' Takes the x position left value of the barrier from a slider''' 
                self.left_x_b=args[1]
                self.barrier()
                self.plot_potential()
        def right_barrier(self,*args): 
                ''' Takes the x position right value of the barrier from a slider''' 
                self.right_x_b=args[1]
                self.barrier()
                self.plot_potential()
        def V_barrier(self,*args): 
                ''' Takes the V value of the barrier from a slider''' 
                self.Vb=args[1]
                self.barrier()
                self.plot_potential()
        def barrier(self): 
                self.potential_type="Barrier"
                self.button_harmonic.background_color=(1,1,1,1)
                self.button_free.background_color=(1,1,1,1)
                self.button_barrier.background_color=(1,0,0,1)
                self.button_water.background_color=(1,1,1,1)

        def left_water(self,*args): 
                ''' Takes the x position left value of the barrier from a slider''' 
                self.left_x_w=args[1]
                self.water_well()
                self.plot_potential()
        def right_water(self,*args): 
                ''' Takes the x position right value of the barrier from a slider''' 
                self.right_x_w=args[1]
                self.water_well()
                self.plot_potential()
        def V_water(self,*args): 
                ''' Takes the V value of the barrier from a slider''' 
                self.Vw=args[1]
                self.water_well()
                self.plot_potential()
        def water_well(self): 
                self.potential_type="well"
                self.button_harmonic.background_color=(1,1,1,1)
                self.button_free.background_color=(1,1,1,1)
                self.button_barrier.background_color=(1,1,1,1)
                self.button_water.background_color=(1,0,0,1)


        def potential(self,x): 
                '''Gives the potential'''
                if self.potential_type=="Harmonic":
                        V=(1/2)*self.hooke_constant*(x-self.x0_harmonic)**2
                                 
                if self.potential_type=="Free particle": #if no potential button is pressed we have a free particle
                        V=0 
                if self.potential_type=="Barrier":
                        V=0 #if the condition is not satisfied V=0
                        #we check that the left position is smaller than the right one 
                        if self.left_x_b<self.right_x_b: 
                                if x<self.left_x_b: V=0 
                                if x>self.left_x_b and x <self.right_x_b: V=self.Vb  
                                if x>self.right_x_b: V=0  
                if self.potential_type=="well":
                        V=0 #if the condition is not satisfied V=0
                        #we check that the left position is smaller than the right one 
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
                plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4]+"   V(x)= "+str(self.potential_type),loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()
                self.float_plot.add_widget(canvas_plot)

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
                plt.title("V(x)= "+str(self.potential_type),loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()
                self.float_plot.add_widget(canvas_plot)

        def plot_wave_function(self):
                '''This function plots into the window the wave function.
                It also plots the potential. Previous to plotting it computes the eigenvalues
                and computes the eigenvector. '''
                #clear previous plot 
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
                plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4]+"   V(x)= "+str(self.potential_type),loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.clear_widgets()
                self.float_plot.add_widget(canvas_plot)
                
        def measure(self): 
                '''This function measures the position of the electron and plots the electron into the screen
                Calls another function to plot the electron when the animations is complete'''
                self.measure_layout.clear_widgets()  #we clear any electron plotted before
                #we make n_rays 
                n_rays=1
                thunder_grid=[None]*n_rays #empty list of 4 elements 
                thunder_anim=[None]*n_rays
                thunder_graphs=[None]*n_rays
                y_ray=0
                for ray in range(0,n_rays):
                        y_ray+=1/(n_rays+1)
                        thunder_grid[ray]=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                        thunder_grid[ray].pos_hint={"center_x":0.5,"center_y":y_ray} #we put the gridlayout in the middle
                        thunder_grid[ray].size_hint_x=0
                        thunder_grid[ray].size_hint_y=0
                        thunder_grid[ray].add_widget(
                                Image(source="graphs/thunder.PNG",allow_stretch=True,keep_ratio=False)) #we import image       
                        thunder_anim[ray]= Animation(size_hint_x=1,size_hint_y=1,duration=0.005) #we make appear the ray
                        thunder_anim[ray]+=Animation(size_hint_x=0,size_hint_y=0,duration=0.005) #we make disappear the ray
                        thunder_anim[ray].start(thunder_grid[ray])
                        self.measure_layout.add_widget(thunder_grid[ray]) #we add the thundergrid layout to the float layout
                
                thunder_anim[n_rays-1].bind(on_complete=self.appear_e) #when the animation is complete we call appear electron
        
        def multi_measure(self): 
                '''This function measures the position of the electron and plots the electron into the screen
                Calls another function to plot the electron when the animations is complete'''
                self.measure_layout.clear_widgets()  #we clear any electron plotted before
                #we make n_rays 
                n_rays=1
                thunder_grid=[None]*n_rays #empty list of 4 elements 
                thunder_anim=[None]*n_rays
                thunder_graphs=[None]*n_rays
                y_ray=0
                for ray in range(0,n_rays):
                        y_ray+=1/(n_rays+1)
                        thunder_grid[ray]=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                        thunder_grid[ray].pos_hint={"center_x":0.5,"center_y":y_ray} #we put the gridlayout in the middle
                        thunder_grid[ray].size_hint_x=0
                        thunder_grid[ray].size_hint_y=0
                        thunder_grid[ray].add_widget(
                                Image(source="graphs/thunder.PNG",allow_stretch=True,keep_ratio=False)) #we import image       
                        thunder_anim[ray]= Animation(size_hint_x=1,size_hint_y=1,duration=0.005) #we make appear the ray
                        thunder_anim[ray]+=Animation(size_hint_x=0,size_hint_y=0,duration=0.005) #we make disappear the ray
                        thunder_anim[ray].start(thunder_grid[ray])
                        self.measure_layout.add_widget(thunder_grid[ray]) #we add the thundergrid layout to the float layout
                
                thunder_anim[n_rays-1].bind(on_complete=self.appear_multi_e) #when the animation is complete 

        def appear_e(self,*args): 
                '''Plots the elctron'''
                probabilities,E=self.wave_function() #compute the probabilities and energy 
                position=random.choices(self.x_list_values,weights=probabilities,k=1) #computes the position accordingly to WF
                e_position=(position[0]+5)/10 #normalising to 0-1 in the box layout
                e_graphs=Image(source="graphs/electron.PNG",allow_stretch=True,keep_ratio=False) #we import image
                #clear image before 
                e_grid=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                e_grid.pos_hint={"center_x":e_position,"center_y":0.5} #we put the gridlayout in the position we want
                e_grid.size_hint_x=0.06
                e_grid.size_hint_y=0.25 #size of gridlayout
                e_grid.add_widget(e_graphs) #add the image to the gridLayout
                self.measure_layout.add_widget(e_grid) #we add the electron grid layout to the float layout
                if e_position<(self.target_position-self.target_epsilon): 
                        self.grid_target.clear_widgets() #erase previous target 
                        self.grid_target.add_widget(self.red)#we turn into red the rectangle
                        #self.grid_target.pos_hint={"center_x": 0,"center_y":0.5}
                        #when the animation is done we generate a new rectangle  
                elif e_position>(self.target_position+self.target_epsilon): 
                        self.grid_target.clear_widgets()
                        self.grid_target.add_widget(self.red) #we turn into red the rectangle 

                        #when the animation is done we generate a new rectangle  
                else: 
                       self.grid_target.clear_widgets()
                       self.grid_target.add_widget(self.green)#we turn into green the rectangle
                

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
                        e_graphs=Image(source="graphs/electron.PNG",allow_stretch=True,keep_ratio=False) #we import image
                        gridlayout[i]=GridLayout(rows=1,cols=1) #we create a gridlayout to put the image 
                        e_position=(position[i]+5)/10 #normalising to 0-1 in the box layout
                        gridlayout[i].pos_hint={"center_x":e_position,"center_y":0.5} #we put the gridlayout in the position we want
                        gridlayout[i].size_hint_x=0.07/4
                        gridlayout[i].size_hint_y=0.25/4 #size of gridlayout
                        gridlayout[i].add_widget(e_graphs) #add the image to the gridLayout
                        self.measure_layout.add_widget(gridlayout[i]) #we add the grid layout to the float layout


        def target_move(self,*args): 
                '''Changes x position of the target'''
                self.target_position=random.random()
             





                     
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
    MygameApp().run() #run method inside App class
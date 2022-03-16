import kivy
from kivy.app import App 
from kivy.lang import Builder 
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.uix.floatlayout import FloatLayout 
from kivy.uix.widget import Widget
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.properties import ObjectProperty
import matplotlib.pyplot as plt 
import math as math
import numpy as np  



#classes that work with screenmanager 
class MainWindow(Screen):
        pass 
     
class TrainerWindow(Screen):
        float_plot=ObjectProperty(None)
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
        n_steps=400 #number of steps taken to integrate the wavefunction 
        L=10.0 #lenght of the box 
        x0=-L/2  
        x_list_values=[x0]
        dx=L/n_steps  
        for i in range(1,n_steps+1):
                x_value=x0+dx*i
                x_list_values.append(x_value)


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
                self.potential_type="Water_well"
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
                if self.potential_type=="Water_well":
                        V=0 #if the condition is not satisfied V=0
                        #we check that the left position is smaller than the right one 
                        if self.left_x_w<self.right_x_w: 
                                if x<self.left_x_w: V=self.Vw 
                                if x>self.left_x_w and x <self.right_x_w: V=0  
                                if x>self.right_x_w: V=self.Vw 

                return V 

        def derivatives(self,x,yin,energy):
                '''This function returns f(x)=dy/dx returns the derivatives of the dependent variable. 
                Takes as an argument value x(A), the independent variable at wich we are, and yin the values of y at x. 
                Yin should be a list such as [phi,dphi/dx] 
                Emergy(eV) is the energy we are working with
                Returns de derivative dyout. 
                This fucntion is called inside RK4 '''
                h2_me=7.6199  # (eV*A) h**2/me
        
                phi=yin[0] #we assign the value of the wave fucntion
                p=yin[1] #also its derivative

                dphi=p #we compute dphi/dx

                dp= (2/h2_me)*(self.potential(x)-energy)*phi #we compute dphi**2/dx**2 

                dyout=[dphi,dp]
                return dyout #return a list with dphi and dp 

        def RK4(self,x,dx,yin,energy): 
                '''THis function makes one step of a RK4 intergation of an ordinary diferential equation. 
                x i the dependent variable at which we start, dx is the step we take. 
                yin are the values of the function at x: should be a list such as [phi,dphi/dx] 
                , yout(return) are the values at x+dx'''
        
                k1=self.derivatives(x,yin,energy) #k1 is a list [dphi,p]
                yytemp=[0,0] #we assign random numbers to yytemp in order to have a 2d list
                yytemp[0]=yin[0]+dx*k1[0]/2 
                yytemp[1]=yin[1]+dx*k1[1]/2 
        


                k2=self.derivatives((x+dx)/2,yytemp,energy) 
                #we now overwrite a new yytemp 
                yytemp[0]=yin[0]+dx*k2[0]/2 
                yytemp[1]=yin[1]+dx*k2[1]/2 
        
                k3=self.derivatives((x+dx)/2,yytemp,energy) 
                #we now overwrite a new yytemp 
                yytemp[0]=yin[0]+dx*k3[0]
                yytemp[1]=yin[1]+dx*k3[1]
        
                k4=self.derivatives((x+dx),yytemp,energy) 
                #we calculate the final values
                finalphi=yin[0]+(1/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dx
                finalp=yin[1]+(1/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dx
                yout=[finalphi,finalp] #we put the final values in a list and return it
    
                return yout 
        
        def wave_function(self,E):
                '''This fucntion solves the schrodingers equation for a certain potential (function called inside) and certain E. 
                It takes as a parameter the energy of the particle.
                The output is a list of two lists with the positions x and phi of the normalized wave function'''
                
                phi=[0] #initial condition for x=-5A 
                p=[2*10**(-6)] #initial derivative for x=-5A
                #Taking first step from x0 
                x=self.x_list_values[0] 
                previous_list=[phi[0],p[0]] #phi and p we have before the step
                next_list=self.RK4(x,self.dx,previous_list,E) #phi and p we have after the step 
                phi.append(next_list[0])
                p.append(next_list[1]) #we append to the phi list and p list the new values 

                for k in range(1,self.n_steps): #we do the rest of steps 
                        x=self.x_list_values[k]  #define the new x from wich we take the step
                        previous_list=[phi[k],p[k]] #phi and p we have before the step
                        next_list=self.RK4(x,self.dx,previous_list,E) #phi and p after the step 
                        phi.append(next_list[0])
                        p.append(next_list[1]) #we append to the phi list and p list the new values 
                        #now we have a list of 401 elements for phi and p 
        
                #now we have to normalize the wave function 
                #normalization is done by Simpson
                integral=0 
                for k in range(0,self.n_steps+1): 
                        if k==0 or k==self.n_steps: #extrem values 
                                integral=integral +(phi[k]**2)/3
                        elif (k % 2) == 0:  #even number
                                integral=integral+2*(phi[k]**2)/3
                        else: #odd number 
                                integral=integral+4*(phi[k]**2)/3
                integral=integral*self.dx 
                #now we have to normalize 
                phi_norm=[element/math.sqrt(integral) for element in phi] #and we have to wave function normalised 

                return phi_norm #list of lists

        def eigenvalue(self): 
                '''Finds eigenvalues for a potential given with bisection, we d'ont want to left out any zero
                Returns a list of the five eigenvalues founded'''
                epsilon=10**(-4) #precission of the zero 
                E1=0 
                dE=0.1 #precission between eigenvalues
                zeros_counter=0 #counts how many zeros we have found 
                eigen_values=[] #list where all eigenvalues will be placed 
                while zeros_counter<5: #we want to find 5 zeros 
                        E2=E1+dE 
                        #we have to now see if there is a zero between E2 and E1 
                        listsE1=self.wave_function(E1)
                        listsE2=self.wave_function(E2)
                        last_phiE1=listsE1[self.n_steps]
                        last_phiE2=listsE2[self.n_steps]#we have to get the last value in order to search for the zero 
                        #we check now if there is a zero between them 
                        if (last_phiE1*last_phiE2)<0: #theres is a zero 
                                B=E2 
                                A=E1 #so we don't modify E1 and E2 in bisection
                                #we do bisection: 
                                n_max=(math.log((B-A)/epsilon)/math.log(2)) +10000 #compute maximum of iterations 
                                for n in range(0,int(n_max)): 
                                        listsA=self.wave_function(A)
                                        listsB=self.wave_function(B)
                                        last_phiA=listsA[self.n_steps]
                                        last_phiB=listsB[self.n_steps]#we have to get the last value in order to search for the zero 
                                        C=(A+B)/2 #take a mid point
                                        listsC=self.wave_function(C)
                                        last_phiC=listsC[self.n_steps]
                                        if (last_phiC*last_phiB)>0:#zero ain't between C an B 
                                                B=C
                                        else: #zero is between c and b 
                                                A=C 
                                        #now we have new A and B, we check if we have enough precission 
                                        dif=abs(B-A) 
                                        if dif < epsilon: 
                                                eigenvalue=C 
                                                break #we exit the loop 
                                #we have foun the zero 
                                eigen_values.append(eigenvalue) 
                                zeros_counter=zeros_counter+1  
                                E1=E2 
                        else: #there is no zero 
                                E1=E2
                return eigen_values

        def plot_energy(self):
                '''This function plots the energy line for a estatic potential'''
                plt.clf()
                fig, ax_phi=plt.subplots()
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                ax_phi.set_ylim((0,1)) #maximum of phi_axis
                
                eigen_values=self.eigenvalue() #we compute the eigen_values 
                E= eigen_values[self.value_n] #we assign Energy 
                


                y2=0
                V_plot=[self.potential(s) for s in self.x_list_values] 
                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_list_values,V_plot, label="V(x)" , color='tab:blue') 
                ax_V.axhline(y=E, color='g', linestyle='-',label="E") #we plot the Energy value too 
                ax_V.fill_between(self.x_list_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                if E<30: ax_V.set_ylim((0,30))
                else: ax_V.set_ylim((0,E+5)) 
                ax_V.legend(loc="upper right")
                plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4]+"   V(x)= "+str(self.potential_type),loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.add_widget(canvas_plot)

        def plot_potential(self): 
                '''This function plots the potential'''
                #clear previous plot 
                plt.clf()


                fig, ax_phi=plt.subplots()
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                ax_phi.set_ylim((0,1)) #maximum of phi_axis

                y2=0
                V_plot=[self.potential(s) for s in self.x_list_values] 
                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_list_values,V_plot, label="V(x)" , color='tab:blue') 
                ax_V.fill_between(self.x_list_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                ax_V.set_ylim((0,30)) 
                ax_V.legend(loc="upper right")
                plt.title("V(x)= "+str(self.potential_type),loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.add_widget(canvas_plot)

        def plot_wave_function(self):
                '''This function plots into the window the wave function.
                It also plots the potential. Previous to plotting it computes the eigenvalues
                and computes the eigenvector. '''
                #clear previous plot 
                plt.clf()
                
                eigen_values=self.eigenvalue() #we compute the eigen_values 
                E= eigen_values[self.value_n] #we assign Energy 
                phi=self.wave_function(E)#we compute the wave_function
                y2=0
                phi_square=[element**2 for element in phi]
                #let's plot:

                V_plot=[self.potential(k) for k in self.x_list_values] #compute potential for every x_position
                fig, ax_phi=plt.subplots() 
                ax_phi.set_xlabel(r"$x(\AA)$")
                ax_phi.set_ylabel(r"$ \phi^2$")
                #we check free partcile with theory to see if we are right 
                #x_4= np.linspace(-5,5,401)
                #y_4=((1/math.sqrt(5))*np.sin(4*math.pi*x_4/10))**2 #plot E3 
                #ax_phi.plot(x_4,y_4,label= "Analytic", color='blue') 
                ax_phi.plot(self.x_list_values,phi_square,label=r"$ \phi^2(x)$" , color='tab:red')

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

                
                
                ax_phi.set_ylim((0,max(phi_square)+0.2)) #maximum of phi_axis= maxim of probability +0.2
                ax_phi.legend(loc="upper left")


                ax_V = ax_phi.twinx() #same x_axis
                ax_V.set_ylabel(r"$V(eV)$")
                ax_V.plot(self.x_list_values,V_plot, label="V(x)" , color='tab:blue')
                ax_V.axhline(y=E, color='g', linestyle='-',label="E") #we plot the Energy value too 
                ax_V.fill_between(self.x_list_values, V_plot,y2, facecolor='blue', alpha=0.3) #paint potential
                if E<30: ax_V.set_ylim((0,30))
                else: ax_V.set_ylim((0,E+5)) 
                ax_V.legend(loc="upper right")
                plt.title("E"+str(self.value_n)+"(eV)="+str(E)[0:4]+"   V(x)= "+str(self.potential_type),loc="right")
                
                canvas_plot=FigureCanvasKivyAgg(plt.gcf())
                self.float_plot.add_widget(canvas_plot)


                     
class GameWindow(Screen):
        pass    

     
Builder.load_file("metodo_tiro_kivy.kv") #d'aquesta manera li podem dir al fitxer kv com volguem

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
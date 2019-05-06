#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:29:25 2019
Manu Canals
"""
###############################################################################
#                               IMPORTS                                       #
###############################################################################
#KIVY'S IMPORTS
from kivy.app import App #The executable class to run de app inherits from App 
from kivy.uix.boxlayout import BoxLayout #The main class need to inherit from 
                #it in order to have acess to the differnt boxes in the window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg #Canvas
                                    #object in Kivy (a Figure should be given)
from kivy.clock import Clock #Tools to manage events in Kivy (used to animate)

#OTHER GENERAL IMPORTS
import numpy as np
from matplotlib.figure import Figure #This figure is the tipical one from 
#matplotlib and is the one we shall 'adapt' to kivy using FigureCanvasKivyAgg

###############################################################################
#                             MAIN CLASS                                      #
###############################################################################

class QMeasures(BoxLayout):
    """ 
    Main class. The one passed to the executable class. It has acces to the 
    differents parts of the app layout (since it inherits from BoxLayout). 
    """
    def __init__(self, **kwargs):
        super(QMeasures, self).__init__(**kwargs) #Runs also the superclass
                                                #BoxLayout's __init__ function
                                                
        # -------------------- SOME DEFINITIONS -------------------------------
        #We shall give initial values but these varibles will be changed by 
        #the values given in the app, when implemented this procedure.
        
                                ###### UNITS ######
        
        #time: fs
        #energy: eV
        #position: Å
        #wave function: Å**-2
        
                            ###### DISCRETIZATION ######
        self.a = -10.   
        self.b = 10.        
        self.N = 300
        self.deltax = (self.b - self.a)/float(self.N)
        self.mesh = np.linspace(self.a, self.b, self.N+1)
        
                        ###### Particle's properties ######
                            
        self.hbar = 0.6582   #In these units
        
        self.m_elec = 0.1316 #Its the m factor explained in eigenparam function
        self.m = self.m_elec #The name 'm' is the one used inside of eigenparam
        
                    ###### Double well potential's properties ######
        self.pot_mu = 0.
        self.pot_sigma = 1.
        self.pot_k = 5.
        #Array with the actual potential values
        self.potential = np.array([self.poten(x) for x in self.mesh])
                
                    ###### Eigenvalues and eigenvectors ######
        self.evals = np.zeros(self.N + 1) #Row (N+1) vector (due to eigh func)
        self.evect = np.zeros((self.N + 1, self.N + 1)) #(N+1)x(N+1) matrix
        
                            ###### Wave functions ######
                            
        #Initial (gaussian parameters)
        self.p0 = 0.    
        self.sigma0 = 1.
        self.mu0 = -2.
        #Related object
        self.psi0 = np.zeros(self.N + 1)    #Value of the initial wave function
        self.comp0 = np.zeros(self.N + 1)   #Its components. They are row vects
        #First creation of the wave function
        self.psi0_initiate()  
        #Evolved
        self.psiev = np.zeros(self.N + 1)
        self.compev = np.zeros(self.N + 1)
        
                                ###### Time steps ######
                                
        #Reset time after measuring (updated after each measure)
        self.rst_time = 0.
        
                                 ###### Measure ######
                                 
        #'Sigma' of dirac's delta after measuring
        self.dirac_sigma = 0.4
        
                                ###### TEXT LABEL ######
                                
        #Updating the text in the kivy's labels (these variables come from the 
        #kivy file).
        self.label1.text = 'New mu0:    ' + '%.1f' % self.mu0 + '\n'\
            'Prob.:    ' + '-'
            
                            ###### UNDERLYING POTENTIAL ######
                            
        self.up_e_walls = 100.
        self.up_expo = 4
        self.rtn_point = 6.

        # ------------ FIRST RUN OF EIGENPARAM AND COMP -----------------------
        
    
        self.eigenparam()
        self.comp()
        

        #Compute and show energy expected value
        energy = np.sum(np.abs(self.comp0)**2 * self.evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy
        
        # ---------------------- PLOT OBJECT ----------------------------------
        """
        Includes the creation of all the plotted objects (legend, axis, ...)
        but the updated data. This will happen in the plotting function. Here 
        we should assign the canvas (created with FigureCanvasKivyAgg) to the 
        'box' where it will be plotted in the app with:
        self.box/panel_id.add_widget(self.FigureCanvasKivyAggs_name)
        """
        
        #Canvas
        self.main_fig = Figure()    #Main ploting object where we will do two 
        #differents plottings ('twins'), one for psi and one for the potential.
        #111 specifies one on top of the other
        #Pot plot
        self.pot_twin = self.main_fig.add_subplot(111)
        self.pot_twin.axis([self.a, self.b, 0, 60])
        self.pot_twin.set_xlabel('x [$\AA$]')
        self.pot_twin.set_ylabel('Potential [eV]')
        #Psi plot
        self.psi_twin = self.pot_twin.twinx()
        self.psi_twin.axis([self.a, self.b, 0, 1])
        self.psi_twin.set_ylabel('Probability [$\AA^{-1}$]')
        
        self.main_canvas = FigureCanvasKivyAgg(self.main_fig)
        
        #Data
        self.pot_data, = self.pot_twin.plot(self.mesh, self.potential)
        self.psi_data, = self.psi_twin.plot(self.mesh, self.psi0)
        
        #Canvas object in kivy file
        self.box1.add_widget(self.main_canvas)

        # ------------------------ CLOCK --------------------------------------
        """
        Here all animation will be happening. The plotting function definied in
        the proper section will be called several times per second. It may
        include a pause or play control (the funciton itself or here in clock)
        """
        Clock.schedule_interval(self.plotpsiev, 1/30.)
      
        
    #========================== FUNCTIONS =====================================
        
    #___________________"OBJECT KIND OF" FUNCTION _____________________________
    
    #POTENTIAL: trivially enough, the potential of the problem. Its parameters
    #will be obtained from the 'objects' in the app (in the layout) such as 
    #sliders or buttons. This information is passed from the other file .kv 
    #with the identification names of each of the objects there
    def gaussian(self, x):
        f = 1./np.sqrt(2*np.pi*self.pot_sigma**2)\
                            *np.exp(-(x-self.pot_mu)**2/(2.*self.pot_sigma**2))
        return f

    def harmonic(self, x):
        V = 0.5*self.pot_k*x**2
        return V
    
    def poten(self, x):
        pot = (50*self.gaussian(x) + self.harmonic(x))  #Potential
        #factor 20 introduced to minimize the influence of the walls, since a 
        #larger potential keeps the main structure of the wave centered.
        return pot
    
    #PSI0: initial wavefunciton. Later on this will be defined, meaning the 
    #initial conditions by the user or the game itself.
    def psi0_initiate(self):
        """
        Creates the initial wave function, a gaussian packet in general. The 
        output's shape is the same as mesh.
        -- This could be also done by defining a function that has only the
        function itself of a gaussian packet and generating psi0 by list 
        comprehention calling this shape function on mesh.
        """        
        self.psi0 = np.sqrt((1./(np.sqrt(2*np.pi)*self.sigma0))* \
                    np.exp(-(self.mesh-self.mu0)**2/(2.*self.sigma0**2)))* \
                    np.exp(complex(0,-1)*self.p0*self.mesh)
    
    #_______________"FUNCTIONAL" FUNCTION (compute something)__________________
    
    #PLOTPSIEV: the one called several times. It includes the calculation of 
    #the new data on the given time (psi(t)), using the given eigenstuff
    #and components (already done by other functions). Also does the 
    #data updating and drawing
    def plotpsiev(self, dt):
        """
        Function to be called in the animation loop (clock.schedule_interval),
        it has to have dt as an argument. Here first is computed psi(t), then
        the data is updated in psi_data and finally it draws on the canvas. 
        This parameter dt is not the actual time, its only the real time 
        interval between calls. The time is obtained with Clock.get_time() 
        """
        #This t is the time from the beggining of the evolution, when psi(t)
        #was psi0, so everytime we change psiev to psi0 again (measuring), 
        #we have to reset the time.
        t = Clock.get_time() - self.rst_time
        
        #COMPUTE PSIEV(t). We do it with two steps.
        #Column vector containing the product between component and 
        #exponential factor. We do it via a list comprehention of one-element
        #lists (this generates a columns vector) containing comp*exp() each .
        compexp = np.matrix([\
        [compval * np.exp(np.complex(0.,-1.) * eigenval * t / 10 * self.hbar)]\
                          for compval, eigenval in zip(self.comp0, self.evals)\
                                                                             ])
        #Psi(t)
        col_psiev = self.evect * compexp #Matrix product
        
        #UPDATE DATA. Since col_psiev is a column vector (N+1,1) and we need to 
        #pass a list, we take out every element individually with a loop. We
        #build the acutal psiev object. Be careful, col_psiev is a matrix and 
        #thus it has an 'extra' index for taking out the value than a normal
        #array would have.
        self.psiev = np.array([psielem[0,0] for psielem in col_psiev])
        self.psi_data.set_data(self.mesh, np.abs(self.psiev)**2)
        
        #Draw
        self.main_canvas.draw()  
    
    
    #EIGENPARAM: Computes the eigenvalues and eigenvectors of the hamiltonian
    #with a certain potential
    def eigenparam(self):
        """
        Compute a vector with the eigenvalues(eV) and another with the 
        eigenvectors ((Aº)**-1/2) of the quantum hamiltonian with potential 
        [poten(eV)] (each column is an eigenvector with [N]+1 components). 
        
           H · phi = E · phi  ;  H = -(1/2m)(d**2/dx**2) + poten(x)
                    m := mass / hbar**2  [(eV·Aº**2)**-1]
    
        
        It solves the 1D time-independent Schrödinger equation for the given 
        potential (callable poten(x[eV])) inside of a box [a(Aº), b(Aº)], with 
        [N] intervals. 
        """
        #Dividing the ab segment in N intervals leave us with a (N+1)x(N+1) 
        #hamiltonian, where indices 0 and N correspond to the potentials 
        #barriers. The hamiltonian operator has 3 non-zero diagonals (the main 
        #diagonal, and the ones next to it), with the following elements.
           
        H = np.zeros((self.N+1,self.N+1))
        
        H[0,0] = 1./(self.m*self.deltax**2) + 1000000000#(eV)
        H[self.N,self.N] = 1./(self.m*self.deltax**2) + 1000000000
        H[1,0] = -1./(2.*self.m*self.deltax**2)
        H[self.N-1,self.N] = -1./(2.*self.m*self.deltax**2)
        
        for i in range(1, self.N):
            x = self.a+self.deltax*i
            #In order to smoothen the walls, we add a underlying potential:
            undrl_pot=np.abs(self.up_e_walls*(x/self.rtn_point)**self.up_expo)
            undrl_pot = 0.
            H[i,i] =1./(self.m*self.deltax**2)+self.poten(x)+undrl_pot
            H[i-1,i] = -1./(2.*self.m*self.deltax**2)
            H[i+1,i] = -1./(2.*self.m*self.deltax**2)
            
        #Diagonalizing H we'll get the eigenvalues and eigenvectors (evect 
        #in columns).
        self.evals, self.evect = np.linalg.eigh(H)
        
        #Normalization. Used trapezoids method formula and that sum(evect**2)=1
        factor = np.zeros((self.N + 1)) #Evect multiplied by 1/sqrt(factor)
        for col in range(self.N+1):
            factor[col]=self.deltax*\
                                (1.-self.evect[0,col]/2.-self.evect[-1,col]/2.)
    
        #Normalized vectors
        for col in range(self.N+1):
            self.evect[:,col] *= 1/np.sqrt(factor[col])
            
            
            
            
            
    
    #COMP: given the initial wave function and the eigenbasis (from eigenparam)
    #gives the components of that wave functioni if that basis.        
    def comp(self, do_psi = False):
        """
        Generates the initial wave function's components on the eigenbasis if 
        d0_psi = False (stored on comp0). If d0_psi = True computes psiev 
        components (stored in compev).
        """
        #This method for computing the components could be improved i think.
        if do_psi == True:
            #Compute psi's components
            compoev = []
            for ev in np.transpose(self.evect): #ev: each basis vector
                #for each ev, we integrate ev conjugate * psi = integrand 
                integrand =[np.conjugate(v)*p for v, p in zip(ev, self.psiev)] 
                #integrations by trapezoids
                compoev.append(self.deltax* \
                             (sum(integrand)-integrand[0]/2.-integrand[-1]/.2))
            self.compev = np.array(compoev) 
        else:
            compo = []
            for ev in np.transpose(self.evect): #ev: each basis vector
                #for each ev, we integrate ev conjugate * psi = integrand 
                integrand =[np.conjugate(v)*p for v, p in zip(ev, self.psi0)] 
                #integrations by trapezoids
                compo.append(self.deltax* \
                             (sum(integrand)-integrand[0]/2.-integrand[-1]/.2))
            self.comp0 = np.array(compo) 
        
    #MEASURE: triggered from the kivy file. It takes the actual psiev(t), 
    #generates the probability distribution and initiates the psi0 again with a
    #new mu. Finally, calls the comp()
    def measure(self):
        """
        Triggered from the kivy file. It takes the actual psiev(t), generates 
        the probability distribution and picks the new value for mu. A new 
        initial wave function is created with a small sigma witch represents
        the delta post measure. The time needs to be reset to zero after 
        measuring. Finally, calls comp() and now plotpsiev() has everything it
        needs to continue plotting.
        """
        #prob = self.deltax*np.abs(self.psiev)**2
        prob = np.abs(self.psiev)**2
        #We have to divide by a constant so the sum gives 1 (1/sum is the ct.)
        #and it doesnt affect the probabilty distribution since all relative
        #probabilites remain the same
        self.mu0 = np.random.choice(self.mesh, p=prob/sum(prob))
        self.sigma0 = self.dirac_sigma 
        self.psi0_initiate()
        self.comp()
        self.rst_time = Clock.get_time()
        #Updating the text in the kivy's labels (these variables come from the 
        #kivy file).
        self.label1.text = 'New mu0:    ' + '%.1f' % self.mu0 + '\n' \
            'Prob.:    '  + '%.3f' \
            % (prob[int((self.mu0 - self.a)/self.deltax)] * 100 * self.deltax)\
            + '%'
        #Compute and show energy expected value
        energy = np.sum(np.abs(self.comp0)**2 * self.evals)
        self.label2.text = '<E> =  \n' + '%.3f' % energy
     

class QMeasuresApp(App):
    """
    This class is the one executed when executing the file. Includes the main 
    class with the app's content. Due to kivy's syntax, the part before App in
    this class name must be also the name of the files .py and .kv 
    """
    def build(self):
        self.title = 'Quantic Measures'
        return QMeasures()

if __name__ == '__main__':
    QMeasuresApp().run()
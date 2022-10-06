import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft
import math as mt
from scipy import integrate as inte
import kivy
from kivy.lang import Builder
from kivy.app import App
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import ScreenManager,Screen
from kivy.uix.floatlayout import FloatLayout
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
import matplotlib.pyplot as plt
from kivy.clock import Clock


hbar=1
m=1
pi=mt.pi
j=complex(0,1)
#______________________________________________________________________________
#Transformed domain points:
#given an initial space domain points, the points in the transformed domain 
#space are calculated.

#arguments:
#x: N initial points x of the initial space
#N: number of points used
#dx:space between points in the initial domain

#outputs:
#k:N corresponding points in the transformed space
#dk:distance between points in the transformed domain

def trans_space(N,x,dx):
    #critic freq.
    kc=1/(2*dx)
    #space between point in transform space
    dk=1/(N*dx)
    #points in transform space
    k=np.linspace(-kc,kc-dk,N)
    return k,dk




#______________________________________________________________________________
#FT routine:
#given a initial function (f) and the corresponding space points (x) returns the
#FT(f) and the corresponding points in the transformed space domain (k).

#arguments:
#gx: function g(x) sampled with N points
#dx:space between points in the initial domain

#outputs:
#FTgk: FT(g(x)) sampled with N points


def fourier_trans(gx,dx):
    #calculate FT
    FTgk=np.fft.fftshift(fft(gx))*dx
    return FTgk





#______________________________________________________________________________
#IFT routine
#given a initial function (f) and the corresponding space points (x) returns the
#IFT(f) and the corresponding points in the transformed space domain (k).
#arguments:
#gk: function g(k) sampled with N points
#dx:not use dk, if you want to use dk you also have to multiply by N, but i prefer
#to not use N because it is more arguments than the strictly necessary

#outputs:
#IFTgx: IFT(g(k)) sampled with N points

def inv_fourier_trans(gk,dx):
    IFTgk=ifft(np.fft.ifftshift(gk))*1/dx
    return IFTgk




#________________________________________________________________________________
#Psi 1 in the x space:
#calculate the first psi of the method in the x domain-

#inputs:
#Vx:vector of N points with the values of the external potential in x space
#g:couple constant of the non linear term
#psi0x: vector with n points corresponding to the initial wavefunction at time t
#in the initial x space
#dt: interval betwen t+dt and t

#outputs:
#psi1x: new wavefunction in the x space




def psi_onex(Vx,g,psi0x,dt):
    #prepare the value complex j:
    j=complex(0,1)
    #exponent of the exponential
    expon=(-j*dt/(hbar*2.0))*(Vx+g*np.abs(psi0x)*np.abs(psi0x))
    #calculate the e^expon
    expterm=np.exp(expon)
    #aply the exponential to psi0x to obtein the new wave function
    psi1x=expterm*psi0x
    return psi1x


#________________________________________________________________________________

#Important to know:
#Important to be aware here that our k is not really the wave number, where  
#wavenumber(wavenum) corresponds to  hbar*wavenum=p. But here k*2pi=wavenum, 
#this is because our k would represent more the invere longth wave


#psi2p:
#psi 2 in the transformed domain k

#inputs:
#k:vector of N points with the values of the k points in the k domain
#psi1p: vector with n points corresponding to the wavefunction 1 in the p domain
#dt: interval betwen t+dt and t

#outputs:
#psi2p:wave function psi 2 in the p domain

def psi_twop(k,psi1p,dt):
    #prepare pi and j values
    pi=mt.pi
    j=complex(0,1)
    #calculate the momentum vector
    p=2*pi*k*hbar
    #calculate the exponent value
    expon=-j*dt*p*p/(2.0*hbar*m)
    #calculate the exponential
    expterm=np.exp(expon)
    #aply the exponential to the initial wave function
    psi2p=expterm*psi1p
    return psi2p





#______________________________________________________________________________
#psi(t+dt)x: calculate function psi in the time t+dt in the x space

#inputs:
#Vx:vector of N points with the values of the external potential in x space
#g:couple constant of the non linear term
#psi0x: vector with n points corresponding to psi0 in the x space
#psi1x: vector with n points corresponding to psi1 in the x space
#psi2x: vector with n points corresponding to psi2 in the x space


#outputs:
#psifx:function psi in the time t+dt in the x space

def psi_fx(Vx,g,psi0x,psi1x,psi2x,dt):
    #prepare the value complex j:
    j=complex(0,1)
    #exponent of the exponential
    mod0=psi0x
    mod1=psi1x
    mod2=psi2x
    mod=np.abs(mod0-mod1+mod2)**2
    expon=(-j*dt/(2.0*hbar))*(Vx+g*mod)
    #calculate the e^expon
    expterm=np.exp(expon)
    #aply the exponential to psi0x to obtein the new wave function
    psifx=expterm*psi2x
    return psifx


#_____________________________________________________________________________
#SPLIT_STEP method:
#from an initial wave function psi(t,x) the split step method wil return the 
#wave function psi(t+dt,x)

#input:
#N:number of points will be used
#x: vector of N points, corresponding to the space values x where psi is calculated
#dx:space between the x points
#psi0:N point vector with the corresponding values of the wave funct. in some x
#dt: interval of time betwen t and t+dt
#Vx:values of the external potential
#g:coupled parameter
#this two are calculated outside split step because there would not
#be changed so it is more optimal only to calculate them once outside the split
#step routine
#k:points in k domain
#dk:interval between k points
#output:
#psifx:the wave function in time dt+t 

def split_step(N,x,dx,psi0x,dt,Vx,g,k,dk):
    #calculate the psi1x
    psi1x=psi_onex(Vx,g,psi0x,dt)
    #Transform psi1x to the momentum space with FT
    psi1p=fourier_trans(psi1x,dx)
    #now calculate the psi2p:
    psi2p=psi_twop(k,psi1p,dt)
    #get psi2x with the inverse fourier transform
    psi2x=inv_fourier_trans(psi2p,dx)
    #now get the final psifx:
    psifx=psi_fx(Vx,g,psi0x,psi1x,psi2x,dt)
    return psifx






#Parameter study of GPE in harmonic potential trap.
##______________________________________________________________________________________

#The first step is to find the ground state of the GPE equation. So using a Gaussian 
#as the initial wave function, the imaginary time evolution method is used to get 
#the ground state

    

#Spatial parameters
#Number of spatial points:
N=2**10
#Array of spatial points
x=np.linspace(-12,12,N)
#length and space between spatial points
L=x[N-1]-(x[0])
dx=dx=L/N

#Vx: Harmonic potential but in oscillator units. 1/2 x**2
Vx=(x*x)*(1/2.0)

#Calculate the k points and the separation between points in the transformed space:
k,dk=trans_space(N,x,dx)

#Couple constant of GPE
g=400









#animate function to do the gif with matplotlib





#class and aplication code:
    
    

class WindowManager(ScreenManager):
    pass


class FirstWindow(Screen):    
#When initialize, start the screen with the plot of the initial function:


    def plot_initial(self):
        
        
        #initial function:
        psi0x=np.exp(-2*x*x)
        #normalize the function
        probx=np.abs(psi0x)**2
        Ix1=inte.simps(probx,x,dx)
        psi0x=psi0x/(mt.sqrt(Ix1))
        #plot the initial probability
        probx=np.abs(psi0x)**2
        
        self.init_func=np.copy(probx)
        self.count1=0
        #make the plot
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(-12, 12)
        ax.set_ylim(0, 1.5)
        plt.xlabel(r'$\bar{x} \:\: (ad.) $')
        plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
        line.set_data(x, probx)
        
        #put it in kivy
        impsit0=self.ids.impsit0
        impsit0.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))


# calculate the ground state
    def ground_state(self):
        #set the parameters for the time evolution
        Nt=1000
        tau=-j*np.linspace(0,10,Nt)
        dtau=tau[1]-tau[0]
        
        #initial function
        psi0x=np.exp(-2*x*x)
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0j (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 1)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)
            count=count+1
            psi0x=np.copy(psifx)
        self.func1=y
        self.data1=tdata
        
        
    
    
        
    def clockkiv1(self):
        
        
        def my_callback(dt):
            count1=self.count1
            y=self.func1
            tdata=self.data1
            f1=np.abs(y[count1])**2
            tit1=tdata[count1]
            
            
            if count1==140:
                self.count1=0
                self.ids.next_btn.disabled=False
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=np.amax(f1)+0.02
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,f1)
        
            #put it in kivy
            impsit0=self.ids.impsit0
            impsit0.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count1+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 10.)
        
    def deleteFirstwindow(self):
        
        self.count1=None
        self.func1=None
        self.data1=None
        self.init_func=None
        
        del self.count1
        del self.func1
        del self.data1
        del self.init_func
        
        plt.close()
        
        
class SecondWindow(Screen):
    pass
            

class Node1Window(Screen):
    
    
    
    def slide_n1_x1(self,*args):
        self.node1x1=args[1]
        self.slide_text_n1x1.text=str(round(args[1],2))
        self.ids.imag_btn_n1.disabled=False
        
    
    def ground_state_n1(self):
        #set the parameters for the time evolution
        Nt=1000
        tau=-j*np.linspace(0,10,Nt)
        dtau=tau[1]-tau[0]
        
        #initial function
        psi0x=np.exp(-2*x*x)
        
        for i in tau:
            
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            psifx=psifx/(mt.sqrt(Ix1))
            psi0x=np.copy(psifx)
            
        self.ground_n1=np.copy(psi0x)
    
    def imagtime_n1(self):
        f0=self.ground_n1
        x1=self.node1x1
        psi0x=f0*np.tanh(x-x1)
        #set the parameters for the time evolution
        Nt=5000
        tau=-j*np.linspace(0,0.2,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0j (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 50)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)

            count=count+1
            psi0x=np.copy(psifx)
            
            
        self.func_n1=y
        self.data_n1=tdata
        self.stable=np.copy(psi0x)
    
    
    def timeevol_n1(self):
        
        f0=self.stable
        
        
        psi0x=np.copy(f0)
        #set the parameters for the time evolution
        Nt=100000
        tau=np.linspace(0,20,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0 (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 500)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)
            count=count+1
            psi0x=np.copy(psifx)

        self.func_n1_v2=y
        self.data_n1_v2=tdata
        
    
    
    
    
    
    def clock_n1_v1(self):
        
        self.count_n1=0
        def my_callback(dt):
            count=self.count_n1
            y=self.func_n1
            tdata=self.data_n1
            prob=np.abs(y[count])**2
            tit1=tdata[count]

            
            
            if count==80:
                self.ids.evol_btn_n1.disabled=False
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=np.amax(prob)+0.02
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n1=self.ids.impsit_n1
            impsit_n1.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n1+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
    
    def clock_n1_v2(self):
        
        self.count_n1=0
        
        
        def my_callback(dt):
            count=self.count_n1
            y=self.func_n1_v2
            tdata=self.data_n1_v2
            prob=np.abs(y[count])**2
            tit1=tdata[count]
            
            
            
            if count==200:
                self.ids.evol_btn_n1.disabled=True
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=0.2
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n1=self.ids.impsit_n1
            impsit_n1.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n1+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
    
    def deletenode1(self):
        self.count_n1=None
        self.func_n1=None
        self.data_n1=None
        self.func_n1_v2=None
        self.data_n1_v2=None
        self.stable=None
        self.ground_n1=None
        self.node1x1=None
        
        del self.count_n1
        del self.func_n1
        del self.data_n1
        del self.func_n1_v2
        del self.data_n1_v2
        del self.stable
        del self.ground_n1
        del self.node1x1
        self.ids.imag_btn_n1.disabled=True
        self.ids.evol_btn_n1.disabled=True

class Node2Window(Screen):
    def slide_n2_x1(self,*args):
        self.node2x1=args[1]
        self.slide_text_n2x1.text=str(round(args[1],2))
        
    def slide_n2_x2(self,*args):
        self.node2x2=args[1]
        self.slide_text_n2x2.text=str(round(args[1],2))
        self.ids.imag_btn_n2.disabled=False
        
    
    def ground_state_n2(self):
        #set the parameters for the time evolution
        Nt=1000
        tau=-j*np.linspace(0,10,Nt)
        dtau=tau[1]-tau[0]
        
        #initial function
        psi0x=np.exp(-2*x*x)
        
        for i in tau:
            
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            psifx=psifx/(mt.sqrt(Ix1))
            psi0x=np.copy(psifx)
            
        self.ground_n2=np.copy(psi0x)
    
    def imagtime_n2(self):
        f0=self.ground_n2
        x1=self.node2x1
        x2=self.node2x2
        psi0x=f0*np.tanh(x-x1)*np.tanh(x-x2)
        #set the parameters for the time evolution
        Nt=5000
        tau=-j*np.linspace(0,0.2,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0j (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 50)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)

            count=count+1
            psi0x=np.copy(psifx)
            
            
        self.func_n2=y
        self.data_n2=tdata
        self.stable=np.copy(psi0x)
    
    
    def timeevol_n2(self):
        
        f0=self.stable
        
        
        psi0x=np.copy(f0)
        #set the parameters for the time evolution
        Nt=100000
        tau=np.linspace(0,20,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0 (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 500)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)
            count=count+1
            psi0x=np.copy(psifx)

        self.func_n2_v2=y
        self.data_n2_v2=tdata
        
    
    
    
    
    
    def clock_n2_v1(self):
        
        self.count_n2=0
        def my_callback(dt):
            count=self.count_n2
            y=self.func_n2
            tdata=self.data_n2
            prob=np.abs(y[count])**2
            tit1=tdata[count]

            
            
            if count==80:
                self.ids.evol_btn_n2.disabled=False
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=np.amax(prob)+0.02
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n2=self.ids.impsit_n2
            impsit_n2.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n2+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
    def clock_n2_v2(self):
        
        self.count_n2=0
        
        
        def my_callback(dt):
            count=self.count_n2
            y=self.func_n2_v2
            tdata=self.data_n2_v2
            prob=np.abs(y[count])**2
            tit1=tdata[count]
            
            
            
            if count==200:
                self.ids.evol_btn_n2.disabled=True
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=0.2
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n2=self.ids.impsit_n2
            impsit_n2.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n2+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
        
    def deletenode2(self):
        self.count_n2=None
        self.func_n2=None
        self.data_n2=None
        self.func_n2_v2=None
        self.data_n2_v2=None
        self.stable=None
        self.ground_n2=None
        self.node2x1=None
        self.node2x2=None
        
        del self.count_n2
        del self.func_n2
        del self.data_n2
        del self.func_n2_v2
        del self.data_n2_v2
        del self.stable
        del self.ground_n2
        del self.node2x1
        del self.node2x2
        self.ids.imag_btn_n2.disabled=True
        self.ids.evol_btn_n2.disabled=True

class Node3Window(Screen):
    
    def slide_n3_x1(self,*args):
        self.node3x1=args[1]
        self.slide_text_n3x1.text=str(round(args[1],2))
        
    def slide_n3_x2(self,*args):
        self.node3x2=args[1]
        self.slide_text_n3x2.text=str(round(args[1],2))
    
    def slide_n3_x3(self,*args):
        self.node3x3=args[1]
        self.slide_text_n3x3.text=str(round(args[1],2))
        self.ids.imag_btn_n3.disabled=False
        
    
    def ground_state_n3(self):
        #set the parameters for the time evolution
        Nt=1000
        tau=-j*np.linspace(0,10,Nt)
        dtau=tau[1]-tau[0]
        
        #initial function
        psi0x=np.exp(-2*x*x)
        
        for i in tau:
            
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            psifx=psifx/(mt.sqrt(Ix1))
            psi0x=np.copy(psifx)
            
        self.ground_n3=np.copy(psi0x)
    
    def imagtime_n3(self):
        f0=self.ground_n3
        x1=self.node3x1
        x2=self.node3x2
        x3=self.node3x3
        psi0x=f0*np.tanh(x-x1)*np.tanh(x-x2)*np.tanh(x-x3)
        #set the parameters for the time evolution
        Nt=5000
        tau=-j*np.linspace(0,0.2,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0j (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 50)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)

            count=count+1
            psi0x=np.copy(psifx)
            
            
        self.func_n3=y
        self.data_n3=tdata
        self.stable=np.copy(psi0x)
    
    
    def timeevol_n3(self):
        
        f0=self.stable
        
        
        psi0x=np.copy(f0)
        #set the parameters for the time evolution
        Nt=100000
        tau=np.linspace(0,20,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0 (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 500)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)
            count=count+1
            psi0x=np.copy(psifx)

        self.func_n3_v2=y
        self.data_n3_v2=tdata
        
    
    
    
    
    
    def clock_n3_v1(self):
        
        self.count_n3=0
        def my_callback(dt):
            count=self.count_n3
            y=self.func_n3
            tdata=self.data_n3
            prob=np.abs(y[count])**2
            tit1=tdata[count]

            
            
            if count==80:
                self.ids.evol_btn_n3.disabled=False
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=np.amax(prob)+0.02
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n3=self.ids.impsit_n3
            impsit_n3.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n3+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
        
    def clock_n3_v2(self):
        
        self.count_n3=0
        
        
        def my_callback(dt):
            count=self.count_n3
            y=self.func_n3_v2
            tdata=self.data_n3_v2
            prob=np.abs(y[count])**2
            tit1=tdata[count]
            
            
            
            if count==200:
                self.ids.evol_btn_n3.disabled=True
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=0.2
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n3=self.ids.impsit_n3
            impsit_n3.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n3+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        

    def deletenode3(self):
        self.count_n3=None
        self.func_n3=None
        self.data_n3=None
        self.func_n3_v2=None
        self.data_n3_v2=None
        self.stable=None
        self.ground_n3=None
        self.node3x1=None
        self.node3x2=None
        self.node3x3=None
        
        
        del self.count_n3
        del self.func_n3
        del self.data_n3
        del self.func_n3_v2
        del self.data_n3_v2
        del self.stable
        del self.ground_n3
        del self.node3x1
        del self.node3x2
        del self.node3x3
        
        self.ids.imag_btn_n3.disabled=True
        self.ids.evol_btn_n3.disabled=True

class Node4Window(Screen):
    def slide_n4_x1(self,*args):
        self.node4x1=args[1]
        self.slide_text_n4x1.text=str(round(args[1],2))
        
    def slide_n4_x2(self,*args):
        self.node4x2=args[1]
        self.slide_text_n4x2.text=str(round(args[1],2))
        
    
    def slide_n4_x3(self,*args):
        self.node4x3=args[1]
        self.slide_text_n4x3.text=str(round(args[1],2))
    
    def slide_n4_x4(self,*args):
        self.node4x4=args[1]
        self.slide_text_n4x4.text=str(round(args[1],2))
        self.ids.imag_btn_n4.disabled=False
        
    
    def ground_state_n4(self):
        #set the parameters for the time evolution
        Nt=1000
        tau=-j*np.linspace(0,10,Nt)
        dtau=tau[1]-tau[0]
        
        #initial function
        psi0x=np.exp(-2*x*x)
        
        for i in tau:
            
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            psifx=psifx/(mt.sqrt(Ix1))
            psi0x=np.copy(psifx)
            
        self.ground_n4=np.copy(psi0x)
    
    def imagtime_n4(self):
        f0=self.ground_n4
        x1=self.node4x1
        x2=self.node4x2
        x3=self.node4x3
        x4=self.node4x4
        psi0x=f0*np.tanh(x-x1)*np.tanh(x-x2)*np.tanh(x-x3)*np.tanh(x-x4)
        #set the parameters for the time evolution
        Nt=5000
        tau=-j*np.linspace(0,0.2,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0j (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 50)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)

            count=count+1
            psi0x=np.copy(psifx)
            
            
        self.func_n4=y
        self.data_n4=tdata
        self.stable=np.copy(psi0x)
    
    
    def timeevol_n4(self):
        
        f0=self.stable
        
        
        psi0x=np.copy(f0)
        #set the parameters for the time evolution
        Nt=100000
        tau=np.linspace(0,20,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0 (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 500)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)
            count=count+1
            psi0x=np.copy(psifx)

        self.func_n4_v2=y
        self.data_n4_v2=tdata
        
    
    
    
    
    
    def clock_n4_v1(self):
        
        self.count_n4=0
        def my_callback(dt):
            count=self.count_n4
            y=self.func_n4
            tdata=self.data_n4
            prob=np.abs(y[count])**2
            tit1=tdata[count]

            
            
            if count==80:
                self.ids.evol_btn_n4.disabled=False
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=np.amax(prob)+0.02
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n4=self.ids.impsit_n4
            impsit_n4.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n4+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
    
    def clock_n4_v2(self):
        
        self.count_n4=0
        
        
        def my_callback(dt):
            count=self.count_n4
            y=self.func_n4_v2
            tdata=self.data_n4_v2
            prob=np.abs(y[count])**2
            tit1=tdata[count]
            
            
            
            if count==200:
                self.ids.evol_btn_n4.disabled=True
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=0.2
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n4=self.ids.impsit_n4
            impsit_n4.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n4+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
        
    
    def deletenode4(self):
        self.count_n4=None
        self.func_n4=None
        self.data_n4=None
        self.func_n4_v2=None
        self.data_n4_v2=None
        self.stable=None
        self.ground_n4=None
        self.node4x1=None
        self.node4x2=None
        self.node4x3=None
        self.node4x4=None
        
        del self.count_n4
        del self.func_n4
        del self.data_n4
        del self.func_n4_v2
        del self.data_n4_v2
        del self.stable
        del self.ground_n4
        del self.node4x1
        del self.node4x2
        del self.node4x3
        del self.node4x4
        
        self.ids.imag_btn_n4.disabled=True
        self.ids.evol_btn_n4.disabled=True



class Node5Window(Screen):
    def slide_n5_x1(self,*args):
        self.node5x1=args[1]
        self.slide_text_n5x1.text=str(round(args[1],2))
        
    def slide_n5_x2(self,*args):
        self.node5x2=args[1]
        self.slide_text_n5x2.text=str(round(args[1],2))
        
    
    def slide_n5_x3(self,*args):
        self.node5x3=args[1]
        self.slide_text_n5x3.text=str(round(args[1],2))
    
    def slide_n5_x4(self,*args):
        self.node5x4=args[1]
        self.slide_text_n5x4.text=str(round(args[1],2))
        
    def slide_n5_x5(self,*args):
        self.node5x5=args[1]
        self.slide_text_n5x5.text=str(round(args[1],2))
        self.ids.imag_btn_n5.disabled=False
        
    
    def ground_state_n5(self):
        #set the parameters for the time evolution
        Nt=1000
        tau=-j*np.linspace(0,10,Nt)
        dtau=tau[1]-tau[0]
        
        #initial function
        psi0x=np.exp(-2*x*x)
        
        for i in tau:
            
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            psifx=psifx/(mt.sqrt(Ix1))
            psi0x=np.copy(psifx)
            
        self.ground_n5=np.copy(psi0x)
    
    def imagtime_n5(self):
        f0=self.ground_n5
        x1=self.node5x1
        x2=self.node5x2
        x3=self.node5x3
        x4=self.node5x4
        x5=self.node5x5
        psi0x=f0*(np.tanh(x-x1)*np.tanh(x-x2)*np.tanh(x-x3)*np.tanh(x-x4)*
        np.tanh(x-x5))
        #set the parameters for the time evolution
        Nt=5000
        tau=-j*np.linspace(0,0.2,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0j (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 50)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)

            count=count+1
            psi0x=np.copy(psifx)
            
            
        self.func_n5=y
        self.data_n5=tdata
        self.stable=np.copy(psi0x)
    
    
    def timeevol_n5(self):
        
        f0=self.stable
        
        
        psi0x=np.copy(f0)
        #set the parameters for the time evolution
        Nt=100000
        tau=np.linspace(0,20,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0 (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 500)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)
            count=count+1
            psi0x=np.copy(psifx)

        self.func_n5_v2=y
        self.data_n5_v2=tdata
        
    
    
    
    
    
    def clock_n5_v1(self):
        
        self.count_n5=0
        def my_callback(dt):
            count=self.count_n5
            y=self.func_n5
            tdata=self.data_n5
            prob=np.abs(y[count])**2
            tit1=tdata[count]

            
            
            if count==80:
                self.ids.evol_btn_n5.disabled=False
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=np.amax(prob)+0.02
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n5=self.ids.impsit_n5
            impsit_n5.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n5+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
    
    def clock_n5_v2(self):
        
        self.count_n5=0
        
        
        def my_callback(dt):
            count=self.count_n5
            y=self.func_n5_v2
            tdata=self.data_n5_v2
            prob=np.abs(y[count])**2
            tit1=tdata[count]
            
            
            
            if count==200:
                self.ids.evol_btn_n5.disabled=True
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=0.2
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n5=self.ids.impsit_n5
            impsit_n5.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n5+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
        
    def deletenode5(self):
        self.count_n5=None
        self.func_n5=None
        self.data_n5=None
        self.func_n5_v2=None
        self.data_n5_v2=None
        self.stable=None
        self.ground_n5=None
        self.node5x1=None
        self.node5x2=None
        self.node5x3=None
        self.node5x4=None
        self.node5x5=None
        
        del self.count_n5
        del self.func_n5
        del self.data_n5
        del self.func_n5_v2
        del self.data_n5_v2
        del self.stable
        del self.ground_n5
        del self.node5x1
        del self.node5x2
        del self.node5x3
        del self.node5x4
        del self.node5x5
        
        self.ids.imag_btn_n5.disabled=True
        self.ids.evol_btn_n5.disabled=True






class Node6Window(Screen):
    def slide_n6_x1(self,*args):
        self.node6x1=args[1]
        self.slide_text_n6x1.text=str(round(args[1],2))
        
    def slide_n6_x2(self,*args):
        self.node6x2=args[1]
        self.slide_text_n6x2.text=str(round(args[1],2))
        
    
    def slide_n6_x3(self,*args):
        self.node6x3=args[1]
        self.slide_text_n6x3.text=str(round(args[1],2))
    
    def slide_n6_x4(self,*args):
        self.node6x4=args[1]
        self.slide_text_n6x4.text=str(round(args[1],2))
        
    def slide_n6_x5(self,*args):
        self.node6x5=args[1]
        self.slide_text_n6x5.text=str(round(args[1],2))
    
    def slide_n6_x6(self,*args):
        self.node6x6=args[1]
        self.slide_text_n6x6.text=str(round(args[1],2))
        self.ids.imag_btn_n6.disabled=False
        
    
    def ground_state_n6(self):
        #set the parameters for the time evolution
        Nt=1000
        tau=-j*np.linspace(0,10,Nt)
        dtau=tau[1]-tau[0]
        
        #initial function
        psi0x=np.exp(-2*x*x)
        
        for i in tau:
            
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            psifx=psifx/(mt.sqrt(Ix1))
            psi0x=np.copy(psifx)
            
        self.ground_n6=np.copy(psi0x)
    
    def imagtime_n6(self):
        f0=self.ground_n6
        x1=self.node6x1
        x2=self.node6x2
        x3=self.node6x3
        x4=self.node6x4
        x5=self.node6x5
        x6=self.node6x6
        psi0x=f0*(np.tanh(x-x1)*np.tanh(x-x2)*np.tanh(x-x3)*np.tanh(x-x4)*
        np.tanh(x-x5)*np.tanh(x-x6))
        #set the parameters for the time evolution
        Nt=5000
        tau=-j*np.linspace(0,0.2,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0j (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 50)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)

            count=count+1
            psi0x=np.copy(psifx)
            
            
        self.func_n6=y
        self.data_n6=tdata
        self.stable=np.copy(psi0x)
    
    
    def timeevol_n6(self):
        
        f0=self.stable
        
        
        psi0x=np.copy(f0)
        #set the parameters for the time evolution
        Nt=100000
        tau=np.linspace(0,20,Nt)
        dtau=tau[1]-tau[0]
        
        count=0
        
        #save function
        y=[]
        tdata=[]
        y.append(psi0x)
        tdata.append('t: 0 (ad.) ')
        for i in tau:
            #time used for label in plot
            title='t:'+ str(np.round(i+dtau,decimals=3)) + r'$\:\:\: (ad.)$'
            #apply step method with pass dtau
            psifx=split_step(N,x,dx,psi0x,dtau,Vx,g,k,dk)
            probx=np.abs(psifx)**2
            Ix1=inte.simps(probx,x,dx)
            #normalize function
            psifx=psifx/(mt.sqrt(Ix1))
            if (count % 500)==0:
                #From the 5000 points, every 10 save in the y variable. There will
                #   be 500 frames
                tdata.append(title)
                y.append(psifx)
            count=count+1
            psi0x=np.copy(psifx)

        self.func_n6_v2=y
        self.data_n6_v2=tdata
        
    
    
    
    
    
    def clock_n6_v1(self):
        
        self.count_n6=0
        def my_callback(dt):
            count=self.count_n6
            y=self.func_n6
            tdata=self.data_n6
            prob=np.abs(y[count])**2
            tit1=tdata[count]

            
            
            if count==80:
                self.ids.evol_btn_n6.disabled=False
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=np.amax(prob)+0.02
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n6=self.ids.impsit_n6
            impsit_n6.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n6+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
        
    
    def clock_n6_v2(self):
        
        self.count_n6=0
        
        
        def my_callback(dt):
            count=self.count_n6
            y=self.func_n6_v2
            tdata=self.data_n6_v2
            prob=np.abs(y[count])**2
            tit1=tdata[count]
            
            
            
            if count==200:
                self.ids.evol_btn_n6.disabled=True
                return False
            plt.close()
            fig, ax = plt.subplots()
            line, = ax.plot([], [], lw=2)
            maxval=0.2
            ax.set_xlim(-12, 12)
            ax.set_ylim(0,maxval)
            ax.set_title(tit1)
            
            plt.xlabel(r'$\bar{x} \:\: (ad.) $')
            plt.ylabel(r'$\Psi^{2}(\bar{x},\bar{t}) \:\: (ad.)$')
            line.set_data(x,prob)
        
            #put it in kivy
            impsit_n6=self.ids.impsit_n6
            impsit_n6.add_widget(FigureCanvasKivyAgg(fig,pos_hint={"x":0,"top":1}))
            self.count_n6+=1
            
            
        Clock.schedule_interval(my_callback, 1 / 20.)
        plt.close()
        
    
    def deletenode6(self):
        self.count_n6=None
        self.func_n6=None
        self.data_n6=None
        self.func_n6_v2=None
        self.data_n6_v2=None
        self.stable=None
        self.ground_n6=None
        self.node6x1=None
        self.node6x2=None
        self.node6x3=None
        self.node6x4=None
        self.node6x5=None
        self.node6x6=None
        
        del self.count_n6
        del self.func_n6
        del self.data_n6
        del self.func_n6_v2
        del self.data_n6_v2
        del self.stable
        del self.ground_n6
        del self.node6x1
        del self.node6x2
        del self.node6x3
        del self.node6x4
        del self.node6x5
        del self.node6x6
        
        self.ids.imag_btn_n6.disabled=True
        self.ids.evol_btn_n6.disabled=True
        

        
        
kv = Builder.load_file("soliton.kv")      
        
        
        
    

class solitonApp(App):
    def build(self):
        return kv
    
if __name__=='__main__':
    solitonApp().run()

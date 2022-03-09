import math as math
import matplotlib.pyplot as plt


def potential(x): 
        '''Gives the potential'''
        V=math.sin(x)**2

        return V 

def derivatives(x,yin,energy):
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

        dp= (2/h2_me)*(potential(x)-energy)*phi #we compute dphi**2/dx**2 

        dyout=[dphi,dp]
        return dyout #return a list with dphi and dp 


def RK4(x,dx,yin,energy): 
        '''THis function makes one step of a RK4 intergation of an ordinary diferential equation. 
        x i the dependent variable at which we start, dx is the step we take. 
        yin are the values of the function at x: should be a list such as [phi,dphi/dx] 
        , yout(return) are the values at x+dx'''
        
        k1=derivatives(x,yin,energy) #k1 is a list [dphi,p]
        yytemp=[0,0] #we assign random numbers to yytemp in order to have a 2d list
        yytemp[0]=yin[0]+dx*k1[0]/2 
        yytemp[1]=yin[1]+dx*k1[1]/2 
        


        k2=derivatives((x+dx)/2,yytemp,energy) 
        #we now overwrite a new yytemp 
        yytemp[0]=yin[0]+dx*k2[0]/2 
        yytemp[1]=yin[1]+dx*k2[1]/2 
        
        k3=derivatives((x+dx)/2,yytemp,energy) 
        #we now overwrite a new yytemp 
        yytemp[0]=yin[0]+dx*k3[0]
        yytemp[1]=yin[1]+dx*k3[1]
        
        k4=derivatives((x+dx),yytemp,energy) 
        #we calculate the final values
        finalphi=yin[0]+(1/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])*dx
        finalp=yin[1]+(1/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])*dx
        yout=[finalphi,finalp] #we put the final values in a list and return it
    
        return yout 

def wave_function(E):
        '''This fucntion solves the schrodingers equation for a certain potential (function called inside) and certain E. 
        It takes as a parameter the energy of the particle.
        The output is a list of two lists with the positions x and phi of the normalized wave function'''
        n_steps=400 #number of steps taken to integrate the wavefunction 
        L=10 #lenght of the box 
        
        x0=-L/2 
        xf=L/2 #box goes from -5A to 5A 
        x_list=[x0]
        dx=L/n_steps #we define the step we take 
        phi=[0] #initial condition for x=-5A 
        p=[2*10**(-6)] #initial derivative for x=-5A
        #Taking first step from x0 
        x=x0 
        previous_list=[phi[0],p[0]] #phi and p we have before the step
        next_list=RK4(x,dx,previous_list,E) #phi and p we have after the step 
        phi.append(next_list[0])
        p.append(next_list[1]) #we append to the phi list and p list the new values 

        for i in range(1,n_steps): #we do the rest of steps 
                x=x0+dx*i #define the new x from wich we take the step
                x_list.append(x)
                previous_list=[phi[i],p[i]] #phi and p we have before the step
                next_list=RK4(x,dx,previous_list,E) #phi and p after the step 
                phi.append(next_list[0])
                p.append(next_list[1]) #we append to the phi list and p list the new values 
                #now we have a list of 401 elements for phi and p 
        x_list.append(xf) #we have to append last value, loop doesn't get till the last one 
        
        #now we have to normalize the wave function 
        #normalization is done by trapezoids 
        integral=0 
        for i in range(0,n_steps+1): 
                if i==0 or i==n_steps+1: #extrem values 
                        integral=integral +(phi[i]**2)/2
                else: 
                        integral=integral+phi[i]**2
        integral=integral*dx 
        #now we have to normalize 
        phi_norm=[element/math.sqrt(integral) for element in phi] #and we have to wave function normalised 

        return [x_list,phi_norm] #list of lists

def eigenvalue(): 
        '''Finds eigenvalues for a potential given with bisection, we d'ont want to left out any zero'''
        epsilon=10**(-4) #precission of the zero 
        E1=0 
        dE=0.1 #precission between eigenvalues
        zeros_counter=0 #counts how many zeros we have found 
        eigen_values=[] #list where all eigenvalues will be placed 
        while zeros_counter<5: #we want to find 5 zeros 
                E2=E1+dE 
                #we have to now see if there is a zero between E2 and E1 
                listsE1=wave_function(E1)
                listsE2=wave_function(E2)
                last_phiE1=listsE1[1][400]
                last_phiE2=listsE2[1][400]#we have to get the last value in order to search for the zero 
                #we check now if there is a zero between them 
                if (last_phiE1*last_phiE2)<0: #theres is a zero 
                        B=E2 
                        A=E1 #so we don't modify E1 and E2 in bisection
                        #we do bisection: 
                        n_max=(math.log((B-A)/epsilon)/math.log(2)) +1 #compute maximum of iterations 
                        for n in range(0,int(n_max)): 
                                listsA=wave_function(A)
                                listsB=wave_function(B)
                                last_phiA=listsA[1][400]
                                last_phiB=listsB[1][400]#we have to get the last value in order to search for the zero 
                                C=(A+B)/2 #take a mid point
                                listsC=wave_function(C)
                                last_phiC=listsC[1][400]
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





r=eigenvalue()   
print(r)     

E=  r[2]
lists=wave_function(E)
x_graph=lists[0]
phi_graph=lists[1]
plt.plot(x_graph,phi_graph)
plt.axhline(y=0, color='r', linestyle='-')
plt.show()













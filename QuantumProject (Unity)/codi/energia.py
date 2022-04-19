# Codi per comprovar el correcte funcionament de CN2D.
# Volem veure que l'energia correspongui al valor teòric.
import numpy as np
import matplotlib.pyplot as plt

# Funció per normalitzar a cada temps fent servir el mètode de Simpson(? millor?)
# per calcular les integrals.


L=3.
m=1.
hbar=1.
w=2.
tb=2.
ta=0.

deltat=0.01
deltax=0.040
#Nx=int((2*L)/deltax)
#Ny=Nx
Nt=int((tb-ta)/deltat)



def trapezis(xa,xb,ya,yb,dx,fun):
    Nx=int((2*L)/dx)
    Ny=Nx
    funsum=0.
    for i in range(Nx+1):
        funsum=funsum+(fun[i,0]+fun[i,-1])*(dx**2)/2
        for j in range(1,Ny):
        	funsum=funsum+fun[i,j]*dx**2
    
    return funsum
	
# Normalització per a cada temps
def normatotal(norma,dx,L):
	normalitza=np.zeros( (len(norma[1,1,:])))
	for j in range(len(norma[1,1,:])):
		normalitza[j]= trapezis(-L,L,-L,L,dx,norma[:,:,j])
	return normalitza
    
    

normes=np.load('normagvlldx{}dt{}.npy'.format(deltax,deltat))
normavect=normatotal(normes,deltax,L)
np.save('normatotgvlldx{}dt{}.npy'.format(deltax,deltat),normavect)


#Càlcul de l'energia per un temps determinat
def Energia(psi,dx,dy,m,hbar,V):
    Ec=np.zeros((np.shape(psi)),dtype=complex)
    Ep=np.zeros((np.shape(psi))) 
    Nx=np.int(len(psi[0,:]))-1
    Ny=Nx
    for i in range (1,Nx):
        for j in range (1,Ny):
            Ec[i,j]=-((hbar**2.)/(2.*m))*(((psi[i+1,j]-2.*psi[i,j]+psi[i-1,j])/(dx**2))+
				((psi[i,j+1]-2.*psi[i,j]+psi[i,j-1])/(dy**2)))
            Ec[i,j]=np.real(Ec[i,j]*np.conj(psi[i,j]))
            Ep[i,j]=psi[i,j]*np.conj(psi[i,j])*V[i,j]
    return Ec,Ep
		
	
	
# Càlcul de l'energia per tots els temps
def Hamt(psi,dx,dy,m,hbar,V):
    #Torna l'energia en funció del temps
    Nt=len(psi[1,1,:])
    Htc=np.zeros(Nt)
    Htp=np.zeros(Nt)
    Nx=len(psi[0,:,1])-1
   

    for j in range(Nt):
        Ec,Ep=Energia(psi[:,:,j],dx,dy,m,hbar,V)
        sumac=0.
        sumap=0.
        for n in range(Nx+1):
            for k in range(Nx+1):
                        sumac=sumac+Ec[n,k]
                        sumap=sumap+Ep[n,k]
        Htc[j]=sumac*dx**2
        Htp[j]=sumap*dx**2

    return Htc,Htp



#Energia teòrica per oscil·lador harmònic 2D en estat fonamental
def Energia_teo(hbar,w):
    Et=hbar*w
    return Et

#Energia teòrica per oscil·lador harmònic 2D en estat fonamental
def Energia_cteo(hbar,w):
    Et=(1./2.)*hbar*w
    return Et

def E1(hbar,w):
    Et=(3./2.)*hbar*w
    return Et

def E2(hbar,w):
    Et=(5./2.)*hbar*w
    return Et

def E3(hbar,w):
    Et=(7./2.)*hbar*w
    return Et


#Representació de la norma prop

#fig=plt.figure(figsize=[10,6])
#plt.title('Norm conservation for a gaussian wave packet',fontsize=13)

normavector=np.load('normatotgvlldx{}dt{}.npy'.format(deltax,deltat))
tvec=np.load('tvecgvlldx{}dt{}.npy'.format(deltax,deltat))
#plt.plot(tvec,(1-normavector)*10**9,'.',color="#003f5c",label='$p_{0x}$=0; $p_{0y}$=8')	
#plt.ticklabel_format(useOffset=False, style='plain')
#mit=0.
#for i in range (len(normavector)):
#    mit=mit+normavector[i]
    
#mit=mit/(len(normavector))
#sd=0.
#for i in range (len(normavector)):
#    sd=sd+(normavector[i]-mit)**2

#sd=np.sqrt(sd/len(normavector))

#print(mit)
#print(sd)
#y_ticks = [3.53785,3.53780, 3.53775, 3.53770]
#y_tick_labels = ['3.53785', '3.53780', '3.53775', '3.53770']
#plt.yticks(y_ticks,fontsize=13)
#plt.xticks(fontsize=13)
#plt.xlabel('t',fontsize=13)
#plt.ylabel('($1-|\psi^{*}\psi|$)·$10^9$',fontsize=13)
#plt.legend(fontsize=13)
#plt.savefig("norm_conservation_gauss.png",dpi=300)

#representació de l'energia numèrica i teòrica
#fig=plt.figure(figsize=[10,8])
#ax=plt.subplot(111)
#plt.suptitle('Energia oscil·lador harmònic (dx=0,03 i dt=0,02)')

#K0=np.load('Echarm0dx{}dt{}.npy'.format(deltax,deltat))

#K1=np.load('Echarm1dx{}dt{}.npy'.format(deltax,deltat))
#K2=np.load('Echarm2dx{}dt{}.npy'.format(deltax,deltat))
#K3=np.load('Echarm3dx{}dt{}.npy'.format(deltax,deltat))

#U=np.load('Epharmdx{}dt{}.npy'.format(deltax,deltat))
#Ecteo0=np.full(26,Energia_cteo(hbar, w))
#Ecteo1=np.full(26,E1(hbar, w))
#Ecteo2=np.full(26,E2(hbar, w))
#Ecteo3=np.full(26,E3(hbar, w))
#ax.set_ylim([6.97,7.03])
#plt.plot(tvec[0:500],normavect[0:500],'.',label='dt={}/dx={}'.format(deltat,deltax))
#plt.plot(tvec,Ecteo0,'-',label='E0 teòrica')
#plt.plot(tvec,K0,'.',label='E0 computada')
#plt.plot(tvec,Ecteo1,'-',label='E1 teòrica')
#plt.plot(tvec,K1,'.',label='E1 computada')
#plt.plot(tvec,Ecteo2,'-',label='E2 teòrica')
#plt.plot(tvec,K2,'.',label='E2 computada')
#plt.plot(tvec,Ecteo3,'-',label='E3 teòrica')
#plt.plot(tvec,K3,'.',label='E3 computada')
	
		
# Shrink current axis by 20%
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
#plt.xlabel('t')
#plt.ylabel('E')
#plt.savefig('energia')

# Gràfic de l'energia segons dx per dt=0,01 i pox=poy=2 i sigma=0,25
#dx=np.array([0.03+i*0.01 for i in range (18)])
dx=np.array([0.03])

def energiesgauss(pox,poy,sig):
    Eg=((hbar**2)/(2.*m))*(pox**2+poy**2+(1/(2.*(sig))))
    return Eg

for i in range (np.size(dx)):
    if (dx[i]==0.1) or (dx[i]==0.2):
        Vvec=np.load('Vvecgbarrerap0y12p0x12dx{}dt0.01.npy'.format(dx[i]))
        psivec=np.load('psigbarrerap0y12p0x12dx{}dt0.01.npy'.format(dx[i]))
        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
        np.save('Kgbarrerap0y12p0x12dx{:.2f}dt{}.npy'.format(dx[i],deltat),Ec)
        np.save('Pgbarrerap0y12p0x12dx{}dt{}.npy'.format(dx[i],deltat),Ep)
    else:
        Vvec=np.load('Vvecgbarrerap0y12p0x12dx{:.2f}dt0.01.npy'.format(dx[i]))
        psivec=np.load('psigbarrerap0y12p0x12dx{:.2f}dt0.01.npy'.format(dx[i]))
        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
        np.save('Kgbarrerap0y12p0x12dx{}dt{}.npy'.format(dx[i],deltat),Ec)
        np.save('Pgbarrerap0y12p0x12dx{}dt{}.npy'.format(dx[i],deltat),Ep)


#for i in range (np.size(dx)):
#    if (dx[i]==0.1) or (dx[i]==0.2):
#        Vvec=np.load('Vvecgvllp0dx{}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0dx{}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0dx{:.2f}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0dx{}dt{}.npy'.format(dx[i],deltat),Ep)
#    else:
#        Vvec=np.load('Vvecgvllp0dx{:.2f}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0dx{:.2f}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0dx{}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0dx{}dt{}.npy'.format(dx[i],deltat),Ep)
        
#for i in range (np.size(dx)):
#    if (dx[i]==0.1) or (dx[i]==0.2):
#        Vvec=np.load('Vvecgvllp0y12dx{}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0y12dx{}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0y12dx{:.2f}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0y12dx{}dt{}.npy'.format(dx[i],deltat),Ep)
#    else:
#        Vvec=np.load('Vvecgvllp0y12dx{:.2f}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0y12dx{:.2f}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0y12dx{}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0y12dx{}dt{}.npy'.format(dx[i],deltat),Ep)

#for i in range (np.size(dx)):
#    if (dx[i]==0.1) or (dx[i]==0.2):
#        Vvec=np.load('Vvecgvllp0x8dx{}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0x8dx{}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0x8dx{:.2f}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0x8dx{}dt{}.npy'.format(dx[i],deltat),Ep)
#    else:
#        Vvec=np.load('Vvecgvllp0x8dx{:.2f}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0x8dx{:.2f}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0x8dx{}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0x8dx{}dt{}.npy'.format(dx[i],deltat),Ep)
        
#for i in range (np.size(dx)):
#    if (dx[i]==0.1) or (dx[i]==0.2):
#        Vvec=np.load('Vvecgvllp0y18dx{}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0y18dx{}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0y18dx{:.2f}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0y18dx{}dt{}.npy'.format(dx[i],deltat),Ep)
#    else:
#        Vvec=np.load('Vvecgvllp0y18dx{:.2f}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0y18dx{:.2f}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0y18dx{}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0y18dx{}dt{}.npy'.format(dx[i],deltat),Ep)
        
#for i in range (np.size(dx)):
#    if (dx[i]==0.1) or (dx[i]==0.2):
#        Vvec=np.load('Vvecgvllp0y4dx{}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0y4dx{}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0y4dx{:.2f}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0y4dx{}dt{}.npy'.format(dx[i],deltat),Ep)
#    else:
#        Vvec=np.load('Vvecgvllp0y4dx{:.2f}dt0.01.npy'.format(dx[i]))
#        psivec=np.load('psigvllp0y4dx{:.2f}dt0.01.npy'.format(dx[i]))
#        Ec,Ep=Hamt(psivec,dx[i],dx[i],m,hbar,Vvec)
#        np.save('Kgvllp0y4dx{}dt{}.npy'.format(dx[i],deltat),Ec)
#        np.save('Pgvllp0y4dx{}dt{}.npy'.format(dx[i],deltat),Ep)
   
fig=plt.figure(figsize=[12,8])
ax=plt.subplot(111)
plt.title('Kinetic energy of the gasussian wave packet')

E1=np.zeros((np.size(dx)))
#E2=np.zeros((np.size(dx)))
#E3=np.zeros((np.size(dx)))
#E4=np.zeros((np.size(dx)))
#E5=np.zeros((np.size(dx)))
for i in range (np.size(dx)):
    En1=np.load('Kgbarrerap0y12p0x12dx{:.2f}dt0.01.npy'.format(dx[i]))
#    En2=np.load('Kgvllp0dx{:.2f}dt0.01.npy'.format(dx[i]))
#    En3=np.load('Kgvllp0y12dx{:.2f}dt0.01.npy'.format(dx[i]))
#    En4=np.load('Kgvllp0x8dx{:.2f}dt0.01.npy'.format(dx[i]))
#    En5=np.load('Kgvllp0y18dx{:.2f}dt0.01.npy'.format(dx[i]))
    E1[i]=En1[0]
#    E2[i]=En2[0]
#    E3[i]=En3[0]
#    E4[i]=En4[0]
#    E5[i]=En5[0]
    
Eg1=np.array([energiesgauss(12.,12.,0.15) for i in range (np.size(dx))])
#Eg2=np.array([energiesgauss(0.,0.,0.25) for i in range (np.size(dx))])
#Eg3=np.array([energiesgauss(0.,12.,0.25) for i in range (np.size(dx))])
#Eg4=np.array([energiesgauss(8.,0.,0.25) for i in range (np.size(dx))])
#Eg5=np.array([energiesgauss(0.,18.,0.25) for i in range (np.size(dx))])

plt.ylabel("K",fontsize=12)
plt.xlabel("dx",fontsize=11)
plt.minorticks_on()
plt.plot(dx,E1,".--",color="#003f5c",label="Computed K, $p_{0x}$=12; $p_{0y}$=12")
plt.plot(dx,Eg1,".-",color="#58508d",label="Analytical K, $p_{0x}$=0; $p_{0y}$=4")
#plt.plot(dx,E2,".--",color="#58508d",label="Computed K, $p_{0x}$=0; $p_{0y}$=0")
#plt.plot(dx,Eg2,"-",color="#58508d",label="Analytical K, $p_{0x}$=0; $p_{0y}$=0")
#plt.plot(dx,E3,".--",color="#bc5090",label="Computed K, $p_{0x}$=0; $p_{0y}$=12")
#plt.plot(dx,Eg3,"-",color="#bc5090",label="Analytical K, $p_{0x}$=0; $p_{0y}$=12")
#plt.plot(dx,E4,".--",color="#ff6361",label="Computed K, $p_{0x}$=8; $p_{0y}$=0")
#plt.plot(dx,Eg4,"-",color="#ff6361",label="Analytical K, $p_{0x}$=8; $p_{0y}$=0")
#plt.plot(dx,E5,".--",color="#ffa600",label="Computed K, $p_{0x}$=0; $p_{0y}$=18")
#plt.plot(dx,Eg5,"-",color="#ffa600",label="Analytical K, $p_{0x}$=0; $p_{0y}$=18")
plt.ylim(((0,170)))
plt.legend(loc=(1.04,0.25),fontsize=11)
#plt.savefig("kineticenergygaussian.png",bbox_inches="tight",dpi=300)


#Gràfic d'energia cinètica en funció de dx per p0x=2 i p0y=0
#fig=plt.figure(figsize=[10,8])

#plt.title('Kinetic energy dependence on dx')
#plt.subplot(312)
#plt.minorticks_on()
#plt.ylabel("K",fontsize=11)
#plt.xlabel("dx",fontsize=11)
#plt.plot(dx,E1,".-",color="#003f5c",label="$p_{0x}$=4; $p_{0y}$=0")
#plt.legend(loc=0,fontsize=11) 
#plt.subplot(311)
#plt.minorticks_on()
#plt.ylabel("K",fontsize=11)
#plt.xlabel("dx",fontsize=11)
#plt.plot(dx,E2,".-",color="#58508d",label="$p_{0x}$=0; $p_{0y}$=0")
#plt.legend(loc=0,fontsize=11) 
#plt.subplot(313)
#plt.minorticks_on()
#plt.ylabel("K",fontsize=11)
#plt.xlabel("dx",fontsize=11)
#plt.plot(dx,E3,".-",color="#bc5090",label="$p_{0x}=0$; $p_{0y}$=12")
#plt.legend(loc=0,fontsize=11)
#plt.subplots_adjust(hspace=0.4)
#plt.savefig("ec_for_p0_vs_dx.png",dpi=300)

# Comprovació de la dispersió correcta de gaussiana
#def dispersiox(xa,ya,fun):
    #Fun es la normatotal
#    Nx=int((2*L)/deltax)
    #Valor esperat de x:
#    fun1=np.array([[fun[i,j]*(xa+deltax*j) for j in range(Nx+1)] 
#        for i in range(Nx+1)])
#    xesp=trapezis(xa,-xa,xa,-xa,deltax,fun1)
#    #Valor esperat de x**2:
#    fun2=np.array([[fun[i,j]*(xa+deltax*j)**2 for j in range(Nx+1)] 
#        for i in range(Nx+1)])
#    xesp2=trapezis(xa,-xa,xa,-xa,deltax,fun2)
    
#    s2=xesp2-xesp**2
#    return s2

#def dispersioy(xa,xb,fun):
    #Fun es la norma
#    Nx=int((2*L)/deltax)
    #Valor esperat de x:
#    fun1=np.array([[fun[i,j]*(xa+deltax*i) for i in range(Nx+1)] 
#        for j in range(Nx+1)])
#    yesp=trapezis(xa,-xa,xa,-xa,deltax,fun1)
    #Valor esperat de x**2:
 #   fun2=np.array([[fun[i,j]*(xa+deltax*i)**2 for i in range(Nx+1)] 
#        for j in range(Nx+1)])
#    yesp2=trapezis(xa,-xa,xa,-xa,deltax,fun2)
    
#    s2=yesp2-yesp**2
#    return s2

#sigcompux=np.zeros((np.size(tvec)))
#for i in range (len(tvec)):
#    sigcompux[i]=dispersiox(-L,-L,normes[:,:,i])
 
#sigcompuy=np.zeros((np.size(tvec)))
#for i in range (len(tvec)):
#    sigcompuy[i]=dispersioy(-L,-L,normes[:,:,i])
    
#sigteo=np.zeros((np.size(tvec)))
#for i in range (len(tvec)):
#    sigteo[i]=0.25+(hbar*tvec[i]/(2.*np.sqrt(0.25)*m))**2



# Dispersió en y i x per un paquet de p0y=8 p0x=0
#fig=plt.figure(figsize=[10,8])
#plt.title('Evolution of the variance of a gaussian wave packet', fontsize=13)
#plt.minorticks_on()
#plt.plot(tvec,sigcompux,color="#003f5c",label="Computed $\sigma_{x}^2$") 
#plt.plot(tvec,sigteo,color="#58508d",label="Theoretical") 
#plt.plot(tvec,sigcompuy,color="#bc5090",label="Computed $\sigma_{y}^2$")
#plt.xlabel('t',fontsize=13)
#plt.ylabel('$\sigma^2$',fontsize=13)
#plt.xticks(fontsize=13)
#plt.yticks(fontsize=13)
#plt.legend(fontsize=13)
#plt.savefig("variance_gauss.png",dpi=300)

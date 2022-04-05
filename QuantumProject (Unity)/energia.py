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
tb=4.
ta=0.
deltax=0.03	
deltay=deltax
deltat=0.02
Nx=int((2*L)/deltax)
Ny=Nx
Nt=int((tb-ta)/deltat)



def trapezis(xa,xb,ya,yb,dx,fun):
  
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
    
    
Vvec=np.load('Vvecharm0dx{}dt{}.npy'.format(deltax,deltat))
normes=np.load('normaharm0dx{}dt{}.npy'.format(deltax,deltat))

psivec=np.load('psiharm0dx{}dt{}.npy'.format(deltax,deltat))	
normavect=normatotal(normes,deltax,L)


np.save('normatotharm0dx{}dt{}.npy'.format(deltax,deltat),normavect)


#Càlcul de l'energia per un temps determinat
def Energia(psi,dx,dy,m,hbar,V):
    Ec=np.zeros((np.shape(psi)),dtype=complex)
    Ep=np.zeros((np.shape(psi))) 
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
    Htc=np.zeros(Nt+1)
    Htp=np.zeros(Nt+1)
   

    for j in range(Nt+1):
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

Ec,Ep=Hamt(psivec,deltax,deltay,m,hbar,Vvec)

np.save('Echarm0dx{}dt{}.npy'.format(deltax,deltat),Ec)
np.save('Epharm0dx{}dt{}.npy'.format(deltax,deltat),Ep)

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

#Representació de la norma

#fig=plt.figure(figsize=[10,8])
#ax=plt.subplot(111)
#plt.suptitle('norma harmònic')

#normavector=np.load('normatotharm3dx{}dt{}.npy'.format(deltax,deltat))

tvec=np.load('tvecharm3dx{}dt{}.npy'.format(deltax,deltat))

#plt.plot(tvec[0:500],normavect[0:500],'.',label='dt={}/dx={}'.format(deltat,deltax))
#plt.plot(tvec,normavector,'.',label='dt={}/dx={}'.format(deltat,deltax))		
		
# Shrink current axis by 20%
#box = ax.get_position()
#    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#ax.set_ylim([0,1.5])
# Put a legend to the right of the current axis
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
#plt.xlabel('t')
#plt.ylabel('norma')
#plt.savefig('normadiscretizatp05')

#representació de l'energia numèrica i teòrica
fig=plt.figure(figsize=[10,8])
ax=plt.subplot(111)
plt.suptitle('Energia oscil·lador harmònic (dx=0,03 i dt=0,02)')

K0=np.load('Echarm0dx{}dt{}.npy'.format(deltax,deltat))

K1=np.load('Echarm1dx{}dt{}.npy'.format(deltax,deltat))
K2=np.load('Echarm2dx{}dt{}.npy'.format(deltax,deltat))
K3=np.load('Echarm3dx{}dt{}.npy'.format(deltax,deltat))

#U=np.load('Epharmdx{}dt{}.npy'.format(deltax,deltat))
Ecteo0=np.full(26,Energia_cteo(hbar, w))
Ecteo1=np.full(26,E1(hbar, w))
Ecteo2=np.full(26,E2(hbar, w))
Ecteo3=np.full(26,E3(hbar, w))
#ax.set_ylim([6.97,7.03])
#plt.plot(tvec[0:500],normavect[0:500],'.',label='dt={}/dx={}'.format(deltat,deltax))
plt.plot(tvec,Ecteo0,'-',label='E0 teòrica')
plt.plot(tvec,K0,'.',label='E0 computada')
plt.plot(tvec,Ecteo1,'-',label='E1 teòrica')
plt.plot(tvec,K1,'.',label='E1 computada')
plt.plot(tvec,Ecteo2,'-',label='E2 teòrica')
plt.plot(tvec,K2,'.',label='E2 computada')
plt.plot(tvec,Ecteo3,'-',label='E3 teòrica')
plt.plot(tvec,K3,'.',label='E3 computada')
	
		
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
plt.xlabel('t')
plt.ylabel('E')
#plt.savefig('energia')

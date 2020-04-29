import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def tridiag(a, b, c, d):
    """
    Analogous to the function tridiag.f
    Refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    """
    n = len(a)

    cp = np.zeros(n, dtype = np.complex)
    dp = np.zeros(n, dtype = np.complex)
    cp[0] = c[0]/b[0]
    dp[0] = d[0]/b[0]

    for i in range(1,n):
        m = (b[i]-a[i]*cp[i-1])
        cp[i] = c[i]/m
        dp[i] = (d[i] - a[i]*dp[i-1])/m

    x = np.zeros(n, dtype = np.complex)
    x[n-1] = dp[n-1]

    for j in range(1,n):
        i = (n-1)-j
        x[i] = dp[i]-cp[i]*x[i+1]

    return x





def simpson(h,vect):

    #add the extrems
    add=(vect[0]+vect[len(vect)-1])

    #add each parity its factor
    for i in np.arange(2,len(vect)-1,2):
        add+=2*vect[i]

    for i in np.arange(1,len(vect)-1,2):
        add+=4*vect[i]

    #add the global factor
    add*=(h/np.float(3))

    return add





def crank(xx,dx,dt,tmax,phi_inicial,V,aa):
    
    """ Calcula l'evolució de la funció d'ona fins a tmax"""

    nx = len(xx)  
    
    nt = int(tmax/dt) + 1
    
    VV = V(xx,aa)
    
    # Definim el paràmetre r
    r = 1j*dt/(2.0*(dx**2))

    # Definim les tres diagonals del problema tridiagonal
    b = 1j*np.zeros(nx-1)
    a = 1j*np.zeros(nx-1)
    c = 1j*np.zeros(nx-1)

    # Omplim les tres diagonals del problema tridiagonal
    for i in range(0,nx-1): 
        b[i] = 2.0*(1.0 + r) + 1.0j*VV[i]*dt 
        if i != 0:
            a[i] = -r
        if i != (nx-2):
            c[i] = -r
          
    #Definim i construim l'operador de la dreta al que anomenarem matriu_B
    matriu_B = 1j*np.zeros((nx-1,nx-1))
    
    for i in range(0,nx-1):
        matriu_B[i,i] = 2.0*(1.0 -r) - 1.0j*VV[i]*dt
        if i != (nx-2):
            matriu_B [i,i+1] = r
            matriu_B[i+1,i] = r
            
        
    # Contruim una matriu solució  on anirem guardant els vectors phi calculats a cada temps
    matriu_sol = np.zeros((nt,nx+1))
    
    # Escrivim l'estat inicial a la primera fila de la matriu solució
    for i in range(0,nx+1):
        matriu_sol[0,i] = (abs(phi_inicial[i]))**2
    
    # Abans de començar el bucle de temps construim el vector phii
    phii = 1j*np.zeros(nx-1)
    phii = phi_inicial[1:nx]
    
    
    phi = 1j*np.zeros(nx+1)
    
    # Ara comença el bucle de temps
    for t in range(1,nt):
        
        d = np.dot(matriu_B,phii)
        
        # Apliquem condicions de contorn
        d[0] = d[0] + 2.0*r*phi_inicial[0]
        d[nx-2] = d[nx-2] + 2.0*r*phi_inicial[nx]
        
        # Cridem la subrutina tridiag que ens calculi phi pel temsp següent
        phii = tridiag(a,b,c,d)
        
        phi[1:nx] = phii
        phi[0] = phi_inicial[0]
        phi[nx] = phi_inicial[nx]
        
        
        for i in range(0,nx+1):
            matriu_sol[t,i] = (abs(phi[i]))**2  

        
    return matriu_sol






def prob_left(xx,dx,dt,tmax,phi_inicial,centre_potencial,V,aa):
    
    """ Calcula la probabilitat que la funció estigui a l'esquerra 
    de la barrera per a un temps determinat tmax """
    
    
    nx = len(xx)
    
    matriu_phi = crank(xx,dx,dt,tmax,phi_inicial,V,aa)
    
    for i in range(0,nx+1):
        if (centre_potencial-dx) < xx[i] < (centre_potencial + dx):
            n = i
            
    funcio_ona = matriu_phi[tmax,:]
    funcio_esquerra = funcio_ona[0:n]
    
    trans = simpson(dx,funcio_esquerra)/simpson(dx,funcio_ona)
    
    return trans




def energia_inicial(xx,dx,phi_inicial,V,aa):
    
    nx = len(xx)
    
    VV = V(xx,aa)
        
    matriu_H = 1j*np.zeros((nx+1,nx+1))
    
    k = (1.05e-34)**2/(2*9.01e-31)*(dx**2)
    
    for i in range(0,nx+1):
        
        matriu_H[i,i] = VV[i] + k
        if i != nx:
            matriu_H[i,i+1] = -k
            matriu_H[i+1,i] = -k
            
    E = abs(np.dot(np.dot(np.conjugate(phi_inicial),matriu_H),phi_inicial))
    
    return E



def T_analitic(xx,dx,phi_inicial,V,aa):
    
    """ Calcula el coeficient de transmissió a partir de l'energia 
    inicial de la funció d'ona """

    E = energia_inicial(xx,dx,phi_inicial,V,aa)
    
    nx = len(xx)
    
    VV = V(xx,aa)
    
    arrel = []
    
    for i in range (0,nx+1):
        if VV[i] > E:
            arrel = np.append(arrel,np.sqrt(2.0*(VV[i]-E)))
    
   
    integral = simpson(dx,arrel)
    
    T = np.exp((-2.0)*integral)
    
    return T




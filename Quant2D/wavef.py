import numpy as np
import scipy.special as ss        

class Phi():
    
    def __init__(self):
        self.functions = np.array([])

    def add_function(self,fun,param):
        self.functions = np.append(self.functions,(fun,param))
        return
    
    def val(self,x,y):
        if self.functions==[]:
            n=len(x)
            value=np.zeros((n,n),dtype=float)
        else:
            value = 0
            r = (x,y)
            for i in range(0,self.functions.shape[0],2):
                value = value + self.functions[i](r,self.functions[i+1])
        return value

    def clear(self):
        self.functions = np.array([])

        


class Wave():
    
    def __init__(self,pot,PsiIni,dt,T):
        self.PsiIni = PsiIni
        self.pot = pot
        self.dt = dt
        self.T = T
    
    def tridiag(self,a, b, c, d):
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
    
    def Adiagonals(self,N,r):
            b = np.full(N, 1 + 2*r)
            a = np.full(N, -r)
            a[N -1] = 0
            c = np.full(N, -r)
            c[0] = 0
            return a, b, c
    
    def CrankEvolution(self):
        print('Calculating')
        
        N=100
        dx=0.05
        dt=self.dt
        hbar=1.
        
        L=dx*N/2.
        r=1j*dt/(4.*dx**2)
        ntime=self.T
        
        x=np.arange(-L,L,dx)
        meshx , meshy = np.meshgrid(x,x,sparse=True)
            
      #  potential=Pot(meshx,meshy)
      #  psi0=PsiIni(meshx,meshy)
        
        #3D array. First index indicates step of time
        psitime = np.zeros([ntime, N, N], dtype = np.complex)
        
        #2D arrays
        psi=self.PsiIni[:,:] + 0j
      #  print(type(self.pot[:,:]))
        V=self.pot[:,:]
        
        #A diagonals
        Asup, Adiag, Ainf = self.Adiagonals(N,r)
        
        #Bmatrix
        B=np.zeros((N,N),dtype=complex)
        for i in range(0,N):
            for j in range(0,N):
                if i==j:
                    B[i,j]=1.-2.*r-1j*dt*V[i,j]/(4.*hbar)
                if i==j+1 or i+1==j:
                    B[i,j]=r
        
        for t in range(0,ntime):
            
            "Evolve for x"
            for j in range(0,N):
                #Psix
                psix=psi[:,j] +0j
                #Bmatrix*Psi
                Bproduct=np.dot(B,psix)
                #Tridiagonal
                psix=self.tridiag(Ainf,Adiag+1j*dt*\
                             V[:,j]/(4.*hbar),Asup,Bproduct)
                #Change the old for the new values of psi
                psi[:,j]=psix[:]
                    
            "Evolve for y"
            for i in range(0,N):
                #Psiy
                psiy=psi[i,:] +0j
                #Bmatrix*Psi 
                Bproduct=np.dot(B,psiy)
                #Tridiagonal
                psiy=self.tridiag(Ainf,Adiag+1j*dt*\
                             V[i,:]/(4.*hbar),Asup,Bproduct)
                #Change the old for the new values of psi
                psi[i,:]=psiy[:]
                
                
            if (t == int(ntime/4.)):
                print('25%')
            if (t == int(ntime/2.)):
                print('50%')
            if (t == int(3.*ntime/4.)):
                print('75%')
            
            psitime[t,:,:]=psi[:,:]
            
        return psitime  #Evolution of the wave function
    
        
        
    def Probability(self,f):
        " Calculate probability matrix where f is wave function matrix "
        n=len(f[0,:])
        p=np.zeros((n,n),dtype=float)
        for i in range (0,n):
            for j in range (0,n):
                p[i,j]=abs(np.real(np.conjugate(f[i,j])*f[i,j]))
        return p
    
    def ProbEvolution(self):
        f=self.CrankEvolution()
        ntime = len(f[:,0,0])
        n=len(f[0,0,:])
        
        p=np.zeros((ntime,n,n),dtype=float)
        for i in range (0,ntime):
            p[i,:,:]=self.Probability(f[i,:,:])
        return p
    
    def Norm(self,f):
        " Calculate norm where f is probability matrix "
        norm=0.
        n=len(f[0,:])
        pas=0.05
        for i in range (0,n):
            for j in range (0,n):
                norm=norm+f[i,j]
        return norm*pas**2
       

class InitWavef():
    
    def OsciEigen(r,param):
        xx=r[0]  #ROW
        yy=r[1]  #COLUMN
        
        x0 = param[0]
        y0 = param[1]
        w = param[2]
        a = int(param[3])
        b = int(param[4])
        
        m=1.
        hbar=1.
        # w=np.sqrt(abs(k)/m)
        
        zetx=np.sqrt(m*w/hbar)*(xx-x0)
        zety=np.sqrt(m*w/hbar)*(yy-y0)
        Hx=ss.eval_hermite(a,zetx)
        Hy=ss.eval_hermite(b,zety)
        
        c_ab=(2**(a+b)*np.math.factorial(a)*np.math.factorial(b))**(-0.5)
        cte=(m*w/(np.pi*hbar))**0.5
        
        return c_ab*cte*np.exp(-(zetx**2+zety**2)/2)*Hx*Hy
    
    def Gauss(r,param):
        x0 = param[0]
        y0 = param[1]
        sig = param[2]
        px0 = param[3]
        py0 = param[4]
        
        xx = r[0] - x0  #ROW
        yy = r[1] - y0  #COLUMN
        
        static_ga = 1./(sig*np.sqrt(2.*np.pi))*np.exp(-(xx**2 + yy**2)/(2*sig**2))
        momentum_ga = static_ga*np.exp(1j*(px0*xx + py0*yy))
        
        return momentum_ga
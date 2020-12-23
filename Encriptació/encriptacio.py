# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:42:50 2020

@author: Marina
"""
import numpy as np
import random
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.widget import Widget
from kivy.graphics import *
from kivy.core.window import Window
from kivy.clock import Clock
import os

with open("encriptacio1.kv", encoding='utf-8') as kv_file:
    Builder.load_string(kv_file.read())

''' Inici aplicació Encriptació  '''
class encriptacioApp(App):
    #Iniciem l'aplicació
    title="Joc d'encriptació"
    def build(self):
        #return GameWidget()
        return WindowManager()
 
    
""" Funcions"""
f=open("fitxerencriptacio.txt","w") #Si ja existeix el fitxer, afegeix al que ja hi ha

fr=open("Guardarllistes.txt","w")
fr.close()
 
def key_encrypt(message,key):
    '''
    Parameters
    ----------
    message : STRING
    key : UN NÚMERO QUE FA "DESPLAÇAR" CADA LLETRA "X" CARÀCTERS
        
    Returns
    -------
    message_en: MISSATGE ENCRIPTAT
    messageb: MISSATGE BINARI ENCRIPTAT

    '''
    #Passo el missatge en una llista on cada element és un símbol en binari
    messagev=[]
    messageb=[]
    for i in message:
        messagev=messagev+[(ord(i)+key)%126]
        messageb=messageb+[format((ord(i)+key),'b')]
        #La funció ord() passa dels caràcters de l'string al seu valor en ASCII
        #Amb el format(,'b') passo els números a binari
        message_en=""
        
    for i in messagev:
        message_en=message_en+chr(i)
    print('Missatge original:', message)    
    print('Missatge encriptat en abc:', message_en)
    print('Missatge encriptat en binari:',messageb)
    return message_en, messageb    


def create_key(message):
    '''
    Parameters
    ----------
    message : MISSATGE QUE VOLS PASSAR, STRING

    Returns
    -------
    key : CLAU ALEATÒRIA AMB TANTS VALORS COM CARÀCTERS TINGUI EL MISSATGE

    '''
    
    key=np.zeros(len(message),dtype='uint8')
    for i in range(0, np.size(key)):
        key[i]=random.randint(0,126)
    
    return key
    
def key_encrypt2(message,key):
    '''
    Parameters
    ----------
    message : MISSATGE QUE ES VOL TRANSMETRE, STRING
    key : CLAU QUE ES VOL QUE S'ENCRIPTI EL MISSATGE DE TIPUS ARRAY 

    Returns
    -------
    message_en : MISSATGE ENCRIPTAT, STRING
    messageb : MISSATGE ENCRIPTAT EN BINARI, LLISTA (ELS ELEMENTS EN STRING)

    '''
    
    messagev=[]
    messageb=[]
    j=0
    for i in message:
        messagev=messagev+[(ord(i)+key[j])%127]
        messageb=messageb+[format((ord(i)+key[j])%127,'b')]
        j+=+1
        
    message_en=""
    #La funció ord() passa dels caràcters de l'string al seu valor en ASCII
    #Amb el format(,'b') passo els números a binari
    for i in messagev:
        message_en=message_en+chr(i)
    
    return message_en, messageb

def key_desencrypt2b(messageb,key):
    '''
    Parameters
    ----------
    messageb : MISSATGE QUE ES VOL DESENCRIPTAR EN BINARI (LLISTA EN STRING)
    key : CLAU, (ARRAY)
    Returns
    -------
    message_out : RECUPERACIÓ DEL MISSATGE, (STRING)

    '''
    
    message_out=''
    j=0
    for i in messageb:
        message_out=message_out+chr(int(i,2)-key[j])
        j+=+1
    
    return message_out

def key_desencrypt2(message,key):
    '''
    Parameters
    ----------
    messageabc : MISSATGE QUE ES VOL DESENCRIPTAR EN abc (LLISTA EN STRING)
    key : CLAU (ARRAY)
    Returns
    -------
    message_out : RECUPERACIÓ DEL MISSATGE, (STRING)

    '''
    
    message_out=''
    j=0
    for i in message:
        if (ord(i)-key[j])<=0:
            message_out=message_out+chr(127-abs(ord(i)-key[j])%127)
            j+=+1
        else: 
            message_out=message_out+chr(ord(i)-key[j])
            j+=+1
    
    return message_out


def guardar(direccio0,bit0,direccio1,bit1,dir0l,bit0l,dir1l,bit1l):    
    '''
    Funció que guarda les direccions i els bits i ho converteix tot en 
    dues arrays: l'enviada i la rebuda'

    Parameters
    ----------
    direccio0 : DIRECCIÓ ENVIAMENT (STR)
    bit0 : BIT ENVIAT (INT)
    direccio1 : DIRECCIÓ REBUDA (STR)
    bit1:   BIT REBUT(INT)
    dir0l : LLISTA DE TOTES LES DIRECCIONS DE L'ENVIAMENT
    bit0l : LLISTA DE TOTS ELS BITS ENVIATS
    dir1l : LLISTA DE TOTES LES DIRECCIONS DE LA REBUDA
    bit1l : LLISTA DE TOTS ELS BITS REBUTS

    Returns
    -------
    array0: ARRAY ENVIADA
    array1: ARRAY REBUDA

    '''
    
    dir0l.append(direccio0)
    bit0l.append(bit0)
    dir1l.append(direccio1)
    bit1l.append(bit1)
    
    array0=np.empty((len(dir0l),2),dtype=str)
    array1=np.empty((len(dir1l),2),dtype=str)
    
    array0[:,0]=dir0l
    array0[:,1]=bit0l
    array1[:,0]=dir1l
    array1[:,1]=bit1l
    
    return array0,array1


def enviament(direccio0,bit0,direccio1,dir0l,bit0l,dir1l,bit1l):
    '''
    Funció que simila l'enviament dels bits i la transformació d'ells

    Parameters
    ----------
    direccio0 : DIRECCIÓ ENVIAMENT (STR)
    bit0 : BIT ENVIAT (INT)
    direccio1 : DIRECCIÓ REBUDA (STR)
    dir0l : LLISTA DE TOTES LES DIRECCIONS DE L'ENVIAMENT
    bit0l : LLISTA DE TOTS ELS BITS ENVIATS
    dir1l : LLISTA DE TOTES LES DIRECCIONS DE LA REBUDA
    bit1l : LLISTA DE TOTS ELS BITS REBUTS

    Returns
    -------
    array0: ARRAY ENVIADA
    array1: ARRAY REBUDA

    '''
    
    #Si es mesura en la mateixa direcció 100% es rebrà igual
    if direccio0==direccio1:
        bit1=bit0
        
    #Si les dues direccions són diferents:   
    else:
        r=random.random()
        if r<=0.5:
            bit1=0
        else: 
            bit1=1
            
    array0,array1=guardar(direccio0,bit0,direccio1,bit1,dir0l,bit0l,dir1l,bit1l)

    return array0,array1


def guardar2(direccio0,bit0,direccio1,bit1,dir0l,bit0l,dir1l,bit1l):    
    '''
    Funció que guarda les direccions i els bits i ho converteix tot en 
    dues arrays: l'enviada i la rebuda'

    Parameters
    ----------
    direccio0 : DIRECCIÓ ENVIAMENT (STR)
    bit0 : BIT ENVIAT (INT)
    direccio1 : DIRECCIÓ REBUDA (STR)
    bit1:   BIT REBUT(INT)
    dir0l : LLISTA DE TOTES LES DIRECCIONS DE L'ENVIAMENT
    bit0l : LLISTA DE TOTS ELS BITS ENVIATS
    dir1l : LLISTA DE TOTES LES DIRECCIONS DE LA REBUDA
    bit1l : LLISTA DE TOTS ELS BITS REBUTS

    Returns
    -------
    array0: ARRAY ENVIADA
    array1: ARRAY REBUDA

    '''
    
    fr=open("Guardarllistes.txt","w")
    
    
    dir0l.append(direccio0)
    bit0l.append(bit0)
    dir1l.append(direccio1)
    bit1l.append(bit1)
    
    array0=np.empty((len(dir0l),2),dtype=str)
    array1=np.empty((len(dir1l),2),dtype=str)
    
    array0[:,0]=dir0l
    array0[:,1]=bit0l
    array1[:,0]=dir1l
    array1[:,1]=bit1l
    
    #Escric al fitxer les noves llistes
    dir0s=""
    for i in dir0l:
        dir0s+=i
    fr.write(dir0s+"\n")
    
    bit0s=""
    for i in bit0l:
        bit0s+=str(i)
    fr.write(bit0s+"\n")
    
    dir1s=""
    for i in dir1l:
        dir1s+=i
    fr.write(dir1s+"\n")
    
    bit1s=""
    for i in bit1l:
        bit1s+=str(i)
    fr.write(bit1s+"\n")
    
    fr.close()
    
    return array0,array1

def enviament2(direccio0,bit0,direccio1,dir0l,bit0l,dir1l,bit1l):
    '''
    Funció que simila l'enviament dels bits i la transformació d'ells
    Parameters
    ----------
    direccio0 : DIRECCIÓ ENVIAMENT (STR)
    bit0 : BIT ENVIAT (INT)
    direccio1 : DIRECCIÓ REBUDA (STR)
    dir0l : LLISTA DE TOTES LES DIRECCIONS DE L'ENVIAMENT
    bit0l : LLISTA DE TOTS ELS BITS ENVIATS
    dir1l : LLISTA DE TOTES LES DIRECCIONS DE LA REBUDA
    bit1l : LLISTA DE TOTS ELS BITS REBUTS

    Returns
    -------
    array0: ARRAY ENVIADA
    array1: ARRAY REBUDA

    '''
    
    #Si es mesura en la mateixa direcció 100% es rebrà igual
    if direccio0==direccio1:
        bit1=bit0
        
    #Si les dues direccions són diferents:   
    else:
        r=random.random()
        if r<=0.5:
            bit1=0
        else: 
            bit1=1
            
    array0,array1=guardar2(direccio0,bit0,direccio1,bit1,dir0l,bit0l,dir1l,bit1l)

    return array0,array1

def arrays():

    if os.path.getsize("Guardarllistes.txt") == 0:
        #fr=open("Guardarllistes.txt","a")
        dir0=[]
        bit0=[]
        dir1=[]
        bit1=[]
        print("No hi ha res al document")
        
    else:
        with open("Guardarllistes.txt", 'r') as fr:
            dir0r= fr.readlines()[-4]

        with open("Guardarllistes.txt", 'r') as fr:
            bit0r= fr.readlines()[-3]

        with open("Guardarllistes.txt", 'r') as fr:
            dir1r = fr.readlines()[-2]
            
        with open("Guardarllistes.txt", 'r') as fr:
            bit1r = fr.readlines()[-1]
            
        dir0=list(dir0r)
        dir0.pop(len(dir0)-1) #El menys 1 és perquè al escriure al final s'ha de posar un \n
        
        bit0=list(bit0r)
        bit0.pop(len(bit0)-1)
        
        dir1=list(dir1r)
        dir1.pop(len(dir1)-1)
        
        bit1=list(bit1r)
        bit1.pop(len(bit1)-1)       


        fr.close()
        
    return dir0,bit0,dir1,bit1

def llegirarrays():
    if os.path.getsize("Guardarllistes.txt") == 0:
        #fr=open("Guardarllistes.txt","a")
        dir0=[]
        bit0=[]
        dir1=[]
        bit1=[]
        print("No hi ha res al document")
        array0=np.array([0])
        array1=np.array([0])
        
    else:
        with open("Guardarllistes.txt", 'r') as fr:
            dir0r= fr.readlines()[-4]

        with open("Guardarllistes.txt", 'r') as fr:
            bit0r= fr.readlines()[-3]

        with open("Guardarllistes.txt", 'r') as fr:
            dir1r = fr.readlines()[-2]
            
        with open("Guardarllistes.txt", 'r') as fr:
            bit1r = fr.readlines()[-1]
            
        dir0=list(dir0r)
        dir0.pop(len(dir0)-1) #El menys 1 és perquè al escriure al final s'ha de posar un \n
        
        bit0=list(bit0r)
        bit0.pop(len(bit0)-1)
        
        dir1=list(dir1r)
        dir1.pop(len(dir1)-1)
        
        bit1=list(bit1r)
        bit1.pop(len(bit1)-1)       
        
        #Omplo les matrius
        array0=np.empty((len(dir0),2),dtype=str)
        array1=np.empty((len(dir1),2),dtype=str)
    
        array0[:,0]=dir0
        array0[:,1]=bit0
        array1[:,0]=dir1
        array1[:,1]=bit1
        
        fr.close()
        
    return array0, array1


def comparardir(array0,array1):
    array02=np.copy(array0)
    array12=np.copy(array1)
    n,m=np.shape(array0)
    nn,mm=np.shape(array1)
    if n<nn:
        nf=nn
    else:
        nf=n
    print('fins a',nf)
    
    n1,m1=np.shape(array02)
    n2,m2=np.shape(array12)
    
    if n1>n2:
        while(n1>n2):
            n1,m1=np.shape(array02)
            print('n1',n1)
            if (n1<=n2):
                break
            array02=np.delete(array02,n2,0)
            print('n2',n2,np.shape(array02),array02)
            print('array1',array12)
            
    if n2>n1:
        while(n2>n1):
            n1,m1=np.shape(array02)
            array12=np.delete(array12,n1,0)
            print('n2',n2,np.shape(array02),array02)
               
                
    for i in range(nf):
        n1,m1=np.shape(array02)
        n2,m2=np.shape(array12)
        
        
                
                
                
        if i<n1 and i<n2:
            endevant=0
            while(endevant==0):
                n1,m1=np.shape(array02)
                n2,m2=np.shape(array12)
                if i<n1 and i<n2:
                    if array02[i,0]!=array12[i,0]:
                        array02=np.delete(array02,i,0)
                        array12=np.delete(array12,i,0)
                        endevant=0
                    else:
                        endevant=1
                else: 
                    endevant=1
                    break
        
                
        n1,m1=np.shape(array02)
        n2,m2=np.shape(array12)
    
        
            
        '''
        if i<n1 and i>n2:
            for j in range(i,n1):
                array02=np.delete(array02,j,0)
                print('estic fent delete:',j,np.shape(array02))
                print(array02)
            break 
        if i>n1 and i<n2:
            for j in range(i,n2):
                array12=np.delete(array12,i,0)
                #print('estic fent delete')
            break
            
         '''   
        #print('i',i, np.shape(array02),'.',np.shape(array12))
        '''
        while(i<n1 and i>n2):
            endevant=0
            while(endevant==0):
                n1,m1=np.shape(array02)
                if i<n1 and i<n2:
                    if array02[i,0]!=array12[i,0]:
                        array02=np.delete(array02,i,0)
                        array12=np.delete(array12,i,0)
                        endevant=0
                    else:
                        endevant=1
                else: 
                    endevant=1
                    break
                
            n1,m1=np.shape(array02)
            array02=np.delete(array02,i,0)
            
        while( i>n1 and i<n2):
            n2,m2=np.shape(array12)
            array12=np.delete(array12,i,0)    
            '''
            
    return array02,array12
    
def escriure(array0,array1):
    
    fw=open("Guardarllistes.txt","w")
    
    dir0l=list(array0[:,0])
    bit0l=list(array0[:,1])
    dir1l=list(array1[:,0])
    bit1l=list(array1[:,1])
    
    #Escric al fitxer les noves llistes
    dir0s=""
    for i in dir0l:
        dir0s+=i
    fw.write(dir0s+"\n")
    
    bit0s=""
    for i in bit0l:
        bit0s+=str(i)
    fw.write(bit0s+"\n")
    
    dir1s=""
    for i in dir1l:
        dir1s+=i
    fw.write(dir1s+"\n")
    
    bit1s=""
    for i in bit1l:
        bit1s+=str(i)
    fw.write(bit1s+"\n")
    
    fw.close()
    
    
"Funcions del joc"

def collide(rect1,rect2):
        r1x=rect1[0][0]
        r1y=rect1[0][1]
        r2x=rect2[0][0]
        r2y=rect2[0][1]
        r1w=rect1[1][0]
        r1h=rect1[1][1]
        r2w=rect2[1][0]
        r2h=rect2[1][1]
    
        if (r1x<r2x+r2w and r1x+r1w>r2x and r1y<r2y+r2h and r1y+r1h>r2y):
            return True
        else: 
            return False
    

def Bobresultats(direccio,llista1,llista2):
    bits=('0','1')
    llista1.append(random.choice(bits))
    llista2.append(str(direccio))
    return llista1,llista2

def Bobresultats2(i,direccio,arr0,arr1):
    '''Dona els resultats segons l'array de l'Alice'''
    bits=('0','1')
    
    if i==0:
        if direccio==arr0[i,0]:
            arr1=np.array([[direccio,arr0[i,1]]])
        else:
            arr1=np.array([[direccio,random.choice(bits)]])

    else:        
        if direccio==arr0[i,0]:
            arr1=np.append(arr1,[[direccio,arr0[i,1]]],axis=0)
        else:
            arr1=np.append(arr1,[[direccio,random.choice(bits)]],axis=0)
    
    i+=1
    return i,arr1
    

    
    
def dadesb(nbits,missatge):
    '''Crea les dades de l'Alice aleatòries'''
    
    longitud=len(missatge)   
    print("La longitud del missatge és:", longitud)  
    
    lenkey=nbits*longitud
    
    #Si volem que es rebi una clau de longitud lenkey haurem d'enviar molts mes
    n=int(lenkey*(5/2))
    # n és el nombre de partícules que volem enviar
    
    ''' Per provar-ho, s'enviaran bits 0 o 1 aleatoris en direcció x o z aleatòria'''
    posdir=['x','z']
    posbit=['0','1']
    
    dir0l=[]
    bit0l=[]
    dir1l=[]
    bit1l=[]
    
    arr0=np.empty((n,2),dtype=str)
    for i in range(n):
        arr0[i,0]=random.choice(posdir)
        arr0[i,1]=random.choice(posbit)
    #print ('arr0',arr0,'shape',np.shape(arr0))
    return arr0
    
                
    
    

"""_______________________Pantalles___________________"""

class HomeScreen(Screen):
    pass
        

class Screen1(Screen):

        
    def btn(self):
        if self.clau.text == "":
            self.clau.text="0"
        
        print("Missatge per xifrar:", self.mxifrar.text,"Clau:", self.clau.text)
        
        mencriptat,mencriptatb=key_encrypt(self.mxifrar.text,int(self.clau.text))
        
        self.mxifrat.text="Missatge xifrat:  "+mencriptat
        self.mxifrat2.text=mencriptat
        
        #Write
        f.write("NOU MISSATGE------------------------------\n")
        f.write("Missatge per xifrar:\n")
        f.write(str(self.mxifrar.text)+"\n")
        f.write("Clau:\n")
        f.write(str(self.clau.text)+"\n")
        f.write(self.mxifrat.text+"\n")
        

    
    def btn1(self):
        if self.clau2.text=="":
            self.clau2.text="0"
        mdencriptat,mdencriptatb=key_encrypt(self.mxifrat2.text,-int(self.clau2.text))
        
        self.mdxifrat.text="Missatge desxifrat:   "+mdencriptat
  
        


class Screen2(Screen):
    
    def btnkey(self):
        randomkey=create_key(self.mxifrar.text)
        print("Clau aleatòrica:",randomkey)
        self.randomkeytxt.text="Clau generada:"+str(randomkey)
        self.randomkey=randomkey
        
        #Write
        f.write("NOU MISSATGE-------\n")
        f.write("Clau aleatòria:\n")
        f.write(str(randomkey)+"\n")
     
        
    def btn2(self):
     
        mencriptat,mencriptatb=key_encrypt2(self.mxifrar.text, self.randomkey)
        
        self.mxifrat.text="Missatge xifrat:  "+mencriptat
        self.mxifrat2.text=mencriptat
        
        print("Missatge encriptat:",mencriptat)
        
        #Write
        f.write("Missatge per xifrar:\n")
        f.write(self.mxifrar.text+"\n")
        f.write("Missatge xifrat:\n")
        f.write(str(self.mxifrat2.text)+"\n")

    def btn3(self):
        
        mdencriptat=key_desencrypt2(self.mxifrat2.text,self.randomkey)
        
        self.mdxifrat.text="Missatge desxifrat:   "+mdencriptat

    
class Screen3(Screen):

    def btn_array(self):
        '''Botó per imprimir per pantalla la direcció i el bit'''
        dir0l,bit0l,dir1l,bit1l=arrays()
        array0,array1=enviament2(self.dir0.text,int(self.bit0.text),self.dir1.text,dir0l,bit0l,dir1l,bit1l)
        self.array0.text= str(array0)
        self.array1.text= str(array1)
        n,m=np.shape(array1)
        self.bit1.text=str(array1[n-1,m-1])
        
    text= StringProperty('')
    text2= StringProperty('')
    text3= StringProperty('')
    
    def change_text(self):
        ''' Funció per passar la informació a la screen Publidir'''
        array0,array1=llegirarrays()
        self.text= str(array0)
        self.text2= str(array1)
        arraydir=np.empty(np.shape(array0),dtype=str)
        arraydir[:,0]=array0[:,0]
        arraydir[:,1]=array1[:,0]
        self.text3= str(arraydir)
        self.manager.current= "Publidir"
    


class Publidir(Screen):
    array0_text= StringProperty('')
    array1_text2= StringProperty('')
    arraydir_text3= StringProperty('')
    
    def comparar(self):
        array0,array1=llegirarrays()
        array01,array11=comparardir(array0, array1)
        self.array0.text= str(array01)
        self.array1.text= str(array11)
        arraydir=np.empty(np.shape(array01),dtype=str)
        arraydir[:,0]=array01[:,0]
        arraydir[:,1]=array11[:,0]
        self.arraydir.text= str(arraydir)
        escriure(array01, array11)
        
    
    
    '''   
    def __init__(self,**kwargs):
        array0,array1=llegirarrays()
        self.array0.text= str(array0)

    '''
    
class Publidir2(Screen):
    arr0_text= StringProperty('')
    arr1_text2= StringProperty('')
    arraydir_text3= StringProperty('')
    
    def comparar(self):
        arr0=Bob.arr0
        arr1=Bob.arr1
        print('arr0',arr0,'arr1',arr1)
        arr01,arr11=comparardir(arr0, arr1)
        self.arr0.text= str(arr01)
        self.arr1.text= str(arr11)


        
class WindowManager(ScreenManager):
    def __init__(self,**kwargs):
        super(WindowManager, self).__init__(**kwargs)
        self.get_screen('bobscreen').gpseudo_init()

    
"___________Joc_____________"

class Bob(Screen):
    text= StringProperty('')
    text2= StringProperty('')
    text3= StringProperty('')
    
    arr0=dadesb(5,'H')
    
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._on_keyboard_closed,self)
        self._keyboard.bind(on_key_down=self._on_key_down)
        self._keyboard.bind(on_key_up=self._on_key_up)
        
        with self.canvas:
            self.player = Rectangle(source="up.png",pos=(Window.size[0]*0.7,Window.size[1]*0.5),size=(50,100)) #Es pot posar una imatge si es vol
            #self.enemy = Rectangle(pos=(400,400),size=(60,60))
            self.player_dir="z"
            
            
        self.keysPressed = set()        
        self._entities= set()
        
        Clock.schedule_interval(self.move_step,0) #El 0 és cada frame, pro si poses un 2 és cada 2 segons (oframes?)        
        
        self.llistaz=[]
        self.llistax=[]
        
        
        self.pause1=0        
        
        
    #..................................................  
    '''
    def add_entity(self,entity):
        self._entities.add(entity)
        self.canvas.add(entity._instruction) #Per afegir-ho al canvas
    
    def remove_entity(self, entity):
        if entity in self._entities:
            self._entities.remove(entity)
            self.canvas.remove(entity._instruction)
            
    def collides(self, e1, e2):#L'he tornat a definir aquí amb una "s"
        r1x=e1.pos[0]
        r1y=e1.pos[1]
        r2x=e2.pos[0]
        r2y=e2.pos[1]
        r1w=e1.size[0]
        r1h=e1.size[1]
        r2w=e2.size[0]
        r2h=e2.size[1]
    
        if (r1x<r2x+r2w and r1x+r1w>r2x and r1y<r2y+r2h and r1y+r1h>r2y):
            return True
        else: 
            return False
        
    def colliding_entities(self,entity):
        result= set()
        for e in self._entities:
            if self.collides(e,entitiy) and e == entity: #Si xoquen però no són el mateix
                result.add(e)
    
    class Entity(object):
        def __init__(self):
            self._pos = (0,0)
            self._size = (50,50)
            self._instruction = Rectangle(pos= self._pos, size= self._size)

    @property
    def pos(self):
        return self._pos
    
    @pos.setter
    def pos(self,value):
        self._pos = value
        self._instructions.pos = self._pos
        
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self,value):
        self.size =  value
        self._instructions.size = self.size
        
    "Falta el source. Fins ara hem connectat "
    
    class Particle(Entity):
        def __init__(self,pos, speed=300):
            self._speed= speed
            self._pos = pos
            game.bind(on_frame= self.move_step)
    
    
    
    #..............................................
    '''    
    
    '-------Per fer anar el teclat----------'
    def _on_keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_key_down)
        self._keyboard.unbind(on_key_up=self._on_key_up)
        self._keyboard = None
        
    def _on_key_down(self,keyboard,keycode,text,modifiers):
        #self.keysPressed.add(text)
        self.keysPressed.add(keycode[1])
    
    def _on_key_up(self,keyboard,keycode):
        text = keycode[1] #es com esta definit
        if text in self.keysPressed:
            self.keysPressed.remove(text)
            
    '----------Altres funcions--------------'      
    def move_step(self,dt): #dt- quan fa des de l'últim frame (en segons)
        ''' Funció per moure el player i per canviar-lo de direcció   '''
        
        currentx = self.player.pos[0]
        currenty = self.player.pos[1]
        
        step_size = 400*dt #seria com la velocitat (?)
        #Si vols moure més depressa, el 100 pot ser 200... 
        
        if ("w" in self.keysPressed) and (currenty<Window.size[1]-2*self.player.size[1]) and (self.pause1==0):
            currenty+=step_size
        if "s" in self.keysPressed and (currenty> self.player.size[1])and (self.pause1==0):
            currenty-=step_size 
                    
        self.player.pos = (currentx, currenty)
        
        #Part de canviar de direcció
        if ("up" in self.keysPressed) and self.pause1==0:
            self.player_dir="z"
            
            with self.canvas:
                self.canvas.remove(self.player)
                self.player = Rectangle(source="up.png",pos=(currentx,currenty),size=(50,100))
            
        if "right" in self.keysPressed and self.pause1==0:
            self.player_dir="x"
            
            with self.canvas:
                self.canvas.remove(self.player)
                self.player = Rectangle(source="right.png",pos=(currentx,currenty),size=(50,100))
            
                   
        
    def particlemoving(self,dt):
        '''Moviment de la partícula'''
        
        if self.j==0 and self.pause1==0:
            
            partx=self.particle.pos[0]
            party=self.particle.pos[1]
        
            step_size=300*dt
            #print('dt',dt, 'step_size',step_size)

            if partx<10000:
                partx+=step_size
        
            self.particle.pos = (partx,party)
        
        
            if collide((self.player.pos,self.player.size),(self.particle.pos,self.particle.size)):
                self.canvas.remove(self.particle)
                
                if self.j==0:
                    '''
                    llista1,llista2=Bobresultats(self.player_dir,self.llistaz,self.llistax)
                    self.bit1.text= str(llista1)
                    self.bit2.text= str(llista2)
                    '''
                    if self.collides==0:
                        self.arr1=np.zeros((1,2),dtype=str)

                    self.collides,self.arr1=Bobresultats2(self.collides,self.player_dir,self.arr0, self.arr1)
                    
                    Bob.arr1=self.arr1
                    
                    #Passo els resultats a pantalla
                    self.bit1.text=str(self.arr1[:,0])
                    self.bit2.text=str(self.arr1[:,1])                    
                    self.bitst+=1
                    self.bits.text="Bits totals:  "+str(self.bitst)
                    
                self.j+=1
                
            if partx>Window.size[0]*0.8 and self.canvas.indexof(self.particle)!=-1:
                
               # print('Què hi ha al canvas?', self.canvas.indexof(self.particle) )
                self.canvas.remove(self.particle)
                self.j+=1
                
            
        
            
            
    def newparticle(self,dt): 
        '''Creació d'una partícula + moviment'''
        self.j=0
        if self.pause1==0: 
            Clock.schedule_interval(self.particlemoving,0)
            
            with self.canvas:
                self.particle = Ellipse(pos=(50,self.randomposition()),size=(10,10))
            

        
    def randomposition(self):
        '''Vull que em dongui una posició aleatòria de les y'''
        
        sizey=Window.size[1]
        j=random.randint(self.player.size[1],sizey-self.player.size[1])
        return j
    
    
    def change_text(self):
        ''''''' Funció per passar la informació a la screen Publidir'''''''
        print('Arr0',self.arr0)
        self.text= str(self.arr0)
        self.text2= str(self.arr1)
        #arraydir=np.empty(np.shape(self.arr0),dtype=str)
        #arraydir[:,0]=self.arr0[:,0]
        #arraydir[:,1]=self.arr1[:,0]
        self.text3= str('Hey')
        self.manager.current= "Publidir2"
    
    
    '--------- Botons --------------'
    def play(self):    
        '''Botó play'''
        self.llistaz=[]
        self.llistax=[]
        self.bit1.text=""
        self.bit2.text=""
        self.bits.text= "Bits totals: "
        
        self.bitst=0
        self.arr0=Bob.arr0
        print('Array ALice',self.arr0)
        #Contador per les mesures
        self.collides=0
        
        
        self.pause1=0
        self.init=0
        if self.pause1==0:
            
            if self.init==0:
                Clock.schedule_interval(self.particlemoving,0)
                self.init=1
                
            Clock.schedule_interval(self.move_step,0)
            Clock.schedule_interval(self.newparticle,2.5)
            
            self.j=0
            
            with self.canvas:
                self.particle = Ellipse(pos=(50,self.randomposition()),size=(10,10))
            
            self.randomposition()
    
    def pause(self):
        self.pause1+=1
        if self.pause1==0:
            self.pause1=1
        if self.pause1==2:
            self.pause1=0
            
        
        
    
        

        
        
        

            
    
            
#---------------------------         
    def gpseudo_init(self):
        pass
    #    self.main_canvas
     #   self.main_canvas.draw()
        






    
if __name__ == "__main__":
    encriptacioApp().run()

f.close()    
#Ara crearé les diferents finestres:
    


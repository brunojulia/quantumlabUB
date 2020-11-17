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
import os

with open("encriptacio1.kv", encoding='utf-8') as kv_file:
    Builder.load_string(kv_file.read())

''' Inici aplicació Encriptació  '''
class encriptacioApp(App):
    #Iniciem l'aplicació
    title="Joc d'encriptació"
    def build(self):
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
    for i in range(n):
        n1,m1=np.shape(array02)
        if i<n1:
            endevant=0
            while(endevant==0):
                n1,m1=np.shape(array02)
                if i<n1:
                    if array02[i,0]!=array12[i,0]:
                        array02=np.delete(array02,i,0)
                        array12=np.delete(array12,i,0)
                        endevant=0
                    else:
                        endevant=1
                else: 
                    endevant=1
                    break
                
                
        else:
            break
        
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
    
        

class WindowManager(ScreenManager):
    

    pass

    
    
if __name__ == "__main__":
    encriptacioApp().run()

f.close()    
#Ara crearé les diferents finestres:



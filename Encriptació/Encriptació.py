# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:42:50 2020

@author: Marina
"""
import numpy as np
import random
from kivy.app import App
#from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen 
#from kivy.lang import Builder

''' Inici aplicació Encriptació  '''
class encriptacioApp(App):
    #Iniciem l'aplicació
    title="Joc d'encriptació"
    def build(self):
        return WindowManager()
 
    
""" Funcions"""
f=open("fitxerencriptacio.txt","w") #Si ja existeix el fitxer, afegeix al que ja hi ha
 
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


"""Pantalles"""

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
        
        
        

class WindowManager(ScreenManager):
    pass

    
    
if __name__ == "__main__":
    encriptacioApp().run()

f.close()    
#Ara crearé les diferents finestres:



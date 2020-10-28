# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:42:50 2020

@author: Marina
"""

from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import ScreenManager, Screen 
from kivy.lang import Builder

''' Inici aplicació Encriptació  '''
class encriptacioApp(App):
    #Iniciem l'aplicació
    title="Joc d'encriptació"
    def build(self):
        return WindowManager()
    
class HomeScreen(Screen):
    pass   
     

class Screen1(Screen):
    
    
        
        
    def btn(self):
        print("Missatge per xifrar:",self.mxifrar.text,"Clau:",self.clau.text)
              
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
                messagev=messagev+[ord(i)+key]
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
        
        mencriptat,mencriptatb=key_encrypt(self.mxifrar.text,int(self.clau.text))
        
        self.mxifrat.text="El missatge encriptat és:   "+mencriptat

class WindowManager(ScreenManager):
    pass


    
if __name__ == "__main__":
    encriptacioApp().run()
    
#Ara crearé les diferents finestres:



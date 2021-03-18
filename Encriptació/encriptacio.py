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
from kivy.lang import Builder
from kivy.properties import StringProperty
#from kivy.uix.widget import Widget
from kivy.graphics import *
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.popup import Popup
from kivy.uix.floatlayout import FloatLayout
#from kivy.uix.gridlayout import GridLayout
#from kivy.uix.button import Button
import os
#Per fer el multiplayer online
from network import Network
from player import Player

with open("encriptacio1.kv", encoding='utf-8') as kv_file:
    Builder.load_string(kv_file.read())

''' Inici aplicació Encriptació  '''
class encriptacioApp(App):
    #Iniciem l'aplicació
    title="Joc d'encriptació"
    def build(self):
        
        return WindowManager()
 
    
"""________________Funcions_____________________"""
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
    '''
    Funció que compara les dues arrays i elimina els valors amb diferentes 
    direccions de mesura.

    Parameters
    ----------
    array0 : ARRAY DE L'ALICE
    array1 : ARRAY D'EN BOB

    Returns
    -------
    array02 :   ARRAY DE L'ALICE EDITADA
    array12 : ARRAY D'EN BOB EDITADA

    '''
    array02=np.copy(array0)
    array12=np.copy(array1)
    n,m=np.shape(array0)
    nn,mm=np.shape(array1)
    
    if n<nn:
        nf=nn
    else:
        nf=n

    n1,m1=np.shape(array02)
    n2,m2=np.shape(array12)
    
    #Borrem els elements de més si les matrius no tenen la mateixa dimensió
    if n1>n2:
        while(n1>n2):
            n1,m1=np.shape(array02)
            if (n1<=n2):
                break
            array02=np.delete(array02,n2,0)

    if n2>n1:
        while(n2>n1):
            n1,m1=np.shape(array02)
            n2,m2=np.shape(array12)
            if (n2<=n1):
                break
            array12=np.delete(array12,n1,0)               
    
    #Comparem i eliminem els que són de diferents direccions            
    for i in range(nf):
        n1,m1=np.shape(array02)
        n2,m2=np.shape(array12)
              
        if i<n1 and i<n2:
            endevant=0
            while(endevant==0):
                n1,m1=np.shape(array02)
                n2,m2=np.shape(array12)
                if i<n1 and i<n2:
                    if array02[i,1]!=array12[i,1]:
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
    '''
    Dona els resultats segons l'array de l'Alice

    Parameters
    ----------
    i : COMPTADOR DE PARTÍCULES ENVIADES
    direccio : DIRECCIÓ DE MESURA
    arr0 : ARRAY AMB LES DIRECCIONS I ELS BITS DE L'ALICE
    arr1 : ARRAY AMB DIRECCIONS I BITS D'EN BOB

    Returns
    -------
    i : # DE PARTÍCULES MESURADES
    arr1 : ARRAY AMB DIRECCIONS I BITS D'EN BOB 
    '''

    
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
    
def Bobresultats2hack(i,hack, direccio,arr0,arr1):
    '''
    Dona els resultats segons l'array de l'Alice amb possibilitat de hacker!

    Parameters
    ----------
    i : COMPTADOR DE PARTÍCULES ENVIADES
    direccio : DIRECCIÓ DE MESURA
    arr0 : ARRAY AMB LES DIRECCIONS I ELS BITS DE L'ALICE (SENSE COLUMNA DE NUMERO DE BITS!)
    arr1 : ARRAY AMB DIRECCIONS I BITS D'EN BOB

    Returns
    -------
    i : # DE PARTÍCULES MESURADES
    arr1 : ARRAY AMB DIRECCIONS I BITS D'EN BOB 
    '''
    
    bits=('0','1')
    if hack==0:
        
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
    else:
        if i==0:
                arr1=np.array([[direccio,random.choice(bits)]])
    
        else:        

                arr1=np.append(arr1,[[direccio,random.choice(bits)]],axis=0)
    i+=1
    return i,arr1


def dadesb(nbits,missatge):
    '''Crea les dades de l'Alice aleatòries'''
    
    longitud=len(missatge)   
    print("La longitud del missatge és:", longitud)  
    print('Missatge que es vol enviar:',missatge)
    lenkey=nbits*longitud
    
    #Si volem que es rebi una clau de longitud lenkey haurem d'enviar molts mes
    n=int(lenkey*(6/2)) #abans tenia int(6/2)
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
    return arr0,lenkey,n

def mirarhack(array0,array1):
    '''
    Funció que et compara alguns elements de les matrius ja amb les direccions 
    iguals per mirar si hi ha hagut un hacker o no. A més esborra els bits
    publicats de les matrius que ens serveixen com a dades.

    Parameters
    ----------
    array0 : ARRAY DE L'ALICE AMB LES DIRECCIONS IGUALS QUE LES DE'N BOB.
            AMB LA PRIMERA COLUMNA, EL NUMERO DE FILA. 
    array1 : ARRAY DE'N BOB AMB LES DIRECCIONS IGUALS QUE LES DE L'ALICE.
    Returns
    -------
    bool: RETORNA SI HI HA HAGUT HACKER - TRUE I SINÓ FALSE
    publiA/B: MATRIU QUE PUBLICA L'ALICE O EN BOB AMB: LES POSICIONS QUE VOLEN 
    COMPARAR AMB LA DIRECCIÓ I EL VALOR OBTINGUT A LA MESURA.
    array0/1: MATRIU AMB ELS BITS PUBLICATS TRETS AMB LA COLUMNA DEL NOMBRE DE PARTÍCULES 
    '''
    if (np.shape(array0)!=np.shape(array1)):
        print("Les dues matrius no són compatibles")
               
    n,m=np.shape(array0)
    #Comparem els primer bits de les arrays a veure si coincideixen les seves direccions
    comptador=0
    '''
    if n<10:
        k=n
    else:
        k=10
    '''
    #Nombre de bits que comparo (un desè de bits)
    if n>8:
        k=n//8
    else:
        k=1
    print("Valors comparats", k)
    
    #Matriu que publicarà l'Alice
    publiA=np.empty(([k,3]),dtype=object)
    publiB=np.copy(publiA)
    
    for i in range(k):
        #rand=random.randint(0,n-1)
        rand=random.choice(array0[:,0])
        #print(publiA)
        while  np.any(publiA==rand):
            #rand=random.randint(0,n-1)
            rand=random.choice(array0[:,0]) 
             
        publiA[i,0]=rand
        publiB[i,0]=rand
        randindex=np.where(array0[:,0] == rand)[0][0]
        publiA[i,1]=array0[randindex,1] #Tot això abans era rand enlloc de randindex
        publiA[i,2]=array0[randindex,2]
        publiB[i,1]=array1[randindex,1]
        publiB[i,2]=array1[randindex,2]
        
        
        if array0[randindex,2]!=array1[randindex,2]:
            comptador+=1
            
    #Eliminem els bits publicats
    #Ordenem les posicions que hem de borrar
    posicions=sorted(list(publiA[:,0]), reverse=True)
    
    for i in posicions:
        array0=np.delete(array0,np.where(array0[:,0] == i)[0][0],0) #és la posició de l'array on el primer numero és el de "posicions"
        array1=np.delete(array1,np.where(array1[:,0] == i)[0][0],0)
                  
    #Ordeno les matrius publi
    publiAlist=list()
    for i in range(np.shape(publiA)[0]):
        publiAlist.append((publiA[i,0],publiA[i,1],publiA[i,2]))
    dtype= [('num', int), ('dir','U10'), ('bit','U10')]
    publiA2=np.array(publiAlist,dtype=dtype)
    publiA=np.sort(publiA2,order='num')
    
    publiBlist=list()
    for i in range(np.shape(publiB)[0]):
        publiBlist.append((publiB[i,0],publiB[i,1],publiB[i,2]))
    publiB2=np.array(publiBlist,dtype=dtype)
    publiB=np.sort(publiB2,order='num')
    
    
    if comptador>=k//4 and k//4!=0: #Em mira que 1/4 dels valors siguin iguals, sinó significa que hi ha un hacker!
        print('Hacker?',True,'PubliA',publiA, 'publiB',publiB, 'Comptador',comptador,k//4) # "arrays",array0, array1 )
        return True, publiA, publiB,array0,array1  
    elif k//4==0 and comptador>=1:
        print('Hacker?',True,'PubliA',publiA, 'publiB',publiB, 'Comptador',comptador,k//4) # "arrays",array0, array1 )
        return True, publiA, publiB,array0,array1 
    else: 
        print('Hacker?',False,'PubliA',publiA, 'publiB',publiB,'Comptador',comptador) # "arrays",array0, array1 )
        return False, publiA, publiB,array0,array1
    
def clauobtinguda(array1):
    "Funció que a apartir de l'array de bits, obtinguem la clau en bits o num"
    finalkeyb=[]
    for i in range(0,lenkey,5):
        #print('Clau obtinguda?',i, 'lenkey', lenkey)
        num=array1[i]+array1[i+1]+array1[i+2]+array1[i+3]+array1[i+4]
          
        finalkeyb.append(num)
    
    finalkey=[]
    for i in finalkeyb:
        finalkey.append(int(i,2))
    
    return finalkeyb,finalkey

def xifrarmissatge(key,message):
    "Funció que a partir de la clau i el missatge, el xifra en majúscules '.' i ' ' "
    
    ''' l'ascii en majúscules va del 65 al 90'''
    
    messagev=[]
    message2=[]
    j=0
    # Per mirar què falla
    #print('Missatge:',message,type(message))
    #print('Key', key, type(key))
    
    for i in message:
        #print('lletra missatge:',i, ord(i))
        if ord(i)==32:
            messagev=messagev+[(ord(i)-6+key[j])%28] #tindrà 27
            message2=message2+[ord(i)-6]
        elif ord(i)==46:
            messagev=messagev+[(ord(i)-19+key[j])%28] #tindrà 27
            message2=message2+[ord(i)-19]
            
        else: 
            messagev=messagev+[(ord(i)-65+key[j])%28]
            message2=message2+[ord(i)-65]

        j+=+1 
        
    #print('HOLA:',message2) #Em sembla que aquest no ens interessa
    #print(messagev)
    message_en=""
    
    for i in messagev:
        if (i==26):
            message_en=message_en+chr(32)
            print('arroba?')
        elif (i==27):
            message_en=message_en+chr(46)
        else:
            message_en=message_en+chr(i+65)
        
    print("---Missatge encriptat amb la clau---")
    print(message_en)
    
    return message_en


def desxifrarmissatge(message_en,key):
    "És la funció Bob tal qual del fitxer enviamentclau.py."
    "Desxifra el missatge encriptat amb la clau obtinguda."
    message_out=''
    message_out2=[]
    
    j=0
    for i in message_en:
        if (ord(i)==32):
            valor=((28+ord(i)-6-key[j])%28)+65
            if valor==91:
                message_out=message_out+chr(valor-59)
            elif  valor==92:
                message_out=message_out+chr(valor-46)
            else:
                message_out=message_out+chr(valor)  
                
        elif (ord(i)==46):
            valor=((28+ord(i)-19-key[j])%28)+65
            if valor==91:
                message_out=message_out+chr(valor-59)
            elif  valor==92:
                message_out=message_out+chr(valor-46)
            else:
                message_out=message_out+chr(valor)
        else:
            valor=((28+ord(i)-65-key[j])%28)+65
            if valor==91:
                message_out=message_out+chr(valor-59)
            elif  valor==92: 
                message_out=message_out+chr(valor-46)
            else:
                message_out=message_out+chr(valor)
                 

        message_out2=message_out2+[(28+ord(i)-64-key[j])%28]
        j+=1
    print('Missatge en números',message_out2)
    print('---Recuperem el missatge inicial---')
    print(message_out)
    return message_out



import numpy as np

def addnumpart(array):
    '''
    Funció per afegir a la primera columna, el nombre de partícules.

    Parameters
    ----------
    array : ARRAY DE BITS/DIRECCIONS O TOTES DUES COSES

    newarray: ARRAY AMB LA PRIMERA COLUMNA AFEGIDA
    -------
    None.

    '''
    n=np.shape(array)[0]
    
    if len(array[0])==1:
        newarray=np.zeros((n,2),dtype=object)
        for i in range(n):
            newarray[i,0]=i+1
        
        newarray[:,1]=array[:]
        
    else:        
        m=np.shape(array)[1]
        
        newarray=np.zeros((n,m+1),dtype=object)
        for i in range(n):
            newarray[i,0]=i+1
        
        newarray[:,1:m+1]=array[:,0:m]
    return newarray

#arr=np.array([['z',1],['z',0],['z',0],['z',1],['z',0],['x',1]])





#Funcions per multiplayers
def read_pos(str): #rebem string i ho passem a tupla
    str=str.split(",")
    return int(str[0]),int(str[1])

def write_pos(tup): #rebem tupla i ho passem a string
    return str(tup[0])+","+str(tup[1])
    
def read_pos2(str):
    str=str.split(",")
    pos=int(str[0]),int(str[1])
    player=int(str[2])
    return pos,player






'------------------Funció Popup-----------------------------'

class P1(FloatLayout):
    def close(self):
        show_popup.popupWindow1.dismiss()
        
    
        
class P2(FloatLayout):
    
    bits=StringProperty('')
     
    def close(self):
        show_popup.popupWindow2.dismiss()

    def change_text1(self):
        ''''''' Funció per passar la informació a la screen Clau'''''''
        '''
        self.bits= str(Publidir2.arr1f[:,1])
        print(self.bits)
        
        '''
        '''
        finalkeyb=[]
        for i in range(0,lenkey,5):
            num=key[i]+key[i+1]+key[i+2]+key[i+3]+key[i+4]
            finalkeyb.append(num)
        
        finalkey=[]
        for i in finalkeyb:
            finalkey.append(int(i,2) )
        '''
        '''  
        
        '''
        #Publidir2.change_text(Publidir2)
        
        
class P3(FloatLayout):
    def close(self):
        show_popup.popupWindow3.dismiss()

class P4(FloatLayout):
    def close(self):
        show_popup.popupWindow4.dismiss()
        
class P5(FloatLayout):    
    def close(self):
        show_popup.popupWindow5.dismiss()
    
class P6(FloatLayout):    
    def close(self):
        show_popup.popupWindow6.dismiss()
        
class P7(FloatLayout):
    def close(self):
        show_popup.popupWindow7.dismiss()
        
class P8(FloatLayout):
    def close(self):
        show_popup.popupWindow8.dismiss()
        
class P9(FloatLayout):
    def close(self):
        show_popup.popupWindow9.dismiss()
      
class P10(FloatLayout):
    def close(self):
        show_popup.popupWindow10.dismiss()          
    
def show_popup(self,valor):
    "Funció que obra finestres popup"
    "1:Yes right    2:Yes wrong     3:No right      4: No wrong"
    if valor==1:
        show = P1()
        self.popupWindow1 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
        self.popupWindow1.open()
    if valor==2:
        show = P2()
        self.popupWindow2 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
        self.popupWindow2.open()
        
    if valor==3:
        show = P3()
        self.popupWindow3 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
        #Es pot posar auto_dismiss=False
        self.popupWindow3.open()    
    if valor==4:
            show = P4()
            self.popupWindow4 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
            self.popupWindow4.open()
    if valor==5:
            show= P5()
            self.popupWindow5 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
            self.popupWindow5.open()
            
    if valor==6:
            show= P6()
            self.popupWindow6 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
            self.popupWindow6.open()
            
    if valor==7:
            show= P7()
            self.popupWindow7 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
            self.popupWindow7.open()
            
    if valor==8:
            show= P8()
            self.popupWindow8 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
            self.popupWindow8.open()
            
    if valor==9:
            show= P9()
            self.popupWindow9 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
            self.popupWindow9.open()
            
    if valor==10:
            show= P10()
            self.popupWindow10 = Popup(title=' ',content=show, size_hint=(None,None),size=(400,400))
            self.popupWindow10.open()
            
        


"""_______________________Pantalles___________________"""

class HomeScreen(Screen):
     def __init__(self,**kwargs):
        
        super(HomeScreen,self).__init__(**kwargs)
        #Window.fullscreen = 'auto'

  
class Keys(Screen):
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
       
    bits=StringProperty('')
    clau1=StringProperty('')
    
    
    
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.continuar=0
                
        self.comparat=0 
        Publidir2.comparat=self.comparat
        self.findhacker_i=0
        self.comparat2=0

    
    def comparar(self):
        self.comparat2=1 #per poder continuar i fer findhacker
        
        arr0=Bob.arr0
        arr1=Bob.arr1
        #print('arr0',arr0,'arr1',arr1)
        arr01,arr11=comparardir(addnumpart(arr0), addnumpart(arr1))
        #Per passar-ho més tard
        Publidir2.arr01=arr01
        Publidir2.arr11=arr11
        #Pantalla
        self.arr0.text= str(arr01[:,0:2])
        self.arr1.text= str(arr11)
        #Mirem si podrem fer el procés de hacker
        ncomp=np.shape(arr01)[0]
        bitsf=ncomp-int(ncomp//5+1)
        if bitsf<lenkey: #No tenim prous dades
            show_popup(show_popup,5)
            self.comparat=self.comparat+1
            Publidir2.comparat=self.comparat
            
            print('comparat',self.comparat)
        
    def findhacker(self):
        
        self.findhacker_i=1
        print('COMPARAT',self.comparat2)
        if self.comparat2==0:
            show_popup(show_popup,10)
        else:
        
            arr0=Bob.arr0
            arr1=Bob.arr1
            arr01,arr11=comparardir(addnumpart(arr0), addnumpart(arr1)) 
            
            global arr1f
            self.boolean,publiAh,publiBh,self.arr0f,arr1f=mirarhack(arr01,arr11)
                
            
            self.publiAh.text= str(publiAh)
            self.publiBh.text= str(publiBh)
            
    
        
    def hackeryes(self):
        if self.findhacker_i==1:
            if self.boolean ==True:
                show_popup(show_popup,1)
            else:
                show_popup(show_popup,2)
                self.continuar=1
        else:
            show_popup(show_popup,9)
                
    def hackerno(self):
        if self.findhacker_i==1:
            if self.boolean== False:
                show_popup(show_popup,3)
                self.continuar=1
            else:
                show_popup(show_popup,4)
        else:
            show_popup(show_popup,9)
            

    def change_text(self):
        ''''''' Funció per passar la informació a la screen Clau'''''''
        #print('self',self)
        #self.findhacker(self)
        if self.continuar == 1:
            
            if np.shape(arr1f)[0]<lenkey:
                show_popup(show_popup,5)
        
            self.bits= str(arr1f[:,2])
            
            print('array per anar a clau',arr1f)
            global key
            finalkeyb,finalkey=clauobtinguda(arr1f[:,2])
            print('finalkey',finalkey,type(finalkey))
            finalkeytext=" "
            for i in finalkey:
                finalkeytext+=str(i)+" ; "
            
            key=finalkey
            self.clau1= str(finalkeytext)
            self.manager.current= "Clau"
            
        if self.continuar == 0:
            show_popup(show_popup,8)
        
        
class Clau(Screen):
    bitsfinals_text= StringProperty('')
    clau1_text=StringProperty('')
    
    def missatgexifrat(self):
        self.missatge_en=xifrarmissatge(key, message)
        self.message1.text=self.missatge_en
        
    def desxifrarmissatgekv(self):
        print(self.missatge_en, self)
        missatge_out=desxifrarmissatge(self.missatge_en,key)
        self.messagef.text=missatge_out        
    

        
class WindowManager(ScreenManager):
    def __init__(self,**kwargs):
        super(WindowManager, self).__init__(**kwargs)
        self.get_screen('bobscreen').gpseudo_init()

    
"___________Joc_____________"



class Bob(Screen):
    text= StringProperty('')
    text2= StringProperty('')
    text3= StringProperty('')
    
    global lenkey
    global message
    global npart
    global hack
    message='HI'
    arr0,lenkey,npart=dadesb(5,message)
    #comparat=0
    
    
    
    
    
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        '''
        self._keyboard = Window.request_keyboard(self._on_keyboard_closed,self)
        self._keyboard.bind(on_key_down=self._on_key_down)
        self._keyboard.bind(on_key_up=self._on_key_up)
        '''
        
        with self.canvas:
            self.player = Rectangle(source="up2.png",pos=(Window.size[0]*1.3,Window.size[1]*0.5),size=(50,100)) #Es pot posar una imatge si es vol
            self.player_dir="z"
            
        '''    
        self.keysPressed = set()        
        self._entities= set()
        '''
        #Clock.schedule_interval(self.move_step_bob,0) #El 0 és cada frame, pro si poses un 2 és cada 2 segons (oframes?)        
        
        self.llistaz=[]
        self.llistax=[]
        
        
        self.pause1=0   
        self.primeraparticula=False
        
        self.nmesura_text="Has de mesurar més de "+str(lenkey*3)+" partícules per obtenir el missatge!"
        
    #Per si vull posar més d'una partícula a l'hora. Encara no sé com va 
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
        self.keysPressed.add(keycode[1])
    
    def _on_key_up(self,keyboard,keycode):
        text = keycode[1] #es com esta definitkeyboard,keycode,text,modifiers
        if text in self.keysPressed:
            self.keysPressed.remove(text)
            
    '----------Altres funcions--------------'      
    def move_step_bob(self,dt): #dt- quan fa des de l'últim frame (en segons)
        ''' Funció per moure el player i per canviar-lo de direcció   '''
        
        currentx = self.player.pos[0]
        currenty = self.player.pos[1]
        
        step_size = 600*dt #seria com la velocitat (?)
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
                self.player = Rectangle(source="up2.png",pos=(currentx,currenty),size=(50,100))
            
        if "right" in self.keysPressed and self.pause1==0:
            self.player_dir="x"
            
            with self.canvas:
                self.canvas.remove(self.player)
                self.player = Rectangle(source="right2.2.png",pos=(currentx,currenty),size=(80,100))
            
                   
 
    def particlemoving(self,dt):
        '''"Moviment de la partícula"'''
        
        
        if self.j==0 and self.pause1==0:
            #print('comparat',comparat)
            partx=self.particle.pos[0]
            party=self.particle.pos[1]
        
            step_size=700*dt
            #print('dt',dt, 'step_size',step_size)

            partx+=step_size
        
            self.particle.pos = (partx,party)
        
            #Col·lisió
            if collide((self.player.pos,self.player.size),(self.particle.pos,self.particle.size)):
                self.canvas.remove(self.particle)
                
                if self.j==0: #Perquè no ho faci més d'un cop
                
                    if self.collides==0: #Això és per crear la primera matriu 
                        self.arr1=np.zeros((1,2),dtype=str)
                    
                    if Publidir2.comparat==0: #Si no hem publicat encara les dades cap cop
                        self.collides,self.arr1=Bobresultats2hack(self.collides,hack,self.player_dir,self.arr0, self.arr1)
                    
                    else: #Ja hem comparat les dades algun cop
                    
                        #Nous valors per l'array de l'Alice
                        arr01,res,n2=dadesb(5,message)
                        #arr02=np.concatenate((Publidir2.arr01,arr01),axis=0)
                        print('publidir2',Publidir2.arr01)
                        print('2na concanate',addnumpart(arr01))
                        self.arr0=np.concatenate((Publidir2.arr01[:,1:3],arr01),axis=0) #Array de l'alice amb nous valors
                        self.arr1=Publidir2.arr11[:,1:3]       #Array d'en Bob
                       
                        self.collides=np.shape(Publidir2.arr01)[0] #Això perquè torni a mirar un índex d'abans
                        
                        self.collides,self.arr1=Bobresultats2hack(self.collides,hack,self.player_dir,self.arr0,self.arr1)
                        
                        global npart
                        npart=npart+n2
                        Publidir2.comparat=0
                        
                        
                        
                    Bob.arr1=self.arr1
                    Bob.arr0=self.arr0
                    
                    #Passo els resultats a pantalla
                    
                    self.bit1.text=str(addnumpart(self.arr1[:,0]))
                    self.bit2.text=str(addnumpart(self.arr1[:,1]))                  
                    self.bitst+=1
                    self.bits.text="Bits totals:  "+str(self.bitst)
                    if self.bitst==1:
                        self.primeraparticula=True
                    if self.bitst==npart and Publidir2.comparat==0:
                        self.pause1=1 #Abans hi havia un 3
                        self.pause_id.text='[color=#FFFFFF]Play[/color]'
                        
                        show_popup(show_popup,6)
                        print('comparat',Publidir2.comparat)
                    #if self.bitst==(npart+n2):
                     #   self.pause1=1 #Abans hi havia un 3
                      #  show_popup(show_popup,6)  
                    
                self.j+=1
                
            if partx>Window.size[0]*0.8 and self.canvas.indexof(self.particle)!=-1:
                #Si la partícula està dintre del canvas o no                
               # print('Què hi ha al canvas?', self.canvas.indexof(self.particle) )
                self.canvas.remove(self.particle)
                self.j+=1
                

    def newparticle(self,dt): 
        '''Creació d'una partícula + moviment'''
        self.j=0
        if self.pause1==0: 
            #Clock.schedule_interval(particlemoving(1/60),1/60*dt) #Hi havia un 0
            
            with self.canvas:
                self.particle = Ellipse(pos=(50,self.randomposition()),size=(10,10))
                
  
        
    def randomposition(self):
        '''Vull que em dongui una posició aleatòria de les y'''
        
        sizey=Window.size[1]
        j=random.randint(self.player.size[1],sizey-self.player.size[1])
        return j
    
    
    def change_text(self):
        ''' Botó: Compartir dades. 
        Funció per passar la informació a la screen Publidir2.'''
        if self.primeraparticula :
            #print('Arr0',self.arr0)
            #print('Arr1',self.arr1)
            self.text= str(addnumpart(self.arr0[:,0]))
            self.text2= str(addnumpart(self.arr1))
            self.manager.current= "Publidir2"
        else:
            show_popup(show_popup,7)
    
    
    '--------- Botons --------------'
    def start(self):    
        'Botó per començar a jugar/Tornar a rebre partícules'
        print('comparat',Publidir2.comparat)
        self.llistaz=[]
        self.llistax=[]
        self.bit1.text=""
        self.bit2.text=""
        self.bits.text= "Bits totals: "
        
        self.bitst=0
        self.arr0=Bob.arr0
        print('Array Alice',self.arr0)
        #Contador per les mesures
        self.collides=0
        #Control de si el joc està en pausa o si és l'inici
        self.pause1=0
        self.init=0
        
        global hack

        hack=(random.choices([0,1],weights=(0.75,0.25)))[0]
        #0.75,0.25
        if hack==0:
            print(hack,'NO HI HA HACKER')
        else:
            print(hack,'HACKER!!!')
            
        'Treiem el Clock si el teniem abans'
        global event_ini1
        global event_ini2
        global event_mov
        
        if self.start1.text=='Tornar a començar':
            event_ini1.cancel()
            event_ini2.cancel()
            event_mov.cancel()
            
        else:
            Clock.schedule_interval(self.move_step_bob,0)  
            #Si no ho poso aquí es va incrementant
       
        '''
        with self.canvas:
            self.particle = Ellipse(pos=(50,self.randomposition()),size=(10,10))
            
        self.randomposition()
        '''    
        #TOT LO DEL TECLAAAT-------------------------
        self._keyboard = Window.request_keyboard(self._on_keyboard_closed,self)
        self._keyboard.bind(on_key_down=self._on_key_down)
        self._keyboard.bind(on_key_up=self._on_key_up)
        
        
        
        
        self.keysPressed = set()        
        self._entities= set()
        
        #-----------------------
           
        if self.pause1==0 and self.init==0:
            self.init+=1
            #Per la partícula inicial
            event_ini1=Clock.schedule_once(self.newparticle)
            #Creació de les altres partícules cada 3 segons
            event_ini2=Clock.schedule_interval(self.newparticle,3)
            #Moviment de la partícula 
            event_mov=Clock.schedule_interval(self.particlemoving,0)            
            self.j=0
            
        if self.start1.text=='Començar':
            self.start1.text= 'Tornar a començar'
        
    
    def pause(self):
        self.pause1+=1
        self.pause_id.background_normal= 'play1.jpg'
        #self.pause_id.text= '[color=#FFFFFF]Reprendre[/color]'
        if self.pause1==0:   
            self.pause1=1
            
        if self.pause1==2:
            self.pause1=0
            #self.pause_id.text= '[color=#FFFFFF]Pause[/color]'
            self.pause_id.background_normal= 'pause1.jpg'
            
        
    def acabaricompartir(self):
        "Acaba de mesurar totes les partícules"
        #Partícules que falten per mesurar           re
        
        if self.primeraparticula :
            #print('npart',npart)
            n=npart-self.bitst
            direccions=['z','x']
                
            for i in range(n):
                #self.arr1=np.zeros((1,2),dtype=str)
        
                self.collides,self.arr1=Bobresultats2hack(self.collides,hack,random.choice(direccions),self.arr0, self.arr1)
                            
                Bob.arr1=self.arr1
                Bob.arr0=self.arr0
                
                self.bit1.text=str(addnumpart(self.arr1[:,0])) #abans no tenia addnumpart
                self.bit2.text=str(addnumpart(self.arr1[:,1]))                    
                self.bitst+=1
                self.bits.text="Bits totals:  "+str(self.bitst)
                    
                if self.bitst==npart:
                    self.pause1=1
                    self.pause_id.background_normal= 'play1.jpg'
                    #self.pause_id.text='[color=#FFFFFF]Reprendre[/color]'
        
#--------        
    def gpseudo_init(self):
        pass
    #    self.main_canvas
     #   self.main_canvas.draw()





class Multiplayer(Screen):
    global n
    n=Network()
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        global n
        self.p=n.getP()
        print('Player',self.p.nplayer)
        print('Llest?',self.p.ready)
        
        Multiplayer.p=self.p
        
        self.p0ready=False
        self.p1ready=False
        
        self.continuar=False
        
    def connection(self,dt):
            #print('Intento enviar p:',self.p.nplayer)

            self.p2=n.send(self.p)
            
            #print('Rebo el p:',self.p2.nplayer,'connectat?',self.p2.ready)
            
            if self.p.nplayer==0 and self.p2.ready:
                self.p1ready=True
                self.waitingbob.text= 'A punt'
            elif self.p.nplayer==0 and self.p2.ready==False:
                self.p1ready=False
                self.waitingbob.text= 'Esperant connexió...'
                
            if self.p.nplayer==1 and self.p2.ready:
                self.p0ready=True
                self.waitingalice.text= 'A punt'
            elif self.p.nplayer==1 and self.p2.ready==False:
                self.p0ready=False
                self.waitingalice.text= 'Esperant connexió...'
                
            if self.p0ready and self.p1ready:
                self.continuar=True
            else:
                self.continuar= False
                
            
    def button(self):
        global n
        
        if self.p.nplayer==0:
            self.waitingalice.text='A punt'
            self.p0ready=True
            self.conectat0=True
        else:
            self.waitingbob.text='A punt'
            self.p1ready=True
            self.conectat1= True
        self.p.ready=True
        
        print('Player:',self.p.nplayer, 'Estàs a punt?', self.p.ready)
        
        self.event_connect=Clock.schedule_interval(self.connection,0)
        
            
    
    def continuar_btt(self):
        ''''''' Funció per passar la informació a la screen Clau'''''''
       
        if self.continuar:
            self.event_connect.cancel()
            self.manager.current= "aliceibob"
            

    


class Aliceibob(Screen):
    #global n
    #n=Network() 
    
    def __init__(self,**kwargs):
       
        super().__init__(**kwargs)
        '''
        self._keyboard_multi = Window.request_keyboard(self._on_keyboard_closed,self)
        self._keyboard_multi.bind(on_key_down=self._on_key_down)
        self._keyboard_multi.bind(on_key_up=self._on_key_up)
        '''
        #per agafar la posició inicial dels dos jugadors, la demanem ja per xarxa
        global n      
        #self.p=n.getP()
        self.p=Multiplayer.p
        
        print('Player',self.p.nplayer)
        startPos=(self.p.x,self.p.y)
        
        #startPos=(100,100)
        self.pause1m=0
        self.init=0
        self.j=0
        
        with self.canvas:
            self.player1 = Rectangle(source="up2.png",pos=startPos,size=(50,100)) #Es pot posar una imatge si es vol com a en Bob
            self.player2=  Rectangle(pos=(Window.size[0]*0.7,Window.size[1]*0.5),size=(50,100))
            
        '''    
        self.keysPressed_multi = set()        
        self._entities_multi= set()
        '''
        
        
        
        
        
        '-------Per fer anar el teclat----------'
        
        
    def _on_keyboard_closed(self):
        self._keyboard_multi.unbind(on_key_down=self._on_key_down)
        self._keyboard_multi.unbind(on_key_up=self._on_key_up)
        self._keyboard_multi = None
        
    def _on_key_down(self,keyboard,keycode,text,modifiers):
        self.keysPressed_multi.add(keycode[1])
    
    def _on_key_up(self,keyboard,keycode):
        text = keycode[1] #es com esta definit
        if text in self.keysPressed_multi:
            self.keysPressed_multi.remove(text)
           
        
            
    '----------Altres funcions--------------'   
    
    def move_step(self,dt): #dt- quan fa des de l'últim frame (en segons)
         #Funció per moure el player i per canviar-lo de direcció   
        
        currentx = self.player1.pos[0]
        currenty = self.player1.pos[1]
        
        step_size = 600*dt #seria com la velocitat (?)
        #Si vols moure més depressa, el 100 pot ser 200... 
        
        
        if ("w" in self.keysPressed_multi) and (currenty<Window.size[1]-2*self.player1.size[1]) and (self.pause1m==0):
            currenty+=step_size
            
        if "s" in self.keysPressed_multi and (currenty> self.player1.size[1])and (self.pause1m==0):
            currenty-=step_size 
                    
        self.player1.pos = (currentx, currenty)
        self.p.x=currentx
        self.p.y=currenty
        
        #Part de canviar de direcció
        if ("up" in self.keysPressed_multi) and self.pause1m==0:
            self.player1_dir="z"
            
            with self.canvas:
                self.canvas.remove(self.player1)
                self.player1 = Rectangle(source="up2.png",pos=(currentx,currenty),size=(50,100))
            
        if "right" in self.keysPressed_multi and self.pause1m==0:
            self.player_dir="x"
            
            with self.canvas:
                self.canvas.remove(self.player1)
                self.player1 = Rectangle(source="right2.2.png",pos=(currentx,currenty),size=(80,100))
                
        if "spacebar" in self.keysPressed_multi and self.pause1m==0 and self.p.nplayer==0:
            event_ini1=Clock.schedule_once(self.newparticle)
            #Moviment de la partícula 
            #
            #self.j-=1    
                
        
        
        #Anem a enviar i rebre coses del network
        global n
        self.p2=n.send(self.p)
        #print('Posició actual (el que s hauria d enviar)',((self.player1.pos[0],self.player1.pos[1])))
        #player2pos= read_pos(n.send(write_pos((int(self.player1.pos[0]),int(self.player1.pos[1])))))
        #print('que rebem dspres d enviar',player2pos)
        self.player2.pos=(self.p2.x,self.p2.y)
        
                
        
        
    def particlemoving(self,dt):
         '''"Moviment de la partícula"'''
        
        
         if self.j==1 and self.pause1m==0:
            #print('comparat',comparat)
            partx=self.particle.pos[0]
            party=self.particle.pos[1]
        
            step_size=300*dt
            #print('dt',dt, 'step_size',step_size)

            partx+=step_size
        
            self.particle.pos = (partx,party)
        
            #
            if self.p.nplayer==0:
                alice_pos=self.player1.pos
                bob_pos=self.player2.pos
            else:
                alice_pos=self.player2.pos
                bob_pos=self.player1.pos 
                
            #Col·lisió
            
            if collide((bob_pos,self.player1.size),(self.particle.pos,self.particle.size)):
                self.canvas.remove(self.particle)
                self.event_mov.cancel()  
                if self.j==1: #Perquè no ho faci més d'un cop
                
                    if self.collides==0: #Això és per crear la primera matriu 
                        self.arr1=np.zeros((1,2),dtype=str)
                    
                    if Publidir2.comparat==0: #Si no hem publicat encara les dades cap cop
                        self.collides,self.arr1=Bobresultats2hack(self.collides,hack,self.player_dir,self.arr0, self.arr1)
                    
                    else: #Ja hem comparat les dades algun cop
                    
                        #Nous valors per l'array de l'Alice
                        arr01,res,n2=dadesb(5,message)
                        #arr02=np.concatenate((Publidir2.arr01,arr01),axis=0)
                        print('publidir2',Publidir2.arr01)
                        print('2na concanate',addnumpart(arr01))
                        self.arr0=np.concatenate((Publidir2.arr01[:,1:3],arr01),axis=0) #Array de l'alice amb nous valors
                        self.arr1=Publidir2.arr11[:,1:3]       #Array d'en Bob
                       
                        self.collides=np.shape(Publidir2.arr01)[0] #Això perquè torni a mirar un índex d'abans
                        
                        self.collides,self.arr1=Bobresultats2hack(self.collides,hack,self.player_dir,self.arr0,self.arr1)
                        
                        global npart
                        npart=npart+n2
                        Publidir2.comparat=0
                        
                        
                        
                    Bob.arr1=self.arr1
                    Bob.arr0=self.arr0
                    
                    #Passo els resultats a pantalla
                    
                    self.bit1.text=str(addnumpart(self.arr1[:,0]))
                    self.bit2.text=str(addnumpart(self.arr1[:,1]))                  
                    self.bitst+=1
                    self.bits.text="Bits totals:  "+str(self.bitst)
                    if self.bitst==1:
                        self.primeraparticula=True
                    if self.bitst==npart and Publidir2.comparat==0:
                        self.pause1=1 #Abans hi havia un 3
                        self.pause_id.text='[color=#FFFFFF]Play[/color]'
                        
                        show_popup(show_popup,6)
                        print('comparat',Publidir2.comparat)
                    #if self.bitst==(npart+n2):
                     #   self.pause1=1 #Abans hi havia un 3
                      #  show_popup(show_popup,6)  
                  
                self.j-=1
                
            if partx>Window.size[0]*0.8 and self.canvas.indexof(self.particle)!=-1:
                #Si la partícula està dintre del canvas o no                
               # print('Què hi ha al canvas?', self.canvas.indexof(self.particle) )
                self.canvas.remove(self.particle)
                #self.j+=1  
                self.event_mov.cancel()
                self.j-=1
        
        
    def newparticle(self,dt): 
        '''Creació d'una partícula + moviment'''
        
        if self.pause1m==0 and self.j==0: 
            self.j+=1
            with self.canvas:
               if self.p.nplayer==0:
                   self.particle = Ellipse(pos=self.player1.pos,size=(10,10))
                   
               else:
                   self.particle = Ellipse(pos=self.player2.pos,size=(10,10))
                   
            self.event_mov=Clock.schedule_interval(self.particlemoving,0) 
            #self.j+=1
        #self.j=0
        
        
            
        
        
        
    def start_multi(self):
        #Coses del teclat
        self._keyboard_multi = Window.request_keyboard(self._on_keyboard_closed,self)
        self._keyboard_multi.bind(on_key_down=self._on_key_down)
        self._keyboard_multi.bind(on_key_up=self._on_key_up)
        self.keysPressed_multi = set()        
        self._entities_multi= set()
        
        #Inicialitzem el moviment del jugador
        if self.start1.text=='Començar':
            self.start1.text= 'Tornar a començar'
            Clock.schedule_interval(self.move_step,0)
            
        if self.start1.text=='Tornar a començar':
            #event_ini1.cancel()
            #event_ini2.cancel()
            #event_mov.cancel()
            pass
            
            #Copiada totalmeeeent           
        if self.pause1m==0 and self.init==0:
            self.init+=1
            #Per la partícula inicial
            #event_ini1=Clock.schedule_once(self.newparticle)
            #Creació de les altres partícules cada 3 segons
            #event_ini2=Clock.schedule_interval(self.newparticle,3)
            #Moviment de la partícula 
            #event_mov=Clock.schedule_interval(self.particlemoving,0)            
            self.j=0
            
            
        self.collides=0
        
        
        
        

      


    
if __name__ == "__main__":
    encriptacioApp().run()

f.close()    
#Ara crearé les diferents finestres:
    


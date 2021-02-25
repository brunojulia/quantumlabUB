# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 16:30:05 2020

@author: Marina
"""

'''
PROJECTE: ENCRYPTION

Última modificacició: 16/10/20 
'''


import numpy as np
import matplotlib.pyplot as plt
import skimage
from scipy import signal 
import math
import scipy
import random
 
'''
Provo com passar d'un text a binari. 
FET
'''


#Passem de string a una llista amb binari però cada número en string. 
string="Hello world"
'''
stringb=[]
for i in string:
    stringb=stringb+[format(ord(i),'b')]

print('En binari és:',''.join((stringb)))

string_out=""
for i in stringb:

    string_out=string_out+chr(int(i,2))
    
print('Tornem a llegir-ho:',string_out)
'''

'''
Com a partir d'una clau podem encriptar un missatge
'''

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
        messagev=messagev+[(ord(i)+key)%127]
        messageb=messageb+[format((ord(i)+key)%127,'b')]
        #La funció ord() passa dels caràcters de l'string al seu valor en ASCII
        #Amb el format(,'b') passo els números a binari
    message_en=""
    
    for i in messagev:
        message_en=message_en+chr(i)
    print('Missatge original:', message)    
    print('Missatge encriptat en abc:', message_en)
    #print('Missatge encriptat en binari:',messageb)
    return message_en, messageb

miss3,miss4=key_encrypt('Hello world',20000)


def key_desencrypt(message, key):
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
        messagev=messagev+[127-abs(ord(i)-key)%127]
        messageb=messageb+[format(127-abs((ord(i)-key))%127,'b')]
        #La funció ord() passa dels caràcters de l'string al seu valor en ASCII
        #Amb el format(,'b') passo els números a binari
    message_en=""
    
    for i in messagev:
        message_en=message_en+chr(i)
    print('Missatge desencriptat en abc:', message_en)
    #print('Missatge desencriptat en binari:',messageb)
    return message_en, messageb

missatge3,missatge4=key_desencrypt(miss3,20000)
    

#%%

'''
Ara anem a fer-ho que la clau sigui un array on cada número sigui la clau 
individual per cada símbol.
'''
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

key1=create_key("Hello bon dia em dic Marina")
    
print('Clau aleatòria:',key1)


#%%

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
        j+=+1  #Eeeeepa aqui no va un j+=1 (?)
        
    message_en=""
    #La funció ord() passa dels caràcters de l'string al seu valor en ASCII
    #Amb el format(,'b') passo els números a binari
    for i in messagev:
        message_en=message_en+chr(i)
    
    return message_en, messageb

message1,message2=key_encrypt2("Hello bon dia", key1)

print('Missatge original:', "Hello bon dia em dic Marina")    
print('Missatge encriptat en abc:', message1)
#print('Missatge encriptat en binari:',message2)

#%%

def key_desencrypt2(message,key):
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
    for i in message:
        message_out=message_out+chr(127-((abs(ord(i)-key[j]))%127))
        j+=+1  #Eeeeepa aqui no va un j+=1 (?)
    
    return message_out



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
        j+=+1  #Eeeeepa aqui no va un j+=1 (?)
    
    return message_out

print('Missatge desencriptat:',key_desencrypt2(message1, key1))

#print('Missatge binari encriptat junt', ''.join(message2))
#print('Long',len(''.join(message2)))



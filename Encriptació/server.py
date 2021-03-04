# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:09:00 2021

@author: Marina
"""

''' Prova de fer multijugador online del projecte
de criptació quàntica'''

import socket
from _thread import *
import sys

server="192.168.0.26" #Això és per network local, suposo per la meva wifi
port= 5555
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind((server, port))
    
except socket.error as e:
    str(e)

s.listen(2) #nse si és 1 o 2 per connectar a 2 persones
print("Waiting for a connection, Server Started")


currentPlayer=0 #això és per veure quin jugador utilitzem

def read_pos(str): #rebem string i ho passem a tupla
    str=str.split(",")
    return int(str[0]),int(str[1])
    
def write_pos(tup): #rebem tupla i ho passem a string
    return str(tup[0])+","+str(tup[1])

#El read_pos va bé

pos=[(100,100),(560,300)]#aquí hi haurà les posicions dels jugadors
#Amb 1.3*win size crec que correspon 1040, 300


def threaded_client(conn, player): 
    global currentPlayer
    #conn.send(str.encode("Connected"))
    #Convertim la posició(del jugador que toca) en string i ho enviem 
    conn.send(str.encode(write_pos(pos[player])))
    
    reply=""
    
    while True:
        try:
            #print('què rebo:',conn.recv(2048).decode())#type(conn.recv(2048).decode()))
            
            data2=read_pos((conn.recv(2048)).decode()) #"utf-8" dintre el decode
            #print('data2:',data2, 'type:',type(data2))
            
            #data=read_pos(conn.recv(2048).decode()) #Quantitat d'informació que rebem
            #print('data:',data)
            #El data no em va bé. Deu ser la manera de descodificar (.decode())?
            
            #reply = data.decode("utf-8") #Sempre rebem info encriptada i això és per desencriptar-ho
            pos[player]=data2
            
            if not data2:
                print("Disconnected")
                #currentPlayer-=1
                break
            else:
                if player==1:
                    reply= pos[0]
                else:
                    reply=pos[1]
                #print("Recieved: ",data2)
                #print("Sending: ",reply)
            
            conn.sendall(str.encode(write_pos(reply)))
            
        except:
            break
        
    print("Lost connection")
    conn.close()
    print("------------------------------------------------------")
    print("")
    #currentPlayer-=1
    #print('currentPlayer', currentPlayer)


#El bucle principal que executarà la funció també
while True: #Busquem conneccions contínuament
    conn,addr = s.accept() #Conexió i adreça
    print("Connected to:",addr)
    print('Player',currentPlayer)
    start_new_thread(threaded_client, (conn,currentPlayer%2)) #Si no poso el %2 em posa index out of range
    currentPlayer+=1
    
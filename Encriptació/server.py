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
import pickle
from player import Player

server="192.168.0.23" #Això és per network local, suposo per la meva wifi
port= 5555
s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind((server, port))
    
except socket.error as e:
    str(e)

s.listen(2) #nse si és 1 o 2 per connectar a 2 persones
print("Waiting for a connection, Server Started")


currentPlayer=0 #això és per veure quin jugador utilitzem
'''
def read_pos(str): #rebem string i ho passem a tupla
    str=str.split(",")
    return int(str[0]),int(str[1])
    
def write_pos(tup): #rebem tupla i ho passem a string
    return str(tup[0])+","+str(tup[1])


#El read_pos va bé

#pos=[(100,100),(560,300)]#aquí hi haurà les posicions dels jugadors
#Amb 1.3*win size crec que correspon 1040, 300
'''
players=[Player(0,100,100),Player(1,560,300)]


def threaded_client(conn, player): 
    global currentPlayer
    conn.send(pickle.dumps(players[player]))
    
    reply=""
    
    while True:
        try:           
            data=pickle.loads(conn.recv(2048))
            players[player]=data
            
            if not data:
                print("Disconnected")
                break
            else:
                if player==1:
                    reply=players[0]
                else:
                    reply=players[1]
                #print("Recieved: ",data2)
                #print("Sending: ",reply)
            
            conn.sendall(pickle.dumps(reply))
            
        except:
            break
        
    print("Lost connection")
    conn.close()
    print("------------------------------------------------------")
    print("")
    currentPlayer-=1
    players[0].ready=False
    players[1].ready=False
    #print('currentPlayer', currentPlayer)


#El bucle principal que executarà la funció també
while True: #Busquem conneccions contínuament
    conn,addr = s.accept() #Conexió i adreça
    print("Connected to:",addr)
    
    #gameId=(currentPlayer)//2
    
    if currentPlayer%2 ==0:
        print('Creating a new game...')
        
    else:
        #players[0].ready=True
        #players[1].ready=True        
        print('Ja es pot començar!')
        
        
    print('Player',currentPlayer)
    start_new_thread(threaded_client, (conn,currentPlayer%2)) #Si no poso el %2 em posa index out of range
    currentPlayer+=1
    
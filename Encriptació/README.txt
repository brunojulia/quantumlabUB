README - Encriptació Quàntica

Per exectuar el programa s'ha de tenir instal·lats els programes indicats a QuantumlabUB i és necessari baixar-se tots aquests arxius, excepte la carpeta de "Versions antigues".  El cos del programa és el fitxer que s'anomena "encriptacio.py". 

Per iniciar el programa s'ha de seguir les següents instruccions:

Primer has de buscar l'adreça ip de la teva xarxa. La pots trobar al "Command Prompt" del sistema escrivint:
	ipconfig
S'ha d'agafar el que posa Dirección IPv4. Exemple: 192.168.0.26.
Després s'han d'obrir amb un editor de fitxers de python (per exemple l'Spider) els fitxers "network.py" i "server.py". A les primeres línies d'aquests fitxers hi ha una variable que s'anomena "server" o bé "self.server". Aquesta variable és la que ha de contenir la adreça IP que s'ha aconseguit abans en forma d'string (de manera similar a la que ja hi ha).

A continuació s'ha d'obrir un "Anaconda Prompt". Sempre que se n'obri un, primer s'ha d'activar l'entorn escrivint:
	activate clavsqua_env 
talcom s'ha fet al instal·lar el Kivy. 
S'ha d'executar el programa de "server.py" i a continuació s'ha d'obrir un altre "Anaconda Prompt", tornar a activar l'entorn i finalment executar "encriptacio.py". 

Si es volgués simular un altre jugador, només cal tornar a obrir un altre "Anaconda Prompt" i tornar a executar un altre cop "encriptacio.py" des d'aquest nou.  

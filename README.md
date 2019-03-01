# QuantumlabUB
## Quantum mechanics popularisation project UB

* QuantumlabUB is a project build by undergraduate students in 
Physics from the University of Barcelona. The main goals of the 
project are:

> Develop applications to popularise aspects of quantum 
mechanics for broader audiences. 

> The applications should be realistic quantum mechanical problems, thus 
they involve the numerical solution of the Schrodinger equation in 
different setups. 

> The coding is done under the github platform. Weekly meetings ensure crosscollaboration between the students and constant feedback on the goals. 


* The project has been initiated by Daniel Allepuz and Jan Albert Iglesias 
in Feb 2018 under the supervision of Prof. Muntsa Guilleumas and 
Bruno Julia Diaz from the department of Quantum Physics and Astrophysics 
of the University of Barcelona. It is work done under the subject Practicas 
de Empresa. 

## VERSION 1.0, Jun 2018
---------------------

In its current version we have two separate modules:

- clavsqua.py, which solves the evolution of a gaussian wave packet on a configurable external trap in one dimension. 

- doubleslit.py, which solves the 2D time dependent Schrodinger equation with 
a configurable grid of slits. 



## INSTALLATION:
---------------------

The code is developed in python3 using the kivy.  

We have installed it recently with the following sequence.

1) First install the latest version of Anaconda and add it to path.

2) 
conda create -n clavsqua_env python=3.6.3

activate clavsqua_env

pip install numpy==1.14.0

pip install matplotlib==2.1.2

pip install Kivy==1.10.0

pip install Kivy-Garden==0.1.4

pip install kivy.deps.glew==0.1.9

pip install kivy.deps.gstreamer==0.1.12

pip install kivy.deps.sdl2==0.1.17

garden install matplotlib

## CONTACT:
---------------------
Any comment or suggestions are more than welcome, brunojulia@ub.edu

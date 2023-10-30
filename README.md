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

---------------------

Currently (May 2019) we have several separate modules in two different directories:

### ClavsQua

- clavsqua.py, which solves the evolution of a gaussian wave packet on a configurable external trap in one dimension. 

### doubleslit

- doubleslit.py, which solves the 2D time dependent Schrodinger equation with 
a configurable grid of slits. 

### PiecewisePotential

- piecewise.py, is a game where you have to design the trapping potential for a particle in 1D in such a way that the probability density reaches target values chosen at random. 

### Solitons

- collision_bright_solitons.py, allows you to play with two bright solitons with different velocities and properties in 1D Gross-Pitaevskii equation


## CCCB version, Feb 2019
----------------------------

We have developed two simpler versions of ClavsQua and doubleslit for the Quantica exhibit at CCCB (2019), 
http://www.cccb.org/ca/exposicions/fitxa/quantica/230323

They are in the directory CCCB.

## INSTALLATION:
---------------------

The code is developed in python3 using the kivy.  

We have installed it recently with the following sequence.

1) First install the latest version of Anaconda and add it to path.

2) 
open the Anaconda Prompt

conda create -n clavsqua_env python=3.6.3

activate clavsqua_env

pip install numpy==1.14.0

pip install matplotlib==2.1.2

pip install Kivy==1.10.0

pip install Kivy-Garden==0.1.4

pip install kivy.deps.glew==0.1.9

pip install kivy.deps.gstreamer==0.1.12

pip install kivy.deps.sdl2==0.1.17

garden install matplotlib --kivy

pip install numba

pip install scipy

## CONTACT:
---------------------
Any comment or suggestions are more than welcome, brunojulia@ub.edu

Check the website https://github.com/brunojulia/quantumlabUB

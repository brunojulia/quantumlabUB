# Quantum Computation


## Installation 
---------------------

In order to install this program first you will need the QuantumLabUB python environment _clavsqua_env_ 
<br/>
<br/>

<details>
<summary>See how to create this environment here</summary>
<br>
The code is developed in python3 using the kivy.  

We have installed it recently with the following sequence.


1. First install the latest version of Anaconda and add it to path.


2. Open the Anaconda Prompt


3. Write the following commands:

```
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

pip install numba

pip install scipy
```
---------------------
</details>
<br/>

Once you have the _clavsqua_env_ in your computer, open the Anaconda Prompt and activate the environment with

```
activate clavsqua_env
```


Now complete the installation for this specific project with:

```
pip install qiskit

pip install qiskit[visualization]
```
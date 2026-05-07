#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 21:06:56 2026

@author: miranda

    Clase 6/5, falté. Esto es lo que pasaron las chicas por el grupo.
    
    "Este código es un ejemplo clásico de la generación de un proceso estocástico 
    (o un conjunto de señales aleatorias). Básicamente, estás creando 200 versiones 
    diferentes de un mismo experimento para ver cómo varían entre sí."
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
R = 200
fs = 1000
omega_0 = fs/100  # o el valor que tengas
ts = 1/fs
# Vector fila con los índices 0..N-1

#np.random.uniform entre -2 y 2 para encontrar fr
fr = np.random.uniform(-2,2,R).reshape((R,1))
# Frecuencias para una fila: Ω0 + r * (2π/N)
omega_1 = omega_0 + fr * (fs / N)  # shape (N,)

# Repetir R veces para formar la matriz RxN
omega = np.tile(omega_1, N)  # shape (R, N)

print(omega.shape)  # → (200, 1000)


tt= (np.arange(0,N)*ts).reshape((1, N))
nn = np.tile(tt, (R,1)) 
# ruido
sigma=1
mu=0

na = np.random.normal (mu,sigma,N) #ruido
# Genero la señal 
xx = np.sqrt(2)*np.sin(np.pi*2*omega*nn) + na

plt.plot (xx.T)
#%%
"""
#%%Parametros para la señal 
N   = 1000          # cantidad de muestras
fs  = 1000          # frecuencia de muestreo [Hz]
Vmax = np.sqrt(2)   #varianza normalizada
f0  = fs / N  


#%%Defino mi señal
tt, xx = mi_funcion_sen(vmax=Vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs) 

#%% ruido
sigma=1
mu=0

Rn = np.random.normal (mu,sigma,N) #ruido
#%% Señal con ruido 
yn= xx + Rn
"""

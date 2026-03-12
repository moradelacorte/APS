# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Mar 11 20:26:22 2026

@author: Mora De La Corte

Descripción: 
    En este archivo hay notas de la clase 11/3/26 - Introduccion a Python

    
"""

import numpy as np

c = np.sin(np.pi/2) * 2.0

d = [1, 2.0, 'pepe'] # en una lista puedo mezclar tipos de datos

e = (1, 2.0) # tupla

print(d[0]) # Primer elemento de la lista d
print(d[1])
print(d[-1]) # Ultimo elemento de la lista d


# Definicion de funciones

def mi_funcion_sen(vmax, dc, ff, ph=0, nn, fs):
    # Lo de ph=0 es el valor default
    
    
    return tt, xx

    
"""    
# Anotaciones de clase

Nyquist = Establece una relaicon minima entre la frecuencia de la secuencia 
real y la digitalizada para evitar solapamiento.
Necesito como minimo 2 muestras por ciclo, 
en principio buscamos mayor a 100 y menor a 10k

Ts = 1/fs
f_nyquist = fs/2 # es una frecuencia límite / situación de frontera

por lo tanto ...
Ts_nyquist = 1/2*f_nyquist = Ts/2

# Expresion matematica
x[n] = dc + a*np.sin(wx*t + ph) = dc + a*np.sin(2*np.pi*f0*t + ph)
t sera nuestro tt
tt = es una lista escalada por Ts, el cual va de 0 a N-1 para que alla N muestras


"""



# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Tue Mar 17 14:39:54 2026

@author: Mora De La Corte

Descripción: 
    Probando primeras convoluciones con numpy. Trabajo propio (no de clase)
"""

import numpy as np
import matplotlib.pyplot as plt

# señal de entrada
x = np.array([1, 2, 3, 4])

# respuesta al impulso (filtro)
h = np.array([1, 1, 1])


y = np.convolve(x, h) # ya esta la funcion en numpy

# ejes temporales
n_x = np.arange(len(x)) # Veo cuantos x[n] tengo para saber cuantos n necesito
n_h = np.arange(len(h))
n_y = np.arange(len(y))

""" np.arange(inicio, fin, paso) 
    Por dafault el paso es de 1
    No incluye el final
    Para discreto
    ej: np.arange(0,4) = [0, 1, 2, 3] No incluye el final
        np.arange(0,10,2) = [0, 2, 4, 6, 8]
    
    np.linspace(inicio, fin, cantidad de puntos)
    Incuye el final
    La cantidad de puntos se refiere a cuantos puntos en totas tengo
    Para continuo
    ej: np.linspace(0,10,5) = [0. , 2.5 , 5. , 7.5 , 10.]
"""

# gráficos
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.stem(n_x, x)
plt.title("Señal x[n]")

""" la funcion plt.stem me crea los palitos tipicos de las discretas
    plt.stem(n, x, linefmt='g-', markerfmt='go', basefmt='r-')
    n = eje temporal
    x = funcion de interes
    linefmt = color de los palitos
    markerfmt = puntos
    basefmt = linea de la base

"""

plt.subplot(3,1,2)
plt.stem(n_h, h)
plt.title("Respuesta al impulso h[n]")

plt.subplot(3,1,3)
plt.stem(n_y, y)
plt.title("Convolución y[n]")

plt.tight_layout()
plt.show()
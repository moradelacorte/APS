# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Sun Apr  5 12:13:57 2026

@author: Mora De La Corte

Descripción: 
    
    Trabajo realizado en la clase presencial 01/04/26
    
"""

''' Consigna:
    
    Sea la señal discreta x[n] = 4 + 3*sen(n*pi/2). Se toman N = 8 muestras
    
    a - Sabiendo que la definicion de la DFT: 
        calcule analiticamente los coeficientes de la DFT X[k] para k pertenecinete a {0,1, ... ,7}
    
    b - Grafique el módulo |X[k]| y la fase <X[k]. Verifique en que indice k se concentra la energia 
        y explique su relacion con la frecuencia senoidal y el valor medio.

'''

import numpy as np
import matplotlib.pyplot as plt


N = 8
tt = np.arange(0, N) # índices de tiempo discreto n
xx = 4 + 3*np.sin(tt*np.pi/2)
kk = np.arange(0, N) # índices de frecuencia discreta k

XX = np.fft.fft(xx) # Calculo de la FFT
XXmod = np.abs(XX) # Modulo
XXfase = np.angle(XX) # Fase

plt.figure(1)

plt.subplot(2, 1, 1)
plt.stem(kk, XXmod, linefmt='slateblue', basefmt="royalblue")
plt.title("Módulo |X[k]|")
plt.xlabel("k")
plt.grid()

plt.subplot(2, 1, 2)
plt.stem(kk, XXfase, linefmt='slateblue', basefmt="royalblue")
plt.title("Fase ∠X[k]")
plt.xlabel("k")
plt.grid()

plt.tight_layout()
plt.show()

# Energía espectral
Energia = (1/N**2)*(XXmod**2)

# Se utiliza la normalización de la potencia media para que 
# los picos de energía sean directamente comparables con las 
# amplitudes de la señal original en el tiempo

plt.figure(2)
plt.stem(kk, Energia, linefmt='khaki', basefmt="darkkhaki")
plt.title("Energía espectral |X[k]|²")
plt.xlabel("k")
plt.grid()
plt.tight_layout()
plt.show()
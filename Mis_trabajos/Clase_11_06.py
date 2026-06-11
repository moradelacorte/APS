# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Jun 11 18:53:54 2026

@author: Mora De La Corte

Descripción: 
    Clase 11/6
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
 
b = [1/2,0, 0, 0, -1/2]  # Coeficientes del numerador (X)

# Agrego L ceros y filtro con una pasa bajo que corte en pi/L --> "subimos la frecuencia de muestreo"
# Quien decide que hacer en el medio? la respuesta al impulso del pasa bajo
# Zero padding --> agregar de prepo muestras, la caja es la que se encarga de la interpolacion
# si descartamos muestras? --> bajamos nyquist
# desacartar la fs D veces --> me quedo con fs/D
# que pasa si me paso y descarto de mas? --> alias! Se nos mete en el espectro digital. Alias digital
# Aplico un filtro en pi/D 
# nuestro ECG esta cuentizado en 16 bits
# Siempre limitar la energia de tu señal antes de diesmar

# T(z) --> Tu(z) version interpolada de la transferencia
# si z = Z^L --> efecto neto de interpolar la respuesta al impulso. Version L veces interpolada
# Aumentar L veces implica redefinir el eje x

a = [1]           # Coeficientes del denominador (Y)

ceros, polos, _ = signal.tf2zpk(b, a)

w, H = signal.freqz(b, a, worN=8000)

modulo = np.abs(H)
fase = np.angle(H)  

plt.figure(figsize=(10, 6))

# Gráfico del Módulo
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, modulo, color='tomato', linewidth=2)
plt.title('Respuesta en Frecuencia )', fontsize=14)
plt.ylabel('Módulo |H(e^{j\omega})|', fontsize=12)
plt.grid(True)

# Gráfico de la Fase
plt.subplot(2, 1, 2)
plt.plot(w / np.pi, fase, color='teal', linewidth=2)
plt.xlabel('Frecuencia Normalizada (x \pi rad/muestra)', fontsize=12)
plt.ylabel('Fase \phi(\omega) (rad)', fontsize=12)
plt.grid(True)

plt.tight_layout()


plt.figure(figsize=(6, 6)) 

theta = np.linspace(0, 2*np.pi, 200)
plt.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', label='Circunferencia Unidad')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.scatter(np.real(ceros), np.imag(ceros), s=80, marker='o', facecolors='none', edgecolors='blue', linewidths=2, label='Ceros')
plt.scatter(np.real(polos), np.imag(polos), s=80, marker='x', color='red', linewidths=2, label='Polos')

plt.title('Plano Z: Polos y Ceros', fontsize=14)
plt.xlabel('Parte Real (Re)', fontsize=12)
plt.ylabel('Parte Imaginaria (Im)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.axis('equal')  
plt.xlim([-1.5, 1.5])
plt.ylim([-1.5, 1.5])
plt.legend(loc='upper right')

plt.show()
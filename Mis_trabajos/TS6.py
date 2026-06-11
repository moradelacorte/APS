# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Jun  4 13:25:04 2026

@author: Mora De La Corte

Descripción: 
    TS6 - Simulacion de modulo y fase
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# y(n) = x(n) + x(n-1) + x(n-2) + x(n-3)
b = [1, 1, 1, 1]  # Coeficientes del numerador (X)
a = [1]           # Coeficientes del denominador (Y)

w, H = signal.freqz(b, a, worN=8000)

modulo = np.abs(H)
fase = np.angle(H)  

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, modulo, color='tomato', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema a)', fontsize=14)
plt.ylabel('Módulo |H(e^{j\omega})|', fontsize=12)
plt.grid(True)


plt.subplot(2, 1, 2)
plt.plot(w / np.pi, fase, color='teal', linewidth=2)
plt.xlabel('Frecuencia Normalizada (x \pi rad/muestra)', fontsize=12)
plt.ylabel('Fase \phi(\omega) (rad)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()

#%% item b
 
# y(n) = x(n) + x(n-1) + x(n-2) + x(n-3) + x(n-4)
b = [1, 1, 1, 1, 1]  # Coeficientes del numerador (X)
a = [1]           # Coeficientes del denominador (Y)

ceros, polos, _ = signal.tf2zpk(b, a)

w, H = signal.freqz(b, a, worN=8000)


modulo = np.abs(H)
fase = np.angle(H)  


plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, modulo, color='tomato', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema b)', fontsize=14)
plt.ylabel('Módulo |H(e^{j\omega})|', fontsize=12)
plt.grid(True)


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

#%% item c
 
# y(n) = x(n) - x(n-1)
b = [1, -1]  # Coeficientes del numerador (X)
a = [1]           # Coeficientes del denominador (Y)

ceros, polos, _ = signal.tf2zpk(b, a)

w, H = signal.freqz(b, a, worN=8000)

modulo = np.abs(H)
fase = np.angle(H)  

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, modulo, color='tomato', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema c)', fontsize=14)
plt.ylabel('Módulo |H(e^{j\omega})|', fontsize=12)
plt.grid(True)

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

#%% item d

# y(n) = x(n) - x(n-2)
b = [1, 0, -1]  # Coeficientes del numerador (X)
a = [1]           # Coeficientes del denominador (Y)

ceros, polos, _ = signal.tf2zpk(b, a)

w, H = signal.freqz(b, a, worN=8000)

modulo = np.abs(H)
fase = np.angle(H)

plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, modulo, color='tomato', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema c)', fontsize=14)
plt.ylabel('Módulo |H(e^{j\omega})|', fontsize=12)
plt.grid(True)

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
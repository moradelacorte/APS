# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Jun 24 11:41:47 2026

@author: Mora De La Corte

Descripción: 
    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# T(z) = z^2 + 1 / z^2 - 0.8*z + 0.64

b = [1, 0, 1]  # Coeficientes del numerador (X)
a = [1, -0.8, 0.64]           # Coeficientes del denominador (Y)

# a = [1, -0.8, 0.16] # Polos reales puros en z = 0.4 (doble)

# a = [1, -1.0, 0.16] # Un polo en z = 0.8 y otro en z = 0.2

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

z, p, k = signal.tf2zpk(b, a)
print("ceros (s):", z, " radio:", np.abs(z))
print("polos (s):", p, " radio:", np.abs(p))

plt.figure(figsize=(6, 6))
ax = plt.gca()
plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.title("Re"); plt.ylabel("Img")
plt.axis('equal'); plt.grid(True); plt.legend()

plt.show()
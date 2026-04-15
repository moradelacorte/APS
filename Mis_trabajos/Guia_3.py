# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Apr 15 15:20:17 2026

@author: Mora De La Corte

Descripción: 
    
    Códigos de verificación para los ejercicos de la guia 3. 
    Brindados por los profes
"""

#%% Ejercicio 1

import numpy as np
import matplotlib.pyplot as plt

N = 8
n = np.arange(N)
x = 4 + 3 * np.sin(np.pi * n / 2) # Senal: x[n] = 4 + 3*sin(pi*n/2)

X = np.fft.fft(x, n=N) # DFT de N puntos
k = np.arange(N)

print("x[n]:", np.round(x, 4))
print("\nX[k] (complejo):")

for kk in range(N):
    print(f" k={kk:2d}: {X[kk]: .3f}")

fig, axes = plt.subplots(3, 1, figsize=(8, 7))
axes[0].stem(n, x)
axes[0].set_title("Senal en tiempo: x[n] = 4 + 3*sin(pi*n/2)")
axes[0].set_xlabel("n"); axes[0].set_ylabel("x[n]"); axes[0].grid(True)

axes[1].stem(k, np.abs(X))
axes[1].set_title("DFT (N=8): Magnitud |X[k]|")
axes[1].set_xlabel("k"); axes[1].set_ylabel("|X[k]|"); axes[1].grid(True)

axes[2].stem(k, np.angle(X))
axes[2].set_title("DFT (N=8): Fase angulo(X[k]) [rad]")
axes[2].set_xlabel("k"); axes[2].set_ylabel("Fase [rad]"); axes[2].grid(True)

plt.tight_layout()
plt.show()
# Verificacion: X[0]=32, X[2]=-12j, X[6]=+12j, resto ~0

#%% Ejercicio 2

import numpy as np
import matplotlib.pyplot as plt

L = 3 # longitud del pulso
N_zp = 100000 # zero-padding grande para aproximar la DTFT continua

# Senal: x[n] = delta[n] + delta[n-1] + delta[n-2]
x = np.zeros(N_zp)
x[:L] = 1.0

X = np.fft.fft(x, n=N_zp)
omega_k = 2 * np.pi * np.arange(N_zp) / N_zp

# DTFT teorica para L=3: |1 + e^{-jw} + e^{-j2w}|
omega_c = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
dtft_mag = np.abs(1 + np.exp(-1j * omega_c) + np.exp(-2j * omega_c))

plt.figure(figsize=(8, 4))
plt.plot(omega_c, dtft_mag, label="DTFT teorica (L=3)")
plt.plot(omega_k, np.abs(X), 'o', ms=2, label=f"Muestras DFT (N={N_zp})")
plt.title("DTFT vs DFT con zero-padding")
plt.xlabel("omega [rad]"); plt.ylabel("Magnitud")
plt.legend(); plt.grid(True)
plt.show()

#%%
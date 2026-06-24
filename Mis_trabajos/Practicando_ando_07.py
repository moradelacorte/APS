# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Jun 24 11:41:47 2026

@author: Mora De La Corte

Descripción: 
    Simulacro parcial claude
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

#%% Punto 4

import numpy as np
import matplotlib.pyplot as plt

# 1. Definición de parámetros (Tus valores propuestos)
fs = 20000  # 20 kHz
N = 300
t = np.arange(N) / fs

# 2. Construcción de las señales en base a los dB del enunciado
# 0 dB -> Amplitud = 1.0
# -30 dB -> Amplitud = 10^(-30/20) = 0.0316
# Ruido -50 dB -> Desviación estándar = 10^(-50/20) = 0.00316
f1, f2 = 5000, 5300
A1 = 1.0
A2 = 10**(-30/20)
A_noise = 10**(-50/20)

np.random.seed(42)  # Para que el ruido sea replicable
ruido = np.random.normal(0, A_noise, N)
x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + ruido

# 3. Aplicación de ventanas
win_rect = np.ones(N)
win_hamm = np.hamming(N)

# 4. Cálculo de la FFT
N_fft = 2048  # Zero-padding para suavizar las curvas del espectro
f = np.fft.rfftfreq(N_fft, 1/fs)

X_rect = np.fft.rfft(x * win_rect, N_fft)
X_hamm = np.fft.rfft(x * win_hamm, N_fft)

# Normalización en dB con corrección de ganancia coherente de cada ventana
mag_rect = 20 * np.log10(np.abs(X_rect) / (N / 2))
mag_hamm = 20 * np.log10(np.abs(X_hamm) / (np.sum(win_hamm) / 2))

# 5. Gráfico de resultados
plt.figure(figsize=(10, 5))
plt.plot(f, mag_rect, label='Ventana Rectangular (Falla)', alpha=0.7)
plt.plot(f, mag_hamm, label='Ventana Hamming (Éxito)', linewidth=2)

# Líneas guía de las frecuencias reales
plt.axvline(f1, color='red', linestyle='--', alpha=0.5, label='5 kHz (0 dB)')
plt.axvline(f2, color='green', linestyle='--', alpha=0.5, label='5.3 kHz (-30 dB)')

plt.xlim(4000, 6000)  # Hacemos zoom en la zona de interés
plt.ylim(-60, 10)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (dB)')
plt.title('Simulación de Estimación Espectral (Problema 4)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
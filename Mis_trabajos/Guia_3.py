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

#%% Ejercico 3

# Código propio

import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
fs = 1000          # Frecuencia de muestreo (Hz)
N = 100            # Número de muestras
df = fs / N        # Resolución: 10 Hz por bin

n = np.arange(N)   # Vector de tiempo discreto
k = np.arange(N)   # Vector de índices de frecuencia (bins)
freqs = k * df     # Frecuencias físicas correspondientes a cada bin

# Caso a) Señal de 200 Hz
x1 = np.cos(2 * np.pi * 200 * n / fs)
X1 = np.fft.fft(x1)

# Caso b) Señal de 205 Hz
x2 = np.cos(2 * np.pi * 205 * n / fs)
X2 = np.fft.fft(x2)

# Graficación
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Gráfico Caso A
ax1.stem(freqs[:50], np.abs(X1[:50]), linefmt='b-', markerfmt='bo', basefmt='r-')
ax1.set_title('Caso A: Señal de 200 Hz (Múltiplo exacto de $\Delta f = 10$ Hz)')
ax1.set_ylabel('Magnitud')
ax1.grid(True, alpha=0.3)
ax1.annotate('Energía concentrada\nen bin 20', xy=(200, 50), xytext=(250, 45),
             arrowprops=dict(facecolor='black', shrink=0.05))

# Gráfico Caso B
ax2.stem(freqs[:50], np.abs(X2[:50]), linefmt='g-', markerfmt='go', basefmt='r-')
ax2.set_title('Caso B: Señal de 205 Hz (Fuga Espectral - Spectral Leakage)')
ax2.set_xlabel('Frecuencia [Hz]')
ax2.set_ylabel('Magnitud')
ax2.grid(True, alpha=0.3)
ax2.annotate('Energía filtrada\nentre bins vecinos', xy=(205, 32), xytext=(250, 35),
             arrowprops=dict(facecolor='black', shrink=0.05))

# SUBIR EL LÍMITE DE Y para que no choque la flecha
ax1.set_ylim([0, 55]) 
ax2.set_ylim([0, 55])

plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Parámetros del problema
fs = 1000          # Frecuencia de muestreo (Hz)
N = 100            # Número de muestras
df = fs / N        # Resolución: 10 Hz por bin

n = np.arange(N)   # Vector de tiempo discreto
k = np.arange(N)   # Vector de índices de frecuencia (bins)
freqs = k * df     # Frecuencias físicas correspondientes a cada bin

# Señal original (205 Hz)
x2 = np.cos(2 * np.pi * 205 * n / fs)
X2 = np.fft.fft(x2) # Caso Rectangular (sin ventana explícita)

# --- INICIO RESOLUCIÓN EJERCICIO 4 ---

# a) Multiplicación por ventana de Hamming
ventana = np.hamming(N)
x2_ventaneada = x2 * ventana

# b) Cálculo de la FFT de la señal ventaneada
X2_ventaneada = np.fft.fft(x2_ventaneada)

# --- GRAFICACIÓN COMPARATIVA ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Gráfico Caso Rectangular
ax1.stem(freqs[:50], np.abs(X2[:50]), linefmt='g-', markerfmt='go', basefmt='r-')
ax1.set_title('Caso Rectangular: FFT de señal de 205 Hz (Sin ventana explícita)')
ax1.set_ylabel('Magnitud')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 55])

# Gráfico Caso con Ventana de Hamming
ax2.stem(freqs[:50], np.abs(X2_ventaneada[:50]), linefmt='m-', markerfmt='mo', basefmt='r-')
ax2.set_title('Caso Ventaneado: FFT de señal de 205 Hz multiplicada por Ventana de Hamming')
ax2.set_xlabel('Frecuencia [Hz]')
ax2.set_ylabel('Magnitud')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 55])

plt.tight_layout()
plt.show()

# --- FFT con zero-padding para ver envolvente tipo sinc ---

Nfft = 4096  # mucho mayor que N

# FFTs con zero-padding
X2_zp = np.fft.fft(x2, Nfft)
X2_vent_zp = np.fft.fft(x2_ventaneada, Nfft)

# Eje de frecuencias
freqs_zp = np.linspace(0, fs, Nfft)

# Normalización en dB
eps = 1e-12
X2_zp_dB = 20 * np.log10(np.abs(X2_zp) / np.max(np.abs(X2_zp)) + eps)
X2_vent_zp_dB = 20 * np.log10(np.abs(X2_vent_zp) / np.max(np.abs(X2_vent_zp)) + eps)

# --- Gráficos tipo "curva continua" ---

plt.figure(figsize=(10,6))

plt.plot(freqs_zp[:Nfft//2], X2_zp_dB[:Nfft//2], label='Rectangular')
plt.plot(freqs_zp[:Nfft//2], X2_vent_zp_dB[:Nfft//2], label='Hamming')

plt.title('Comparación espectral (con zero-padding)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True, alpha=0.3)
plt.legend()

plt.ylim([-80, 5])

plt.show()


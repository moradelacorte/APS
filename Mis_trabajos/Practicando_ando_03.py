# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu May  7 14:35:37 2026

@author: Mora De La Corte

Descripción: 
    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, freqz

# 1. Configuración de parámetros
N = 51  # Largo de la ventana
fft_size = 2048  # zero-padding
win_names = ['boxcar', 'hamming', 'hann', 'blackman', 'flattop']
labels = ['Rectangular', 'Hamming', 'Hann', 'Blackman', 'flattop']
colors = ['#1f77b4', '#7cfc00', '#ffbc00', '#ff0000', 'pink'] # Colores similares a tu imagen

plt.figure(figsize=(12, 7))

# 2. Generación y Procesamiento
for i, win_type in enumerate(win_names):
    # Generar ventana
    win = windows.get_window(win_type, N)
    
    # Calcular respuesta en frecuencia
    w, h = freqz(win, a=1, worN=fft_size, whole=True)
    
    # Centrar en 0 (de -pi a pi) y normalizar
    w = np.fft.fftshift(w)
    w = np.where(w > np.pi, w - 2*np.pi, w)
    h = np.fft.fftshift(h)
    
    # Magnitud en dB normalizada al máximo (0 dB)
    mag = 20 * np.log10(np.abs(h) / np.max(np.abs(h)))
    
    # 3. Graficar
    plt.plot(w, mag, label=labels[i], color=colors[i], lw=1.5)

# 4. Estética del gráfico
plt.title('Comparación de Ventanas Espectrales (Respuesta en Frecuencia)', fontsize=14)
plt.xlabel('Frecuencia Angular $\omega$ [rad/muestra]')
plt.ylabel('Magnitud Normalizada [dB]')
plt.ylim(-80, 5) # Rango de tu imagen
plt.xlim(-np.pi, np.pi)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Formatear el eje X para mostrar múltiplos de pi
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])

plt.tight_layout()
plt.show()

#%% 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows, freqz

# 1. Configuración de la señal
N = 51 
n = np.arange(N)
omega_0 = np.pi / 2  # Frecuencia del coseno (90 grados)
x = np.cos(omega_0 * n)

# Parámetros de visualización
fft_size = 2048
win_names = ['boxcar', 'hamming', 'hann', 'blackman']
labels = ['Rectangular', 'Hamming', 'Hann', 'Blackman']
colors = ['#1f77b4', '#7cfc00', '#ffbc00', '#ff0000']

plt.figure(figsize=(12, 7))

# 2. Procesamiento de cada ventana aplicada al coseno
for i, win_type in enumerate(win_names):
    # Generar ventana
    win = windows.get_window(win_type, N)
    
    # APLICAR VENTANA: Multiplicación punto a punto en el tiempo
    x_windowed = x * win
    
    # Calcular espectro (usamos freqz para evaluar la DTFT de la señal resultante)
    w, h = freqz(x_windowed, a=1, worN=fft_size, whole=True)
    
    # Centrar y normalizar
    w = np.fft.fftshift(w)
    w = np.where(w > np.pi, w - 2*np.pi, w)
    h = np.fft.fftshift(h)
    
    # Magnitud en dB (normalizada a 0dB para ver la forma)
    mag = 20 * np.log10(np.abs(h) / np.max(np.abs(h)))
    
    plt.plot(w, mag, label=labels[i], color=colors[i], lw=1.5)

# 3. Estética del gráfico
plt.title(r'Espectro de un Coseno Ventaneado $|\hat{X}(\omega)|_{dB}$', fontsize=14)
plt.xlabel('Frecuencia Angular $\omega$')
plt.ylabel('Magnitud [dB]')
plt.ylim(-80, 5)
plt.xlim(-np.pi, np.pi)
plt.grid(True, linestyle='--', alpha=0.5)
plt.axvline(x=omega_0, color='gray', linestyle=':', alpha=0.5) # Línea en +pi/2
plt.axvline(x=-omega_0, color='gray', linestyle=':', alpha=0.5) # Línea en -pi/2
plt.legend(loc='center right')

plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           ['$-\pi$', '$-\pi/2$', '0', '$\pi/2$', '$\pi$'])

plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Apr 23 18:25:54 2026

@author: Mora De La Corte

Descripción: 
    Apunte de clase virtual - continuacion de la TS3
"""

import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.linspace(0, (nn-1)*ts, nn)
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    return tt, xx

fs = 1000
N = 100 
vmax = np.sqrt(2)

k_values = [N/4]
labels = ['k = N/4']
markers = ['o']

#k_values = [N/4, N/4 + 0.25, N/4 + 0.5]
#labels = ['k = N/4', 'k = N/4 + 0.25', 'k = N/4 + 0.5']
#markers = ['o', 's', '^']


# --- Seccion 1: Sin zero padding ---
fig1, ax1 = plt.subplots(1, 1, figsize=(10, 4))
delta_f = fs / N
pos_freqs = np.arange(N//2) * delta_f

for k, label, marker in zip(k_values, labels, markers):
    f0 = k * (fs / N)
    tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs)

    XX = np.fft.fft(xx) / N
    psd_XX = 10 * np.log10(2 * np.abs(XX[:N//2])**2)

    ax1.plot(pos_freqs, psd_XX, marker, label=label)

ax1.set_xlabel('Frecuencia [Hz]')
ax1.set_ylabel('Densidad de Potencia [dB]')
ax1.set_title('Sin zero padding')
ax1.legend(loc='upper right')

# --- Seccion 2: Con zero padding ---
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 4))
N_zp = 9 * N
delta_f_zp = fs / N_zp
pos_freqs_zp = np.arange(N_zp//2) * delta_f_zp

for k, label, marker in zip(k_values, labels, markers):
    f0 = k * (fs / N)
    tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs)

    xx_zp = np.concatenate([xx, np.zeros(N_zp - N)])

    XX_zp = np.fft.fft(xx_zp) / N   
    psd_XX_zp = 10 * np.log10(2 * np.abs(XX_zp[:N_zp//2])**2)

    ax2.plot(pos_freqs_zp, psd_XX_zp, marker, label=label, markersize=3)

ax2.set_xlabel('Frecuencia [Hz]')
ax2.set_ylabel('Densidad de Potencia [dB]')
ax2.set_title(f'Con zero padding (N_zp  = {N_zp})')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- Seccion 3: np.tile (repeticion de la señal) ---
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4))
N_tile = 9 * N
delta_f_tile = fs / N_tile
pos_freqs_tile = np.arange(N_tile//2) * delta_f_tile

for k, label, marker in zip(k_values, labels, markers):
    f0 = k * (fs / N)
    tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs)

    xx_tiled = np.tile(xx, 9)

    XX_tile = np.fft.fft(xx_tiled) / N
    psd_XX_tile = 10 * np.log10(2 * np.abs(XX_tile[:N_tile//2])**2)

    ax3.plot(pos_freqs_tile, psd_XX_tile, marker, label=label, markersize=3)

ax3.set_xlabel('Frecuencia [Hz]')
ax3.set_ylabel('Densidad de Potencia [dB]')
ax3.set_title('Con np.tile')
ax3.legend(loc='upper right')

# Para limitar el eje x

ax1.set_xlim(200, 300)
ax2.set_xlim(200, 300)
ax3.set_xlim(200, 300)

plt.tight_layout()
plt.show()


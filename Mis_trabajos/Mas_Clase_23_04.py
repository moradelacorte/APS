# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Apr 23 19:56:14 2026

@author: Mora De La Corte

Descripción: 
    Apunte de clase virtual - continuacion de la TS3
    Pasado en limpio el codigo anterior (el de Clase_23_04.py) + grafico de la fase + Pasreval
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

k_values = [N/4, N/4 + 0.25, N/4 + 0.5]
labels = ['k = N/4', 'k = N/4 + 0.25', 'k = N/4 + 0.5']
markers = ['o', 's', '^']

N_zp = N + (9 * N) # el N para el zero padding pedido por la consigna
delta_f = fs / N
delta_f_zp = fs / N_zp

# --- Figure 1: Sin ZP vs Con ZP, un subplot por cada k ---
fig1, axes = plt.subplots(1, 3, figsize=(18, 5))
fig1.suptitle('Sin Zero Padding vs Con Zero Padding', fontsize=14)

for ax, k, label, marker in zip(axes, k_values, labels, markers):
    f0 = k * (fs / N)
    tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs)

    # Sin zero padding
    pos_freqs = np.arange(N//2) * delta_f
    XX = np.fft.fft(xx) / N
    psd_XX = 10 * np.log10(2 * np.abs(XX[:N//2])**2)
    ax.plot(pos_freqs, psd_XX, marker, label='Sin ZP', color='blue', zorder=5)

    # Con zero padding
    pos_freqs_zp = np.arange(N_zp//2) * delta_f_zp
    xx_zp = np.concatenate([xx, np.zeros(N_zp - N)])
    XX_zp = np.fft.fft(xx_zp) / N
    psd_XX_zp = 10 * np.log10(2 * np.abs(XX_zp[:N_zp//2])**2)
    ax.plot(pos_freqs_zp, psd_XX_zp, marker, label='Con ZP', color='orange', zorder=3)

    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('Densidad de Potencia [dB]')
    ax.set_title(label)
    ax.set_xlim(200, 300)
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- Figure 2: Fase Sin ZP vs Con ZP ---
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle('Fase: Sin Zero Padding vs Con Zero Padding', fontsize=14)


for ax, k, label, marker in zip(axes2, k_values, labels, markers):
    f0 = k * (fs / N)
    tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs)

    # Sin zero padding
    XX = np.fft.fft(xx) / N
    modulo_dB = 10 * np.log10(2 * np.abs(XX[:N//2])**2)
    fase_XX = np.angle(XX[:N//2])
    pos_freqs = np.arange(N//2) * delta_f

    ax.plot(pos_freqs, fase_XX, marker, label='Sin ZP', color='blue', zorder=5)

    # Con zero padding
    xx_zp = np.concatenate([xx, np.zeros(N_zp - N)])
    XX_zp = np.fft.fft(xx_zp) / N
    modulo_dB_zp = 10 * np.log10(2 * np.abs(XX_zp[:N_zp//2])**2)
    fase_XX_zp = np.angle(XX_zp[:N_zp//2])
    pos_freqs_zp = np.arange(N_zp//2) * delta_f_zp

    ax.plot(pos_freqs_zp, fase_XX_zp, '-', label='Con ZP', color='orange', zorder=3)

    ax.set_xlabel('Frecuencia [Hz]')
    ax.set_ylabel('Fase [rad]')
    ax.set_title(label)
    #ax.set_xlim(200, 300)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

# --- Lo de Parseval---
print("VERIFICACIÓN IDENTIDAD DE PARSEVAL")

for k, label in zip(k_values, labels):
    f0 = k * (fs / N)
    tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs)
    
    # Potencia = varianza (por el dc=0)
    potencia_tiempo = np.mean(xx**2)

    XX = np.fft.fft(xx) / N 
    # Potencia frecuencia 
    potencia_freq = N * np.sum(np.abs(XX)**2)
    
    # solo frecuencias positivas
    XX_raw = np.fft.fft(xx)
    potencia_parseval = (1/N**2) * (
        np.abs(XX_raw[0])**2 +
        2 * np.sum(np.abs(XX_raw[1:N//2])**2) +
        np.abs(XX_raw[N//2])**2
    )
    
    print(f"\n{label}  →  f0 = {f0:.2f} Hz")
    print(f"  Potencia dominio tiempo  : {potencia_tiempo:.6f}")
    print(f"  Parseval (todos los bins): {potencia_freq/N:.6f}")   
    print(f"  Parseval (frec. pos.)    : {potencia_parseval:.6f}")
    print(f"  Potencia teórica (vmax²/2): {vmax**2/2:.6f}")
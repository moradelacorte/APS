# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Apr 22 18:26:38 2026

@author: Mora De La Corte

Descripción: 
    TS3
"""
import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.linspace(0, (nn-1)*ts, nn)
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    return tt, xx

fs = 1000
N = 100*9
k = N/4# [N/4, N/4 + 0.25, N/4 + 0.5]
f0 = k * (fs/N)          
vmax = np.sqrt(2)


tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=f0, ph=0, nn=N, fs=fs)

XX  = np.fft.fft(xx)  / N

freqs     = np.fft.fftfreq(N, 1/fs)
pos_freqs = freqs[:N//2]


psd_XX  = 10 * np.log10(2 * np.abs(XX [:N//2])**2 )

plt.figure(1)
plt.plot(pos_freqs, psd_XX, 'o')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.legend(loc='upper right')


plt.show()
"""
kk = np.arange(start=-200, stop=200, step=1)
n0 = 2

kernel_dirchlet = n0 *np.sin(np.pi * kk * n0) / np.sin(np.pi * kk / N)

plt.figure(2)
plt.plot(kk, np.abs(kernel_dirchlet))
plt.plot(np.array([-10,10]), np.array([0,0]), ':k')  

plt.show()

"""

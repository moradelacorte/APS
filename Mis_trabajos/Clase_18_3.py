# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Mar 18 20:55:38 2026

@author: Mora De La Corte

Descripción: 
    
"""

# Bloque para crear la función pedida
import numpy as np

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):

    # tiempo de muestreo
    ts = 1/fs
    
    # vector de tiempo
    tt = np.linspace(0, (nn-1)*ts, nn).reshape(nn,1) # El reshape es para que sean vectores de Nx1
    
    # señal senoidal
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    
    return tt, xx


vmax = np.sqrt(2)
ff = 1
N = 100
fs = 100
snr = 20 #SNR=20dB: señal 100 veces más potente que el ruido

tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs)

ps = np.var(xx) # Uso la variancia para calcular la potencia de la señal
pr = 10**-(snr/10) # Despejando de la ecuacion para SNR_dB

ruido = np.random.normal(0,np.sqrt(pr),(N,1))

yy = xx + ruido

import matplotlib.pyplot as plt

plt.plot(tt,xx,label="señal limpia")
plt.plot(tt,yy,label="señal con ruido")

plt.xlabel("tiempo")
plt.ylabel("Voltaje")
plt.title("Senoidal con ruido")
plt.legend()

plt.show()

print('Potencia de la señal: ', ps)
print('Potencia del ruido: ', pr)


from scipy import signal as sig

n0 = 10 #muestras
dd = np.zeros(N)
dd[n0] = 1.

hh = sig.convolve(ruido, np.flip(ruido)) # autocorrelacion

#cuatizacion

B = 4 #bits
Vfs = 3 #volts

qq = Vfs/2**B

xxq = np.round(xx/qq)

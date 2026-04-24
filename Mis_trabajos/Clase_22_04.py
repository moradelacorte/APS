# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Apr 22 18:26:38 2026

@author: Mora De La Corte

Descripción: 
    TS3 - visto en clase 
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

#%% codigo que pasaron por telegram


import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft


def sen(ff, nn, vmax=1, dc=0, ph=0, fs=2):
    n = np.arange(0, nn)
    tt = n/fs
    w0 = 2 * np.pi * ff
    xx = dc + vmax * np.sin(w0 * tt + ph)
    return tt, xx


N = 1000
Fs = N
df = Fs/N
vmx= np.sqrt(2)
# Definimos las tres frecuencias pedidas
f1 = (N/4) * df
f2 = (N/4 + 0.25) * df
f3 = (N/4 + 0.5) * df

# Señales
_, x1 = sen(ff=f1, nn=N, vmax= vmx,fs=Fs)
_, x2 = sen(ff=f2, nn=N, vmax= vmx, fs=Fs)
_, x3 = sen(ff=f3, nn=N, vmax= vmx, fs=Fs)


# FFT y PSD
X1 = fft(x1)
X2 = fft(x2)
X3 = fft(x3)

PSD1 = (np.abs(X1)/N)**2
PSD2 = (np.abs(X2)/N)**2
PSD3 = (np.abs(X3)/N)**2

frec = np.arange(N) * df

frec_half = frec[:N//2 + 1]



# PSD unilateral
PSD1_uni = PSD1[:N//2 + 1].copy()
PSD1_uni*= 2

PSD2_uni = PSD2[:N//2 + 1].copy()
PSD2_uni *= 2


PSD3_uni = PSD3[:N//2 + 1].copy()
PSD3_uni *= 2


# PSD en Db
PSD1_uni_db = 10 * np.log10(PSD1_uni + 1e-20)


PSD2_uni_db = 10 * np.log10(PSD2_uni + 1e-20)


PSD3_uni_db = 10 * np.log10(PSD3_uni + 1e-20)

# a) Graficar las PSD
plt.figure()
plt.plot(frec_half, PSD1_uni,':x', label="k0 = N/4")
plt.plot(frec_half, PSD2_uni,':x', label="k0 = N/4 + 0.25")
plt.plot(frec_half, PSD3_uni,':x', label="k0 = N/4 + 0.5")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad Espectral de Potencia")
plt.title("PSD de senoidales con ligera desintonía")
plt.legend()
plt.grid(True)
plt.show()



plt.figure()
plt.plot(frec_half, PSD1_uni_db,':x', label="k0 = N/4")
plt.plot(frec_half, PSD2_uni_db,':x',label="k0 = N/4 + 0.25")
plt.plot(frec_half, PSD3_uni_db,':x', label="k0 = N/4 + 0.5")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad Espectral de Potencia dB")
plt.title("PSD de senoidales con ligera desintonía")
plt.legend()
plt.grid(True)
plt.show()
#

# c) Zero padding
Npad = 9* N

# Hacemos zero-padding sobre x3, así coincide con f3
x3_pad = np.pad(x3, (0, Npad - N))

# FFT con zero-padding
X3z = fft(x3_pad)

# Eje de frecuencias
frec_pad = np.arange(Npad) * Fs / Npad
frec_pad_half = frec_pad[:Npad//2 + 1]

# PSD unilateral
PSD3z = (np.abs(X3z) / N)**2
PSD3z_uni = PSD3z[:Npad//2 + 1].copy()
PSD3z_uni[1:-1] *= 2


PSD3z_uni_db = 10 * np.log10(PSD3z_uni + 1e-20)



# Gráfico comparativo
plt.figure(figsize=(10,5))
plt.plot(frec_half, PSD3_uni_db, ':x', label="Sin zero-padding")
plt.plot(frec_pad_half, PSD3z_uni_db, ':x', label="Con zero-padding (9N)")

plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad Espectral de Potencia")
plt.title("Efecto del Zero-padding en la PSD")
plt.legend()
plt.grid(True)
plt.show()


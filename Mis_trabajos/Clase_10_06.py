# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Jun 10 18:26:09 2026

@author: Mora De La Corte

Descripción: 
    FIR por Cuadrados Mínimos y Parks-Mc Clellan-Remez.
    hicieron una plantilla para 2000 coef
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

fs = 1000 # fs_ecg 
 
nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = 0.2
wp1 = 0.5
wp2 = 35
ws2 = 37

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ])
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

desired = np.array([0, 0, 1, 1, 0, 0])

# Aplicando filtro IIR------------------------------------------------------------

wp = [wp1, wp2]
ws = [ws1, ws2]

numtaps = 1701
weight = np.array([2,1,1])

h_firls = sig.firls(numtaps=numtaps, bands=frecs, desired=desired, weight=weight, fs=fs)

ww = np.concat([np.logspace(start=-2,stop = 0.1, num = 500),
                np.linspace(start = 1.26,stop = 35, num = 200),
                np.logspace(start=1.55,stop = 1.65, num = 300),
                np.linspace(start = 46,stop = fs//2, num = 50)])

w, h = sig.freqz(h_firls, worN=ww, fs=fs)

# Grafica plantilla vs filtro -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(w, 20 * np.log10(np.abs(h)), label='FIR firls', color='C0', linewidth=2)

frecs_hz = (frecs * nyq_frec) #Pasamos frecs de nuevo a Hz

plt.scatter(frecs_hz, gains_db, color='red', zorder=5, label='Puntos de la Plantilla')
plt.plot(frecs_hz, gains_db, color='red', linestyle=':', alpha=0.7, label='Forma aproximada')

plt.title('Verificación: firls')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud (dB)')

# zoom en la zona de interés (0 a 60 Hz) porque ws2 es 45 Hz.
plt.xlim([0, 100]) 
plt.ylim([-100, 50]) # De -50 dB a 5 dB para ver bien la caída y el ripple

# 1. Banda de parada inferior (De 0 Hz a ws1): 
# La ganancia NO puede estar por encima de -40 dB.
plt.fill_between([0, ws1], -atenuacion, 5, color='red', alpha=0.15, label='Zonas Prohibidas')

# 2. Banda de paso (De wp1 a wp2): 
# La ganancia NO puede caer por debajo de -1 dB.
plt.fill_between([wp1, wp2], -50, -ripple, color='red', alpha=0.15)


# 3. Banda de parada superior (De ws2 en adelante): 
# La ganancia NO puede estar por encima de -40 dB. 
plt.fill_between([ws2, 60], -atenuacion, 5, color='red', alpha=0.15)

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.show()

#%% Parks-Mc Clellan-Remez.


fs = 1000 # fs_ecg 
 
nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = 0.2
wp1 = 0.5
wp2 = 35
ws2 = 36

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ])
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

desired = np.array([0, 1, 0])


wp = [wp1, wp2]
ws = [ws1, ws2]

numtaps = 2001
weight = np.array([1, 1, 1])

h_remez = sig.remez(numtaps=numtaps, bands=frecs, desired=desired, weight=weight, type='bandpass', fs=fs)

w, h = sig.freqz(h_remez, worN=ww, fs=fs)

plt.figure(figsize=(10, 5))

plt.plot(w, 20 * np.log10(np.abs(h)), label='FIR firls', color='C0', linewidth=2)

frecs_hz = (frecs * nyq_frec) #Pasamos frecs de nuevo a Hz

plt.scatter(frecs_hz, gains_db, color='red', zorder=5, label='Puntos de la Plantilla')
plt.plot(frecs_hz, gains_db, color='red', linestyle=':', alpha=0.7, label='Forma aproximada')

plt.title('Verificación: firls')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud (dB)')

# zoom en la zona de interés (0 a 60 Hz) porque ws2 es 45 Hz.
plt.xlim([0, 100]) 
plt.ylim([-100, 50]) # De -50 dB a 5 dB para ver bien la caída y el ripple

# 1. Banda de parada inferior (De 0 Hz a ws1): 
# La ganancia NO puede estar por encima de -40 dB.
plt.fill_between([0, ws1], -atenuacion, 5, color='red', alpha=0.15, label='Zonas Prohibidas')

# 2. Banda de paso (De wp1 a wp2): 
# La ganancia NO puede caer por debajo de -1 dB.
plt.fill_between([wp1, wp2], -50, -ripple, color='red', alpha=0.15)


# 3. Banda de parada superior (De ws2 en adelante): 
# La ganancia NO puede estar por encima de -40 dB. 
plt.fill_between([ws2, 60], -atenuacion, 5, color='red', alpha=0.15)

plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.show()

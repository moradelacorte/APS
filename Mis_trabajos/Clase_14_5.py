# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:49:50 2026

@author: ECyT
 la flattop no es una ventana para buscar estimacion espectral
 mantener siempre la misma ventana 
 scipy.signal.periodogram(x,fs,window,detrend con orden 1 = False (si fuera con 
 ruido usar true),scaling = density (NO TOCAR), axis)
 
 welch(promedio = media)
 
 probabilidad acumulada integral izquierda(suma acumulativa) de una funcion densidad
 veo cuando acumulo hasta saturar --> establesco un umbral ej: 95% de la potencia, 
                                      5% lo explican el ruido. 
                                      si no tiene ruido, acumulo el 100? nunca en la 
                                      practica sigo teniendo ruido despreciable
 probrar con varias ventanas y elegir una para cada caso
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

#%% ECG sin ruido -------------------------------------------------------------

fs_ecg = 1000

ecg_one_lead = np.load('ecg_sin_ruido.npy')

# Visualizano la señal

# tiempo = número_de_muestra / fs
tiempo = np.arange(len(ecg_one_lead)) / fs_ecg

plt.figure()
plt.title('ECG sin ruido')
plt.plot(tiempo, ecg_one_lead)
plt.xlabel('Tiempo [s]')       
plt.ylabel('Amplitud [u.a.]')      # <--- Unidades arbitrarias/digitales
plt.grid(True)                
plt.show()

# Aplicando ventana de welch

nperseg_test = 1024 # potencias de 2 (por la FFT), 256 por defecto
nfft_test = nperseg_test * 2 # Zero-padding
'''
    El tamaño de la ventana esta dado por  T = nperseg / fs
    Como fs = 1000, si nperseg = 1024 --> T aprox = 1 segundo. Esto es ideal para ECG 
    porque captura la estructura completa de un latido (la onda P, el complejo QRS 
    y la onda T)
'''

f, Pxx_den = sig.welch(ecg_one_lead, fs_ecg, nperseg = nperseg_test, nfft = nfft_test)
Pxx_den_log = 10*np.log10(Pxx_den)

plt.figure()
plt.title(f'PSD de ECG sin ruido. nperseg = {nperseg_test}')
plt.plot(f, Pxx_den_log)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# Calculo del ancho de banda

potencia_total = np.sum(Pxx_den_log)
umbral_95 = 0.95 * potencia_total



#%% Lectura de pletismografía (PPG) -------------------------------------------

fs_ppg = 400 # Hz

# PPG con ruido ---------------------------------------------------------------

ppg_cr = np.genfromtxt('PPG.csv', delimiter=',', skip_header=1)

# Visualizano la señal
plt.figure()
plt.title('PPG con ruido')
plt.plot(ppg_cr)

# Aplicando ventana de welch

nperseg = 2048
'''
    Nos interesa estudiar la tendencia de los latidos, por lo que necesitamos una ventana
    que nos permita ver más de un latido, es decir que T sea > 1 segundo.
    solapamiento estandar = 50% (nperseg//2)
    45000 muestras 
    --> nperseg = 2048 / segmentos = 43
    --> nperseg = 4096 / segmentos = 21
    --> nperseg = 8192 / segmentos = 10
    Mientras mas granda el nperseg aumento la resolcion teorica matematica, pero cada vez
    tengo menos segmentos para trabajar; deformo la señal
'''
f, Pxx_den = sig.welch(ppg_cr, fs_ppg, nperseg=nperseg)
Pxx_den_log = 10*np.log10(Pxx_den)

plt.figure()
plt.title(f"PDS de PPG con ruido. nperseg = {nperseg}")
plt.plot(f, Pxx_den_log)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

# PPG sin ruido ---------------------------------------------------------------

ppg_sr = np.load('ppg_sin_ruido.npy')

# Visualizano la señal
plt.figure()
plt.title('PPG sin ruido')
plt.plot(ppg_sr)

# Aplicando ventana de welch

nperseg = 2048

f, Pxx_den = sig.welch(ppg_sr, fs_ppg, nperseg=nperseg)
Pxx_den_log = 10*np.log10(Pxx_den)

plt.figure()
plt.title(f"PDS de PPG sin ruido. nperseg = {nperseg}")
plt.plot(f, Pxx_den_log)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
#%% La cucaracha --------------------------------------------------------------

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')

# Visualizano la señal
plt.figure()
plt.title('La cucaracha')
plt.plot(wav_data)

nperseg = 4096
'''
    Si nos fijamos en la ventana de explorador de variables vemos que fs_audio = 48000 Hz
    Por lo que nuestra ventana queda T = nperseg / 48000
    El estandar en audio es usar ventanas de 40 ms y 100 ms, de ahi despejamos el valor
    para nperseg
'''
f, Pxx_den = sig.welch(wav_data, fs_audio, nperseg=nperseg)
Pxx_den_log = 10*np.log10(Pxx_den)

plt.figure()
plt.title(f"PSD de 'La cucaracha'. nperseg = {nperseg}")
plt.plot(f, Pxx_den_log)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#%% Silbido -------------------------------------------------------------------

fs_audio, wav_data = sio.wavfile.read('silbido.wav')

# Visualizano la señal
plt.figure()
plt.title('Silbido')
plt.plot(wav_data)

nperseg = 4096

f, Pxx_den = sig.welch(wav_data, fs_audio, nperseg=nperseg)
Pxx_den_log = 10*np.log10(Pxx_den)

plt.figure()
plt.title(f"PSD de 'Silbido'. nperseg = {nperseg}")
plt.plot(f, Pxx_den_log)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

#%% Prueba psd ----------------------------------------------------------------

fs_audio, wav_data = sio.wavfile.read('prueba psd.wav')

# Visualizano la señal
plt.figure()
plt.title('prueba psd')
plt.plot(wav_data)

nperseg = 4096

f, Pxx_den = sig.welch(wav_data, fs_audio, nperseg=nperseg)
Pxx_den_log = 10*np.log10(Pxx_den)

plt.figure()
plt.title(f"PSD de 'Prueba psd'. nperseg = {nperseg}")
plt.plot(f, Pxx_den_log)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()


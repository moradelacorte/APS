# -*- coding: utf-8 -*-
"""
Created on Thu May 14 20:49:50 2026

@author: ECyT
 la flattop no es una ventana para buscar estimacion espectral
 mantener siempre la misma ventana 
 scipy.signal.periodogram(x,fs,window,detrend con orden 1 = False (si fuera con 
 ruido usar true),scaling = density (NO TOCAR), axis)
 
 welch(promedio = media)
"""

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

fs_ecg = 1000

##################
## ECG sin ruido
##################

ecg_one_lead = np.load('ecg_sin_ruido.npy')


f, Pxx_den = signal.welch(ecg_one_lead, fs_ecg)
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write


##################
## ECG sin ruido
##################

fs_audio, wav_data = sio.wavfile.read('la cucaracha.wav')
nperseg = 20000

f, Pxx_den = sig.welch(wav_data, fs_audio, nperseg=nperseg)
Pxx_den_log = 10*np.log10(Pxx_den)

plt.figure()
plt.plot(f, Pxx_den_log)
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()

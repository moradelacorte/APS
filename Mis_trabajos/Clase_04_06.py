# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Jun  4 19:19:19 2026

@author: Mora De La Corte

Descripción: 
    Clase virtual 4/6. Diseño de filtros FIR
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
 
ws1 = 0.1
wp1 = 0.5
wp2 = 35
ws2 = 45

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

# Aplicando filtro IIR------------------------------------------------------------

wp = [wp1, wp2]
ws = [ws1, ws2]

sos_but = sig.iirdesign(wp, ws, ripple/2, atenuacion/2, analog=False, ftype='butter', output='sos', fs=fs)

sos = sos_but 
ww = np.concat([np.logspace(start=-2,stop = 0.1, num = 500),
                np.linspace(start = 1.26,stop = 35, num = 200),
                np.logspace(start=1.55,stop = 1.65, num = 300),
                np.linspace(start = 46,stop = fs//2, num = 50)])

omega, resp_freq = sig.freqz_sos(sos, worN=ww, fs=fs)

#%% Diseño de FIR

numtaps = 3601 # Como minimo de mayor orden del peor de los IIR
demora = (numtaps - 1)//2 

freq = np.array([0.0, ws1, wp1, wp2, ws2, fs//2]) 
# gain = 10**((-1)*np.array([atenuacion, atenuacion, ripple, ripple, atenuacion, atenuacion])/20)
gain = np.array([0, 0, 1, 1, 0, 0])

# if numtaps % 2 == 0:
#     gain[-1] = 0.

b_win = sig.firwin2(numtaps, freq, gain, nfreqs = 2**14, window='boxcar', fs=fs)

w, h = sig.freqz(b_win, worN=ww, fs=fs)

z, p, k = sig.tf2zpk(b_win, a=1)


#%%
# Grafica plantilla vs filtro -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(w, 20 * np.log10(np.abs(h)), label='FIR boxcar', color='C0', linewidth=2)

frecs_hz = (frecs * nyq_frec) #Pasamos frecs de nuevo a Hz

plt.scatter(frecs_hz, gains_db, color='red', zorder=5, label='Puntos de la Plantilla')
plt.plot(frecs_hz, gains_db, color='red', linestyle=':', alpha=0.7, label='Forma aproximada')

plt.title('Verificación: Filtro Diseñado vs Plantilla')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud (dB)')

# zoom en la zona de interés (0 a 60 Hz) porque ws2 es 45 Hz.
plt.xlim([0, 60]) 
plt.ylim([-50, 5]) # De -50 dB a 5 dB para ver bien la caída y el ripple

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

#%% # 2. RETARDO DE GRUPO 

fase = np.unwrap(np.angle(h))

gd = -np.diff(fase)/np.diff(2*np.pi*ww/fs)
gd = np.append(gd[0], gd)
w_gd = ww
 
plt.figure()
plt.plot(w_gd, gd, color='green', linewidth=2)
plt.title('Retardo de Grupo (Group Delay)')
plt.ylabel('Retardo (muestras)')
plt.xlabel('Frecuencia (Hz)')
plt.grid(True, linestyle='--', alpha=0.7)
# axs[1].set_xlim([0, 60])

# 3. DIAGRAMA DE POLOS Y CEROS

plt.figure(figsize=(6, 6)) # Figura independiente y cuadrada para el plano Z

# Dibujar el círculo unitario
circulo = plt.Circle((0, 0), radius=1, color='black', fill=False, linestyle='--', alpha=0.5)
plt.gca().add_patch(circulo)

# Graficar ceros y polos usando plt directamente
plt.scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='blue', s=60, label='Ceros')
plt.scatter(np.real(p), np.imag(p), marker='x', color='red', s=60, label='Polos')

plt.title('Diagrama de Polos y Ceros (Plano Z)')
plt.xlabel('Parte Real')
plt.ylabel('Parte Imaginaria')
plt.axvline(0, color='black', lw=0.5)
plt.axhline(0, color='black', lw=0.5)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.axis('equal') # Mantiene la proporcionalidad circular del plano Z

plt.tight_layout()
plt.show()

#%% mas ventanas

b_win_ham = sig.firwin2(numtaps, freq, gain, nfreqs=2**14, window='hamming', fs=fs)

# Respuestas específicas para la ventana Hamming
# w_ham, h_ham = sig.freqz(b_win_ham, worN=ww, fs=fs)
# z_ham, p_ham, k_ham = sig.tf2zpk(b_win_ham, a=1)

b_win_han = sig.firwin2(numtaps, freq, gain, nfreqs=2**14, window='hann', fs=fs)

# Respuestas específicas para Hanning
# w_han, h_han = sig.freqz(b_win_han, a=1, worN=ww, fs=fs)
# z_han, p_han, k_han = sig.tf2zpk(b_win_han, a=1)

b_win_bla = sig.firwin2(numtaps, freq, gain, nfreqs=2**14, window='blackman', fs=fs)

# Respuestas específicas para Blackman
# w_bla, h_bla = sig.freqz(b_win_bla, a=1, worN=ww, fs=fs)
# z_bla, p_bla, k_bla = sig.tf2zpk(b_win_bla, a=1)

#%% Aplicando los FIR

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

ecg_one_lead = mat_struct['ecg_lead'].flatten()
ecg_f_win = sig.lfilter(b_win, 1, ecg_one_lead)
ecg_f_win_ham = sig.lfilter(b_win_ham, 1, ecg_one_lead)
ecg_f_win_han = sig.lfilter(b_win_han, 1, ecg_one_lead)
ecg_f_win_bla = sig.lfilter(b_win_bla, 1, ecg_one_lead)

ecg_but = sig.sosfiltfilt(sos_but, ecg_one_lead)

# Regiones de interés sin ruido  
 
regs_interes = (
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_but[zoom_region], label='Butterworth', color='red')
    plt.plot(zoom_region, ecg_f_win[zoom_region + demora], label='FIR boxcar', color='tan')
    plt.plot(zoom_region, ecg_f_win_ham[zoom_region + demora], label='FIR hamming', color='orchid')
    plt.plot(zoom_region, ecg_f_win_han[zoom_region + demora], label='FIR hanning', color='orange')
    plt.plot(zoom_region, ecg_f_win_bla[zoom_region + demora], label='FIR blackman', color='green')


   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
    
# Regiones de interés con ruido  
regs_interes = (
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )
 
for ii in regs_interes:
   
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(np.max([0, ii[0]]), np.min([N, ii[1]]), dtype='uint')
   
    plt.figure()
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    plt.plot(zoom_region, ecg_but[zoom_region], label='Butterworth', color='red')
    plt.plot(zoom_region, ecg_f_win[zoom_region + demora], label='FIR boxcar', color='tan')
    plt.plot(zoom_region, ecg_f_win_ham[zoom_region + demora], label='FIR hamming', color='orchid')
    plt.plot(zoom_region, ecg_f_win_han[zoom_region + demora], label='FIR hanning', color='orange')
    plt.plot(zoom_region, ecg_f_win_bla[zoom_region + demora], label='FIR blackman', color='green')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Sat Jun 13 11:40:12 2026

@author: Mora De La Corte

Descripción: 
    Pasando en limpio el codigo de TS7
    Filtros FIR ventanas
"""

#%% Importo los modulos y archivos

import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

ecg_one_lead = mat_struct['ecg_lead'].flatten()

#%% Plantilla

def limites_plantilla():
    ws1 = 0.1
    wp1 = 0.5
    wp2 = 35
    ws2 = 45

    f_min, f_max = 0, 100
    gain_min, gain_max = -60, 5

    # Banda de rechazo inferior (0 a ws1): Prohibido arriba de -40 dB
    plt.fill_between([0, ws1], -40, gain_max, color='red', alpha=0.3, label='Zona Prohibida')

    # Banda de paso (wp1 a wp2): 
    # Prohibido abajo de -3 dB (tolerancia de rizado) y arriba de 0 dB
    plt.fill_between([wp1, wp2], gain_min, -3, color='red', alpha=0.3)
    plt.fill_between([wp1, wp2], 0, gain_max, color='red', alpha=0.3)

    # Banda de rechazo superior (ws2 a 100 Hz): Prohibido arriba de -40 dB
    plt.fill_between([ws2, f_max], -40, gain_max, color='red', alpha=0.3)

    plt.axhline(y=-40, color='gray', linestyle='-', alpha=0.5)
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    plt.xlim(f_min, f_max)
    plt.ylim(gain_min, gain_max)
    #plt.xlabel('Frecuencia [Hz]', fontsize=11)
    #plt.ylabel('Módulo [dB]', fontsize=11)
    #plt.title('Plantilla de Diseño', fontsize=13, fontweight='bold')
    #plt.grid(True, which='both', linestyle=':', alpha=0.5)
    #plt.legend(loc='lower right')
    
    return

#%% Creo los filtros FIR win

fs = 1000 # fs_ecg 
 
nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = 0.4
wp1 = 1
wp2 = 35
ws2 = 40

# from scipy.signal import kaiserord

# # Atenuación garantizada por cada ventana:
# ventanas = {
#     'Boxcar':   21,   # dB → NO cumple 40 dB, descartada
#     'Hamming':  53,
#     'Hanning':  44,
#     'Blackman': 74,
# }

# delta_f = min(wp1 - ws1, ws2 - wp2) / nyq_frec  # normalizado

# for nombre, aten in ventanas.items():
#     n, _ = kaiserord(aten, delta_f)
#     if n % 2 == 0: n += 1  # forzar impar
#     print(f"{nombre:10s}: {n} taps")
    
# n_kaiser, beta = kaiserord(40, delta_f)
# print(f"Taps necesarios (Kaiser): {n_kaiser}")

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ])
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

wp = [wp1, wp2]
ws = [ws1, ws2]

numtaps = 3001 # Como minimo de mayor orden del peor de los IIR mas de 3000 no hace falta
demora = (numtaps - 1)//2 

# gain = 10**((-1)*np.array([atenuacion, atenuacion, ripple, ripple, atenuacion, atenuacion])/20)
gain = np.array([0, 0, 1, 1, 0, 0])

# if numtaps % 2 == 0:
#     gain[-1] = 0.

b_win_box = sig.firwin2(numtaps, frecs, gain, nfreqs = 2**14, window='boxcar', fs=fs)
b_win_ham = sig.firwin2(numtaps, frecs, gain, nfreqs=2**14, window='hamming', fs=fs)
b_win_han = sig.firwin2(numtaps, frecs, gain, nfreqs=2**14, window='hann', fs=fs)
b_win_bla = sig.firwin2(numtaps, frecs, gain, nfreqs=2**14, window='blackman', fs=fs)


ww = np.concatenate([np.logspace(start=-2,stop = 0.1, num = 500),
                np.linspace(start = 1.26,stop = 35, num = 200),
                np.logspace(start=1.55,stop = 1.65, num = 300),
                np.linspace(start = 46,stop = fs//2, num = 50)])

filtros = [
    (b_win_box, 'Boxcar', 'C0', '--'),
    (b_win_ham,  'Hamming', 'C1', '-'),
    (b_win_han, 'Hanning', 'C2', '-'),
    (b_win_bla, 'blackman', 'C3', '-'),
]

#%% grafico
plt.figure(figsize=(12, 6))

for b_win, label, color, ls in filtros:
    omega, H = sig.freqz(b_win, worN=ww, fs=fs)
    plt.plot(omega, 20 * np.log10(np.abs(H)),
             label=label, color=color, linestyle=ls, linewidth=1.8)

limites_plantilla()
plt.title('Comparación de filtros FIR win — todos deben cumplir la plantilla')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Módulo (dB)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#%% Aplico los filtros a las zonas de interes

ecg_f_win_box = sig.lfilter(b_win_box, 1, ecg_one_lead)
ecg_f_win_ham = sig.lfilter(b_win_ham, 1, ecg_one_lead)
ecg_f_win_han = sig.lfilter(b_win_han, 1, ecg_one_lead)
ecg_f_win_bla = sig.lfilter(b_win_bla, 1, ecg_one_lead)

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
    plt.plot(zoom_region, ecg_f_win_box[zoom_region + demora], label='FIR boxcar', color='tan')
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
    plt.plot(zoom_region, ecg_f_win_box[zoom_region + demora], label='FIR boxcar', color='tan')
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

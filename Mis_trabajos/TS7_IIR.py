# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Sat Jun 13 11:14:34 2026

@author: Mora De La Corte

Descripción: 
    Pasando en limpio el codigo de TS7
    Filtros IIR
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

#%% Creo los filtros IIR

fs = 1000 # fs_ecg 

nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1_iir = 0.1
wp1_iir = 0.5
wp2_iir = 35
ws2_iir = 45

frecs = np.array([0.0,         ws1_iir,         wp1_iir,     wp2_iir,     ws2_iir,         nyq_frec   ]) / nyq_frec
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

wp_iir = [wp1_iir, wp2_iir]
ws_iir = [ws1_iir, ws2_iir]

sos_coef = sig.iirdesign(wp_iir, ws_iir, ripple, atenuacion, analog=False, ftype='butter', output='sos', fs=fs)
sos_but = sig.iirdesign(wp_iir, ws_iir, ripple/2, atenuacion, analog=False, ftype='butter', output='sos', fs=fs)
sos_chb1 = sig.iirdesign(wp_iir, ws_iir, ripple/2, atenuacion, analog=False, ftype='cheby1', output='sos', fs=fs)
sos_chb2 = sig.iirdesign(wp_iir, ws_iir, ripple/2, atenuacion, analog=False, ftype='cheby2', output='sos', fs=fs)
sos_cau = sig.iirdesign(wp_iir, ws_iir, ripple/2, atenuacion, analog=False, ftype='cauer', output='sos', fs=fs)


ww = np.concatenate([np.logspace(start=-2,stop = 0.1, num = 500),
                     np.linspace(start = 1.26,stop = 35, num = 200),
                     np.logspace(start=1.55,stop = 1.65, num = 300),
                     np.linspace(start = 46,stop = fs//2, num = 50)])

filtros = [
    (sos_coef, 'Butter (referencia, 1 dB / 40 dB)', 'C0', '--'),
    (sos_but,  'Butter  (0.5 dB / 40 dB)',           'C1', '-'),
    (sos_chb1, 'Chebyshev I',                         'C2', '-'),
    (sos_chb2, 'Chebyshev II',                        'C3', '-'),
    (sos_cau,  'Cauer (elíptico)',                    'C4', '-'),
]

plt.figure(figsize=(12, 6))

for sos, label, color, ls in filtros:
    omega, H = sig.freqz_sos(sos, worN=ww, fs=fs)
    plt.plot(omega, 20 * np.log10(np.abs(H)),
             label=label, color=color, linestyle=ls, linewidth=1.8)

limites_plantilla()
plt.title('Comparación de filtros IIR — todos deben cumplir la plantilla')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Módulo (dB)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#%% Aplico los filtros a las zonas de interes

ecg_but = sig.sosfiltfilt(sos_but, ecg_one_lead)
ecg_chb1 = sig.sosfiltfilt(sos_chb1, ecg_one_lead)
ecg_chb2 = sig.sosfiltfilt(sos_chb2, ecg_one_lead)
ecg_cau = sig.sosfiltfilt(sos_cau, ecg_one_lead)

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
    plt.plot(zoom_region, ecg_but[zoom_region], label='Butterworth', color='tan')
    plt.plot(zoom_region, ecg_chb1[zoom_region], label='cheby1', color='yellowgreen')
    plt.plot(zoom_region, ecg_chb2[zoom_region], label='cheby2', color='teal')
    plt.plot(zoom_region, ecg_cau[zoom_region], label='cauer', color='mediumvioletred')
   
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
    plt.plot(zoom_region, ecg_but[zoom_region], label='Butterworth', color='tan')
    plt.plot(zoom_region, ecg_chb1[zoom_region], label='cheby1', color='yellowgreen')
    plt.plot(zoom_region, ecg_chb2[zoom_region], label='cheby2', color='teal')
    plt.plot(zoom_region, ecg_cau[zoom_region], label='cauer', color='mediumvioletred')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Sat Jun 13 14:24:17 2026

@author: Mora De La Corte

Descripción: 
    Pasando en limpio el codigo de TS7
    Filtros FIR por Cuadrados Mínimos y Parks-Mc Clellan-Remez.
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

#%% Creo los filtros FIR

# Cuadrados Mínimos

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

omega_firls, H_firls = sig.freqz(h_firls, worN=ww, fs=fs)

# Parks-Mc Clellan-Remez

ws1_rm = 0.1
wp1_rm  = 0.8
wp2_rm  = 35
ws2_rm  = 36

frecs_rm  = np.array([0.0,         ws1_rm ,         wp1_rm ,     wp2_rm ,     ws2_rm ,         nyq_frec   ])

numtaps_remez = 3200
desired_remez = np.array([0,1,0])
weight_remez = np.array([2,1,1])

h_remez = sig.remez(numtaps = numtaps_remez, bands = frecs_rm , desired = desired_remez, weight = weight_remez, fs = fs)  

omega_remez, H_remez = sig.freqz(h_remez, worN=ww, fs=fs)
#%% grafico
plt.figure(figsize=(12, 6))

plt.plot(omega_firls, 20 * np.log10(np.abs(H_firls)), label='Cuadrados minimos', color='C0', linewidth=1.8)
plt.plot(omega_remez, 20 * np.log10(np.abs(H_remez)), label='Remez', color='C1', linewidth=1.8)

limites_plantilla()
plt.title('FIR cuadrados minimos vs remez')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Módulo (dB)')
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

#%% Aplico los filtros a las zonas de interes

ecg_firls = sig.lfilter(h_firls, 1, ecg_one_lead)
ecg_remez = sig.lfilter(h_remez, 1, ecg_one_lead)

demora_firls = (numtaps - 1)//2 
demora_remez = (numtaps_remez - 1)//2 

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
    plt.plot(zoom_region, ecg_firls[zoom_region + demora_firls], label='Cuadrados minimos', color='tan')
    # plt.plot(zoom_region, ecg_remez[zoom_region + demora_remez], label='remez', color='yellowgreen')
   
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
    plt.plot(zoom_region, ecg_firls[zoom_region + demora_firls], label='Cuadrados minimos', color='tan')
    # plt.plot(zoom_region, ecg_remez[zoom_region + demora_remez], label='remez', color='yellowgreen')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
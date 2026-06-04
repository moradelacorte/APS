# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Jun  3 13:47:32 2026

@author: Mora De La Corte

Descripción: 
    TS7 - Filtrado digital lineal de ECG
    Trabajado en clase
"""
import numpy as np
from scipy import signal as sig

import matplotlib.pyplot as plt
   
import scipy.io as sio
from scipy.io.wavfile import write

# Cargando el archivo como lectura_sigs.py ------------------------------------

fs = 1000 # fs_ecg 

sio.whosmat('ECG_TP4.mat')
mat_struct = sio.loadmat('ECG_TP4.mat')

ecg_one_lead = mat_struct['ecg_lead']
N = len(ecg_one_lead)

hb_1 = mat_struct['heartbeat_pattern1']
hb_2 = mat_struct['heartbeat_pattern2']

# Para visualizar los datos

# plt.figure()
# plt.plot(ecg_one_lead[5000:12000]) # Esta limitado porque son demasiados datos
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.figure()
# plt.plot(hb_1)

# plt.figure()
# plt.plot(hb_2)

#%% Señal cuadrada

frecuencia = 5       
fs = 1000          
duracion = 1         

t = np.linspace(0, duracion, int(fs * duracion), endpoint=False)

senal_cuadrada = sig.square(2 * np.pi * frecuencia * t)

#%% Plantilla -------------------------------------------------------------------
 
nyq_frec = fs/2
ripple = 1 # dB
atenuacion = 40 # dB
 
ws1 = 0.1
wp1 = 0.5
wp2 = 45
ws2 = 35

frecs = np.array([0.0,         ws1,         wp1,     wp2,     ws2,         nyq_frec   ]) / nyq_frec
gains_db = np.array([-atenuacion, -atenuacion, -ripple, -ripple, -atenuacion, -atenuacion])

# Aplicando filtro ------------------------------------------------------------
wp = [wp1, wp2]
ws = [ws1, ws2]

ftype = 'butter'
# ftype = 'cheby1'
# ftype = 'cheby2'
# ftype = 'cauer'

sos_coef = sig.iirdesign(wp, ws, ripple, atenuacion, analog=False, ftype=ftype, output='sos', fs=fs)
sos_ff = sig.iirdesign(wp, ws, ripple/2, atenuacion/2, analog=False, ftype=ftype, output='sos', fs=fs)
sos_but = sig.iirdesign(wp, ws, ripple/2, atenuacion/2, analog=False, ftype='butter', output='sos', fs=fs)
sos_chb1 = sig.iirdesign(wp, ws, ripple/2, atenuacion/2, analog=False, ftype='cheby1', output='sos', fs=fs)
sos_chb2 = sig.iirdesign(wp, ws, ripple/2, atenuacion/2, analog=False, ftype='cheby2', output='sos', fs=fs)
sos_cau = sig.iirdesign(wp, ws, ripple/2, atenuacion/2, analog=False, ftype='cauer', output='sos', fs=fs)



ww = np.concat([np.logspace(start=-2,stop = 0.1, num = 500),
                np.linspace(start = 1.26,stop = 35, num = 200),
                np.logspace(start=1.55,stop = 1.65, num = 300),
                np.linspace(start = 46,stop = fs//2, num = 50)])

omega, resp_freq = sig.freqz_sos(sos_coef, worN=ww, fs=fs)

#%%
# Grafica plantilla vs filtro -------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(omega, 20 * np.log10(np.abs(resp_freq)), label='Filtro Diseñado', color='C0', linewidth=2)

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

#%%
# fase, retardo y polos -------------------------------------------------------

fig, axs = plt.subplots(3, 1, figsize=(10, 15))

# 1. GRÁFICO DE FASE
fase = np.unwrap(np.angle(resp_freq))

axs[0].plot(omega, fase, color='orange', linewidth=2)
axs[0].set_title('Respuesta de Fase')
axs[0].set_ylabel('Fase (radianes)')
axs[0].set_xlabel('Frecuencia (Hz)')
axs[0].grid(True, linestyle='--', alpha=0.7)
# axs[0].set_xlim([0, 60]) # Hacemos zoom en nuestra zona de interés

# 2. RETARDO DE GRUPO 
# b, a = sig.sos2tf(sos_coef)
# w_gd, gd = sig.group_delay((b, a), w=omega, fs=fs)

gd = -np.diff(fase)/np.diff(ww)
gd = np.append(gd[0], gd)
w_gd = ww

# Evitamos graficar picos infinitos por errores numéricos cerca de los cortes
gd = np.clip(gd, -100, 1000) 

axs[1].plot(w_gd, gd, color='green', linewidth=2)
axs[1].set_title('Retardo de Grupo (Group Delay)')
axs[1].set_ylabel('Retardo (muestras)')
axs[1].set_xlabel('Frecuencia (Hz)')
axs[1].grid(True, linestyle='--', alpha=0.7)
# axs[1].set_xlim([0, 60])

# 3. DIAGRAMA DE POLOS Y CEROS
z, p, k = sig.sos2zpk(sos_coef)

circulo = plt.Circle((0, 0), radius=1, color='black', fill=False, linestyle='--', alpha=0.5)
axs[2].add_patch(circulo)

axs[2].scatter(np.real(z), np.imag(z), marker='o', facecolors='none', edgecolors='blue', s=60, label='Ceros')
axs[2].scatter(np.real(p), np.imag(p), marker='x', color='red', s=60, label='Polos')

axs[2].set_title('Diagrama de Polos y Ceros (Plano Z)')
axs[2].set_xlabel('Parte Real')
axs[2].set_ylabel('Parte Imaginaria')
axs[2].axvline(0, color='black', lw=0.5)
axs[2].axhline(0, color='black', lw=0.5)
axs[2].grid(True, alpha=0.3)
axs[2].legend(loc='upper right')
axs[2].axis('equal') 

plt.tight_layout()
plt.show()

#%%
ecg_one_lead = mat_struct['ecg_lead'].flatten()

# # Aplicamos el filtro con fase cero
ecg_filtrado = sig.sosfilt(sos_coef, ecg_one_lead)
ecg_filtrado_ff = sig.sosfiltfilt(sos_ff, ecg_one_lead)

ecg_but = sig.sosfiltfilt(sos_but, ecg_one_lead)
ecg_chb1 = sig.sosfiltfilt(sos_chb1, ecg_one_lead)
ecg_chb2 = sig.sosfiltfilt(sos_chb2, ecg_one_lead)
ecg_cau = sig.sosfiltfilt(sos_cau, ecg_one_lead)

square_filtrado = sig.sosfilt(sos_coef, senal_cuadrada)
square_filtrado_ff = sig.sosfiltfilt(sos_ff, senal_cuadrada)

#%% Regiones de interés sin ruido  
 
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

    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
   
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Adimensional')
    plt.xlabel('Muestras (#)')
   
    axes_hdl = plt.gca()
    axes_hdl.legend()
    axes_hdl.set_yticks(())
           
    plt.show()
 
#%% Regiones de interés con ruido  
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

#%%
# Comparar el antes y el después
# plt.figure(figsize=(10, 4))
# plt.plot(ecg_one_lead[5000:12000], label='Original Sucio', alpha=0.7)
# plt.plot(ecg_filtrado[5000:12000], label='Filtrado / Limpio', color='red', linewidth=1.5)
# plt.plot(ecg_filtrado_ff[5000:12000], label='filtfilt', color='orange', linewidth=1.5)
# plt.title('Efecto del Filtro Pasabanda en el ECG')
# plt.legend()
# plt.grid(True)
# plt.show()

plt.figure(figsize=(10, 4))
plt.plot(senal_cuadrada, label='Original Sucio', alpha=0.7)
plt.plot(square_filtrado, label='Filtrado / Limpio', color='red', linewidth=1.5)
plt.plot(square_filtrado_ff, label='filtfilt', color='orange', linewidth=1.5)
plt.title('Efecto del Filtro Pasabanda en el ECG')
plt.legend()
plt.grid(True)
plt.show()
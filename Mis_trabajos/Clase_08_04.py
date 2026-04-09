# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Apr  8 19:53:22 2026

@author: Mora De La Corte

Descripción: 
    
    Trabajado en la clase virtual del 8/4/26
    
"""
import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.linspace(0, (nn-1)*ts, nn)
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    return tt, xx

fs = 1000          # Frecuencia de muestreo (1000 Hz)
N = 1000      # 250 ciclos completos → 250 000 muestras
f0 = N/4             # lo pidio el profe
vmax = np.sqrt(2)
B = 4
V_R = 2.0
q = V_R / (2**B)
kn = 1 # con 1/10 la tendencia uniforme del histograma se pierde

tt, s = mi_funcion_sen(vmax, 0, f0, 0, N, fs)

Pq = (q**2) / 12
Pn = kn * Pq
# np.random.seed(0)
n = np.random.normal(0, np.sqrt(Pn), N)

sR = s + n
sQ = np.round(sR / q) * q

# ── Gráfico 1: señal en el tiempo ──────────────────────────────────────────
plt.figure(1)
plt.plot(tt, sQ, label=r'$s_Q = Q_{B,V_R}\{s_R\}$ (ADC out)', linewidth=2.5, color='C0')
plt.plot(tt, sR, 'g.:', label=r'$s_R = s + n$ (ADC in)', markersize=4, alpha=0.8)
plt.plot(tt, s,  color='orange', linestyle=':', label=r'$s$ (analog)')
plt.title(f'Señal muestreada por un ADC de {B} bits - $\\pm V_R = {V_R}$ V - $q = {q}$ V')
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ── Transformadas de Fourier ───────────────────────────────────────────────
S  = np.fft.fft(s)  / N
SR = np.fft.fft(sR) / N
SQ = np.fft.fft(sQ) / N

freqs     = np.fft.fftfreq(N, 1/fs)
pos_freqs = freqs[:N//2]

eps = 1e-12
psd_s  = 10 * np.log10(2 * np.abs(S [:N//2])**2 + eps)
psd_sR = 10 * np.log10(2 * np.abs(SR[:N//2])**2 + eps)
psd_sQ = 10 * np.log10(2 * np.abs(SQ[:N//2])**2 + eps)

floor_digital_dB = 10 * np.log10(2 * Pq / N)
floor_analog_dB  = 10 * np.log10(2 * Pn / N)

# Por las dudas revisar que el tema de la normalizacion tenga sentido

# ── Gráfico 2: solo señal analógica + pisos ────────────────────────────────
plt.figure(2)
plt.plot(pos_freqs, psd_s, color='orange', linestyle=':', label=r'$s$ (analog)', alpha=0.8)
plt.axhline(floor_analog_dB,  color='r', linestyle='--',
            label=f'$\\bar{{n}} = {floor_analog_dB:.1f}$ dB (piso analog.)')
plt.axhline(floor_digital_dB, color='c', linestyle='--',
            label=f'$\\bar{{n}}_Q = {floor_digital_dB:.1f}$ dB (piso digital)')
plt.title(f'Señal muestreada por un ADC de {B} bits - $\\pm V_R = {V_R}$ V - $q = {q}$ V')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Densidad de Potencia [dB]')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# ── Gráfico 3: histograma del error de cuantización ───────────────────────
eq = sQ - sR

plt.figure(3)
counts, bins, patches = plt.hist(eq, bins=10, range=(-q/2, q/2), color='C0')
ideal_count = N / 10
plt.plot([-q/2, -q/2, q/2, q/2], [0, ideal_count, ideal_count, 0], 'r--', linewidth=2)
plt.title(f'Ruido de cuantización para {B} bits - $\\pm V_R = {V_R}$ V - $q = {q}$ V')
plt.tight_layout()
plt.show()
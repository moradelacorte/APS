# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Mar 26 19:08:38 2026

@author: Mora De La Corte

Descripción: 
    
"""

import numpy as np
import matplotlib.pyplot as plt


def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs
    tt = np.linspace(0, (nn - 1) * ts, nn).reshape(nn, 1)
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx


# ─── Parámetros base ────────────────────────────────────────────────────────
fs   = 1000          # frecuencia de muestreo [Hz]
N    = int(fs * 1)   # número de muestras (1 segundo)
ff   = fs / N        # f0 = fs/N = Δf  (frecuencia de la senoidal)
vmax = np.sqrt(2)    # amplitud → varianza/potencia = 1 W (energía normalizada)
B    = 4             # bits del ADC
VF   = 2             # semirango analógico [V]  → rango total ±VF
Vfs  = VF            # full-scale = VF → q = VF/2^B = 2/16 = 0.125 V
qq   = Vfs / 2**B    # paso de cuantización  q = 2/16 = 0.125 V
kn   = 9            # escala de potencia del ruido

# ─── Potencia de cuantización y ruido aditivo ───────────────────────────────
Pq = qq**2 / 12      # potencia de ruido de cuantización teórica
Pn = kn * Pq         # potencia del ruido aditivo gaussiano

# ─── Señal senoidal pura s[n] ────────────────────────────────────────────────
tt, ss = mi_funcion_sen(vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs)
ss = ss.flatten()

# Verificación de energía normalizada
print(f'Verificación de energía (varianza de s): {np.var(ss):.4f}  (debe ser ≈ 1.0)')

# ─── Ruido gaussiano incorrelado n[n] ────────────────────────────────────────
np.random.seed(42)   # semilla para reproducibilidad
noise = np.random.normal(0, np.sqrt(Pn), N)

# ─── Señal de entrada al ADC: sR[n] = s[n] + n[n] ───────────────────────────
sR = ss + noise

# ─── Cuantización uniforme con saturación ────────────────────────────────────
#   Clipping al rango [-VF, +VF] antes de cuantizar (comportamiento real del ADC)
sR_clipped = np.clip(sR, -VF, VF)
s0 = np.round(sR_clipped / qq) * qq   # salida del ADC (cuantizada)

# ─── Parámetros numéricos ────────────────────────────────────────────────────
print(f'\n─── Parámetros ─────────────────────────────────────────────────────')
print(f'  B   = {B} bits          → niveles = {2**B}')
print(f'  ±VF = ±{VF} V           → Vfs total = {Vfs} V')
print(f'  q   = Vfs/2^B = {qq:.4f} V')
print(f'  f0  = fs/N = {ff:.4f} Hz')
print(f'  Pq  = q²/12  = {Pq:.6f} V²')
print(f'  kn  = {kn}  →  Pn = kn·Pq = {Pn:.6f} V²')
print(f'  σ_n = √Pn    = {np.sqrt(Pn):.6f} V')

# ─── Figura: Señal muestreada por el ADC ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))

fig.suptitle(
    f'Señal muestreada por un ADC de {B} bits  -  '
    f'±$V_R$ = {VF:.1f} V  -  q = {qq:.3f} V',
    fontsize=13, fontweight='bold'
)

# s (analógica) — línea punteada amarilla/dorada
ax.plot(tt.flatten(), ss,
        color='goldenrod', lw=0.8, ls=':', alpha=0.9,
        label=r'$s$ (analog)')

# sR = s + n (entrada al ADC) — puntos verdes con línea delgada
ax.plot(tt.flatten(), sR,
        color='seagreen', lw=0.6, marker='o', markersize=1.5,
        alpha=0.7, ls='-',
        label=r'$s_R = s + n$  (ADC in)')

# s0 = Q_B,Vr{sR} (salida del ADC) — línea azul sólida
ax.plot(tt.flatten(), s0,
        color='steelblue', lw=1.2, alpha=0.85,
        label=r'$s_0 = Q_{B,V_r}\{s_R\}$  (ADC out)')

ax.set_xlabel('tiempo [segundos]', fontsize=11)
ax.set_ylabel('Amplitud [V]', fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(-VF * 1.15, VF * 1.15)
ax.axhline( VF, color='gray', lw=0.6, ls='--', alpha=0.5)
ax.axhline(-VF, color='gray', lw=0.6, ls='--', alpha=0.5)
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ─── Error de cuantización ───────────────────────────────────────────────────
ee = (s0 - sR_clipped)   # error = cuantizada − entrada (después de clip)

# ─── Figura 2: Espectro en dB ────────────────────────────────────────────────
freqs = np.fft.rfftfreq(N, d=1/fs)

def calc_psd_dB(sig, N, floor_dB=1e-12):
    X   = np.fft.rfft(sig)
    # PSD una cara: (2/N²)|X|²  → pico senoidal con Vmax=√2 queda en Ps=1 W → 0 dB
    psd = (2 / N**2) * (np.abs(X) ** 2)
    psd[0]  /= 2          # DC no se duplica
    if N % 2 == 0:
        psd[-1] /= 2      # Nyquist tampoco
    return 10 * np.log10(np.maximum(psd, floor_dB))

psd_s_dB  = calc_psd_dB(ss, N)
psd_sR_dB = calc_psd_dB(sR, N)
psd_s0_dB = calc_psd_dB(s0, N)

# Pisos de ruido teóricos
# Con normalización (2/N²)|X|², la densidad por bin = P_ruido / (N/2)
# expresado en V²/bin (no V²/Hz), que es lo que grafica la FFT discreta
Vn_teorico = Pn  / (N / 2)    # piso analógico  [V²/bin]
Vq_teorico = Pq  / (N / 2)    # piso digital     [V²/bin]
Vn_dB      = 10 * np.log10(Vn_teorico)
Vq_dB      = 10 * np.log10(Vq_teorico)

fig2, ax2 = plt.subplots(figsize=(13, 6))
fig2.suptitle(
    f'Señal muestreada por un ADC de {B} bits  -  '
    f'±$V_R$ = {VF:.1f} V  -  q = {qq:.3f} V',
    fontsize=13, fontweight='bold'
)

ax2.plot(freqs, psd_s_dB,  color='goldenrod',  lw=0.8, ls=':',  alpha=0.9,
         label=r'$s$ (analog)')
ax2.plot(freqs, psd_sR_dB, color='steelblue',  lw=0.7, ls=':', alpha=0.7,
         label=r'$s_R = s + n$  (ADC in)')
ax2.plot(freqs, psd_s0_dB, color='steelblue',  lw=1.0, alpha=0.9,
         label=r'$s_0 = Q_{B,V_r}\{s_R\}$  (ADC out)')

ax2.axhline(Vn_dB, color='tomato',    lw=1.5, ls='--',
            label=rf'$\overline{{n}}$ = {Vn_dB:.1f} dB (piso analog.)')
ax2.axhline(Vq_dB, color='darkcyan',  lw=1.5, ls='--',
            label=rf'$\overline{{n_0}}$ = {Vq_dB:.1f} dB (piso digital)')

ax2.set_xlabel('Frecuencia [Hz]', fontsize=11)
ax2.set_ylabel('Densidad de Potencia [dB]', fontsize=11)
ax2.set_xlim(0, fs/2)
ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ─── Figura 3: Histograma del ruido de cuantización ─────────────────────────
fig3, ax3 = plt.subplots(figsize=(10, 5))
fig3.suptitle(
    f'Ruido de cuantización para {B} bits  -  '
    f'±$V_R$ = {VF:.1f} V  -  q = {qq:.3f} V',
    fontsize=12, fontweight='bold'
)

# Distribución uniforme teórica: altura = N / (qq * num_bins)
num_bins  = 8
bin_edges = np.linspace(-qq/2, qq/2, num_bins + 1)
counts, _ = np.histogram(ee, bins=bin_edges)

ax3.bar(bin_edges[:-1], counts,
        width=np.diff(bin_edges),
        align='edge',
        color='steelblue', edgecolor='white', lw=0.5)

# Línea roja horizontal: nivel teórico uniforme = N / num_bins
nivel_uniforme = N / num_bins
ax3.axhline(nivel_uniforme, color='red', lw=1.5, ls='--',
            label=f'Nivel uniforme teórico = {nivel_uniforme:.0f}')

ax3.set_xlabel('Error de cuantización (V)', fontsize=11)
ax3.set_ylabel('Conteo', fontsize=11)
ax3.set_xlim(-qq/2 * 1.05, qq/2 * 1.05)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.2)

plt.tight_layout()

plt.show()

print(f'\n─── Resumen numérico ───────────────────────────────────────────────')
print(f'  Varianza del error    : {np.var(ee):.6f} V²  |  q²/12 = {Pq:.6f} V²')
print(f'  Potencia senoidal Ps  : {np.var(ss):.4f} V²  (normalizada = 1)')
print(f'  Potencia ruido Pn     : {np.var(noise):.6f} V²  (teórico = {Pn:.6f} V²)')
print(f'  Piso analógico (Vn)   : {Vn_dB:.1f} dB')
print(f'  Piso digital  (Vq)    : {Vq_dB:.1f} dB')
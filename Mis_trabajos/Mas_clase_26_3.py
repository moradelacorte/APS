# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Mar 26 21:33:25 2026

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
vmax   = np.sqrt(2)
ff     = 1
B      = 3
Vfs    = 3
qq     = Vfs / 2**B
snr_db = 15

# ─── Barrido analítico: piso vs fs ──────────────────────────────────────────
ps      = vmax**2 / 2                        # potencia senoidal teórica
pr      = ps / (10 ** (snr_db / 10))        # potencia de ruido
ps_dB   = 10 * np.log10(ps)                 # nivel del pico en dB

# fs_cruce: punto donde Vv_teorico = ps  (piso iguala el pico en densidad)
# Vv = 2*pr/fs = ps  →  fs = 2*pr/ps
fs_cruce = 2 * pr / ps
Vv_cruce_dB = 10 * np.log10(2 * pr / fs_cruce)   # debería ser = ps_dB

# Barrido de fs en escala logarítmica
fs_sweep  = np.logspace(np.log10(0.05), np.log10(20000), 500)
Vv_sweep  = 2 * pr / fs_sweep               # piso teórico [V²/Hz]
Vv_sw_dB  = 10 * np.log10(Vv_sweep)

# ─── Figura 1: barrido analítico ────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(11, 5))
fig1.suptitle(
    f'Piso espectral Vv = 2·Pv/fs  vs  fs  —  SNR = {snr_db} dB',
    fontsize=13, fontweight='bold'
)

ax.semilogx(fs_sweep, Vv_sw_dB, color='steelblue', lw=2,
            label='Vv = 2·Pv/fs  (piso teórico)')

# Nivel del pico de la senoidal
ax.axhline(ps_dB, color='tomato', lw=1.5, ls='--',
           label=f'Nivel del pico  Ps = {ps_dB:.2f} dB')

# Punto de cruce
ax.axvline(fs_cruce, color='darkorange', lw=1.5, ls=':',
           label=f'Cruce: fs = {fs_cruce:.4f} Hz')
ax.scatter([fs_cruce], [Vv_cruce_dB], color='darkorange', zorder=5, s=80)

# Marcar los fs del ejercicio anterior
fs_ejemplos = [1000, 2000, 4000, 8000]
colors_ej   = ['seagreen', 'mediumorchid', 'goldenrod', 'crimson']
for fse, col in zip(fs_ejemplos, colors_ej):
    Vv_ej = 10 * np.log10(2 * pr / fse)
    ax.scatter([fse], [Vv_ej], color=col, zorder=5, s=60,
               label=f'fs={fse} Hz → Vv={Vv_ej:.1f} dB')

ax.set_xlabel('fs  [Hz]  (escala logarítmica)')
ax.set_ylabel('Densidad espectral de potencia  [dB]')
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(fs_sweep[0], fs_sweep[-1])

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cruce_piso_vs_fs.png', dpi=150)
plt.show()

# ─── Figura 2: espectros superpuestos con referencia del pico ───────────────
fs_list = [1000, 2000, 4000, 8000]
colors  = ['steelblue', 'seagreen', 'tomato', 'darkorchid']

fig2, ax2 = plt.subplots(figsize=(12, 5))
fig2.suptitle(
    f'Espectros de v[n] normalizados — se ve cuánto falta para alcanzar el pico\n'
    f'SNR = {snr_db} dB  |  Pico senoidal = {ps_dB:.2f} dB',
    fontsize=12, fontweight='bold'
)

for fs, color in zip(fs_list, colors):
    N  = int(fs * 1)
    tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs)
    ruido  = np.random.normal(0, np.sqrt(pr), (N, 1))
    v_arr  = ruido.flatten()

    freqs      = np.fft.rfftfreq(N, d=1/fs)
    freqs_norm = freqs / (fs / 2)
    X_v        = np.fft.rfft(v_arr) / N
    psd_v      = (2 / N) * (np.abs(X_v) ** 2)
    psd_v[0]  /= 2
    psd_v_dB   = 10 * np.log10(np.maximum(psd_v, 1e-12))

    Vv_teorico = 2 * pr / fs
    Vv_dB      = 10 * np.log10(Vv_teorico)
    distancia  = ps_dB - Vv_dB       # cuántos dB falta para llegar al pico

    ax2.plot(freqs_norm, psd_v_dB, color=color, lw=0.7, alpha=0.5)
    ax2.axhline(Vv_dB, color=color, lw=2, ls='--',
                label=f'fs={fs} Hz | Vv={Vv_dB:.1f} dB | Δ={distancia:.1f} dB al pico')

# Nivel del pico de la senoidal como referencia
ax2.axhline(ps_dB, color='black', lw=2, ls='-',
            label=f'Pico senoidal = {ps_dB:.2f} dB  ← SNR separa el piso de aquí')

ax2.annotate('', xy=(0.5, ps_dB), xytext=(0.5, 10*np.log10(2*pr/1000)),
             arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5))
ax2.text(0.52, (ps_dB + 10*np.log10(2*pr/1000))/2,
         f'SNR={snr_db} dB\n+ spread\nen densidad', fontsize=7, color='gray')

ax2.set_xlabel('Frecuencia normalizada  f / (fs/2)  [0 … 1]')
ax2.set_ylabel('dB')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()

# ─── Resumen numérico ────────────────────────────────────────────────────────
print(f"\nPs (potencia senoidal)  = {ps:.4f} V²  →  {ps_dB:.2f} dB")
print(f"Pv (potencia ruido)     = {pr:.4f} V²  (SNR = {snr_db} dB)")
print(f"\nCruce analítico (Vv = Ps en densidad):")
print(f"  fs_cruce = 2·Pv/Ps = {fs_cruce:.5f} Hz")
print(f"  → Para fs > fs_cruce el piso BAJA por debajo del pico")
print(f"  → Para fs < fs_cruce el piso SUBE por encima del pico")
print()
print(f"{'fs (Hz)':>10}  {'Vv (dB)':>10}  {'Δ al pico (dB)':>16}")
print("-" * 42)
for fs in fs_list:
    Vv = 10 * np.log10(2 * pr / fs)
    print(f"{fs:>10}  {Vv:>10.2f}  {ps_dB - Vv:>16.2f}")
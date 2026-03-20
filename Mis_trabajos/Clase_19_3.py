# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Mar 19 21:09:46 2026

@author: Mora De La Corte

Descripción: 
    Generado con ayuda de claude.ai
"""
import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1/fs
    tt = np.linspace(0, (nn-1)*ts, nn).reshape(nn, 1)
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    return tt, xx

# Parámetros base
vmax = np.sqrt(2)
ff = 1
fs = 1000
N = int(fs * 1)
B = 3
Vfs = 3
qq = Vfs / 2**B

tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs)
ps = np.var(xx)

snr_list = [None, 15]

for snr_db in snr_list:

    if snr_db is None:
        xx_in = xx.copy()
        titulo_ruido = "sin ruido"
    else:
        pr = ps / (10**(snr_db/10))
        ruido = np.random.normal(0, np.sqrt(pr), (N, 1))
        xx_in = xx + ruido
        titulo_ruido = f"SNR = {snr_db} dB"

    # Cuantización y error
    xxq = np.round(xx_in / qq) * qq
    ee = (xxq - xx_in).flatten()

    # Autocorrelación del error
    ee_centered = ee - np.mean(ee)
    autocorr = np.correlate(ee_centered, ee_centered, mode='full')
    autocorr = autocorr / autocorr.max()  # normalizada
    lags = np.arange(-(N-1), N)           # eje de lags

    # --- 4 subplots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        f'Cuantización uniforme — B={B} bits, Vfs={Vfs}V, q={qq:.4f}V | {titulo_ruido}',
        fontsize=12
    )

    ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

    # Panel 1: señal original vs cuantizada
    ax1.plot(tt, xx_in, color='steelblue', linewidth=1, alpha=0.6, label='xx_in (original)')
    ax1.plot(tt, xxq,   color='tomato',    linewidth=1.2, linestyle='--', label='xxq (cuantizada)')
    ax1.set_ylabel('Amplitud (V)')
    ax1.set_title('Señal cuantizada xxq')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5)

    # Panel 2: error de cuantización
    ax2.plot(tt, ee, color='seagreen', linewidth=0.8)
    ax2.axhline( qq/2, color='red', linewidth=0.8, linestyle='--', label=f'+q/2 = +{qq/2:.4f}V')
    ax2.axhline(-qq/2, color='red', linewidth=0.8, linestyle='--', label=f'-q/2 = -{qq/2:.4f}V')
    ax2.set_ylabel('Error (V)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_title('Error de cuantización ee = xxq − xx_in')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)

    # Panel 3: histograma del error
    ax3.hist(ee, bins=20, color='steelblue', edgecolor='white', linewidth=0.4, density=True)
    ax3.axvline( qq/2, color='red', linewidth=0.8, linestyle='--', label=f'+q/2 = {qq/2:.4f}V')
    ax3.axvline(-qq/2, color='red', linewidth=0.8, linestyle='--', label=f'-q/2 = {-qq/2:.4f}V')
    ax3.set_xlabel('Error (V)')
    ax3.set_ylabel('Densidad')
    ax3.set_title('Histograma del error ee')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: autocorrelación del error
    ax4.plot(lags, autocorr, color='darkorange', linewidth=0.8)
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_xlabel('Lag (muestras)')
    ax4.set_ylabel('Autocorrelación normalizada')
    ax4.set_title('Autocorrelación del error ee')
    ax4.set_xlim(-N, N)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f'[{titulo_ruido}] Varianza del error: {np.var(ee):.6f} V² | q²/12 = {qq**2/12:.6f} V²')
 
# FFT

XX = np.fft.fft(xx)
XX_modu = np.abs(XX)
XX_ph = np.angle(XX)

fig1, (ax5,ax6) = plt.subplots(1,2)
ax5.plot(XX_modu)
ax6.plot(XX_ph)
plt.show()
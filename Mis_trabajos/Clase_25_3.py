# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Mar 25 20:25:32 2026

@author: Mora De La Corte

Descripción: 
    consigna:
    - ver el espectro de la señal de entrada la senoidal mas ruido
    
senoidal mas ruido uniformemente distribuido -- > x[n] = xr[n] + v[n] -- > ADC -- > xxq
Me interesa el espectro de  x[n] : 
    Espectro de xr[n] espero una delta; 
    Espectro de v[n] espero una constante de altura (area igual a la potencia) --> hasta N/2 (porque es par) --> ancho de banda = fs/2; 
    altura = Valor maximo que puede tomar V: Vv / Vv = 2*Pv/fs    
    Que pasa si cambio el ancho de banda? Vv se achica --> conversores sigma-delta: bajan el Vv aumentando el ancho de banda
    - visualizar esto en dB
"""
import numpy as np
import matplotlib.pyplot as plt
 
 
def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):
    ts = 1 / fs
    tt = np.linspace(0, (nn - 1) * ts, nn).reshape(nn, 1)
    xx = vmax * np.sin(2 * np.pi * ff * tt + ph) + dc
    return tt, xx
 
 
# ─── Parámetros base ────────────────────────────────────────────────────────
vmax = np.sqrt(2)   # amplitud senoidal  → potencia = 1 W
ff   = 1            # frecuencia [Hz]
fs   = 1000         # frecuencia de muestreo [Hz]
N    = int(fs * 1)  # número de muestras (1 segundo)
B    = 3            # bits del ADC
Vfs  = 3            # rango full-scale [V]
qq   = Vfs / 2**B   # paso de cuantización
 
tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs)
ps = np.var(xx)     # potencia de la senoidal
 
snr_list = [None, 15]   # None = sin ruido aditivo; 15 → SNR = 15 dB
 
for snr_db in snr_list:
 
    # ── Señal de entrada ─────────────────────────────────────────────────────
    if snr_db is None:
        xx_in         = xx.copy()
        titulo_ruido  = "sin ruido"
        pr            = 0.0
    else:
        pr    = ps / (10 ** (snr_db / 10))
        ruido = np.random.normal(0, np.sqrt(pr), (N, 1))
        xx_in = xx + ruido
        titulo_ruido = f"SNR = {snr_db} dB"
 
    # ── Cuantización y error ─────────────────────────────────────────────────
    xxq = np.round(xx_in / qq) * qq
    ee  = (xxq - xx_in).flatten()
 
    # ── Autocorrelación del error ─────────────────────────────────────────────
    ee_centered = ee - np.mean(ee)
    autocorr = np.correlate(ee_centered, ee_centered, mode='full')
    autocorr = autocorr / autocorr.max()
    lags     = np.arange(-(N - 1), N)
 
    # ── Espectro de x[n] = xr[n] + v[n]  (consigna principal) ───────────────
    freqs = np.fft.rfftfreq(N, d=1/fs)          # eje de frecuencia [Hz], 0 … fs/2
 
    # Espectro de xr[n] (senoidal pura)
    X_r   = np.fft.rfft(xx.flatten()) / N
    psd_r = (2 / N) * (np.abs(X_r) ** 2)        # densidad de potencia normalizada
    psd_r[0] /= 2                                 # corrección componente DC
 
    # Espectro de v[n] (ruido aditivo gaussiano)
    if snr_db is not None:
        v_arr = (xx_in - xx).flatten()
        X_v   = np.fft.rfft(v_arr) / N
        psd_v = (2 / N) * (np.abs(X_v) ** 2)
        psd_v[0] /= 2
    else:
        psd_v = np.zeros_like(psd_r)
 
    # Espectro de x[n] = xr[n] + v[n]
    X_x   = np.fft.rfft(xx_in.flatten()) / N
    psd_x = (2 / N) * (np.abs(X_x) ** 2)
    psd_x[0] /= 2
 
    # Nivel teórico del piso de ruido uniforme:
    #   Pv distribuida en ancho de banda fs/2  → Vv = 2*Pv/fs
    if snr_db is not None:
        Vv_teorico = 2 * pr / fs
    else:
        Vv_teorico = None
 
    # Conversión a dB (floor para evitar log(0))
    floor_dB = 1e-12
    psd_r_dB = 10 * np.log10(np.maximum(psd_r, floor_dB))
    psd_v_dB = 10 * np.log10(np.maximum(psd_v, floor_dB))
    psd_x_dB = 10 * np.log10(np.maximum(psd_x, floor_dB))
 
    # ── Figura 1: señal temporal + error + histograma + autocorrelación ────────
    fig1, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig1.suptitle(
        f'Cuantización uniforme — B={B} bits, Vfs={Vfs} V, q={qq:.4f} V | {titulo_ruido}',
        fontsize=12, fontweight='bold'
    )
 
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]
 
    # Panel 1: señal original vs cuantizada
    ax1.plot(tt, xx_in, color='steelblue',  lw=1,   alpha=0.6, label='x[n] (entrada)')
    ax1.plot(tt, xxq,   color='tomato',     lw=1.2, ls='--',   label='xxq (cuantizada)')
    ax1.set_ylabel('Amplitud (V)')
    ax1.set_title('Señal cuantizada xxq')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', lw=0.5)
 
    # Panel 2: error de cuantización
    ax2.plot(tt, ee, color='seagreen', lw=0.8)
    ax2.axhline( qq/2, color='red', lw=0.8, ls='--', label=f'+q/2 = +{qq/2:.4f} V')
    ax2.axhline(-qq/2, color='red', lw=0.8, ls='--', label=f'-q/2 = -{qq/2:.4f} V')
    ax2.set_ylabel('Error (V)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_title('Error de cuantización  ee = xxq − x[n]')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='black', lw=0.5)
 
    # Panel 3: histograma del error
    ax3.hist(ee, bins=20, color='steelblue', edgecolor='white', lw=0.4, density=True)
    ax3.axvline( qq/2, color='red', lw=0.8, ls='--', label=f'+q/2 = {qq/2:.4f} V')
    ax3.axvline(-qq/2, color='red', lw=0.8, ls='--', label=f'-q/2 = {-qq/2:.4f} V')
    ax3.set_xlabel('Error (V)')
    ax3.set_ylabel('Densidad')
    ax3.set_title('Histograma del error ee')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
 
    # Panel 4: autocorrelación del error
    ax4.plot(lags, autocorr, color='darkorange', lw=0.8)
    ax4.axhline(0, color='black', lw=0.5)
    ax4.set_xlabel('Lag (muestras)')
    ax4.set_ylabel('Autocorrelación normalizada')
    ax4.set_title('Autocorrelación del error ee')
    ax4.set_xlim(-N, N)
    ax4.grid(True, alpha=0.3)
 
    plt.tight_layout()

    plt.show()
 
    # ── Figura 2: espectros separados en dB ─────────────────────────────────
    if snr_db is not None:
        n_rows = 3
    else:
        n_rows = 1   # sin ruido solo tiene sentido el espectro de xr[n]
 
    fig2, axes2 = plt.subplots(n_rows, 1, figsize=(11, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes2 = [axes2]   # para iterar igual en ambos casos
 
    fig2.suptitle(
        f'Espectros individuales en dB — {titulo_ruido}',
        fontsize=12, fontweight='bold'
    )
 
    # ── Espectro 1: xr[n] — senoidal pura ───────────────────────────────────
    ax_r = axes2[0]
    ax_r.plot(freqs, psd_r_dB, color='tomato', lw=1.2)
    ax_r.set_ylabel('dB')
    ax_r.set_title(r'Espectro de $x_r[n]$ (senoidal) — se espera una $\delta$ en $f_0$')
    ax_r.grid(True, alpha=0.3)
    ax_r.axvline(ff, color='tomato', lw=0.8, ls=':', alpha=0.7, label=f'f₀ = {ff} Hz')
    ax_r.legend(fontsize=8)
 
    if snr_db is not None:
        # ── Espectro 2: v[n] — ruido gaussiano ──────────────────────────────
        ax_v = axes2[1]
        ax_v.plot(freqs, psd_v_dB, color='seagreen', lw=0.9, alpha=0.9,
                  label='v[n] (ruido gaussiano)')
        Vv_dB = 10 * np.log10(Vv_teorico)
        ax_v.axhline(Vv_dB, color='orange', lw=1.2, ls='--',
                     label=f'Vv teórico = {Vv_dB:.1f} dB  (= 2·Pv/fs)')
        ax_v.set_ylabel('dB')
        ax_v.set_title(r'Espectro de $v[n]$ (ruido) — se espera piso constante hasta $f_s/2$')
        ax_v.legend(fontsize=8)
        ax_v.grid(True, alpha=0.3)
 
        # ── Espectro 3: x[n] = xr[n] + v[n] ────────────────────────────────
        ax_x = axes2[2]
        ax_x.plot(freqs, psd_x_dB, color='steelblue', lw=0.9,
                  label=r'$x[n] = x_r[n] + v[n]$')
        ax_x.axhline(Vv_dB, color='orange', lw=1.0, ls='--',
                     label=f'Vv teórico = {Vv_dB:.1f} dB')
        ax_x.axvline(ff, color='tomato', lw=0.8, ls=':', alpha=0.7, label=f'f₀ = {ff} Hz')
        ax_x.set_ylabel('dB')
        ax_x.set_title(r'Espectro de $x[n]$ (señal completa) — $\delta$ + piso de ruido')
        ax_x.legend(fontsize=8)
        ax_x.grid(True, alpha=0.3)
 
    axes2[-1].set_xlabel('Frecuencia (Hz)')
    plt.tight_layout()

    plt.show()
 
    # ── Figura 3: espectros separados en escala lineal ───────────────────────
    fig3, axes3 = plt.subplots(n_rows, 1, figsize=(11, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes3 = [axes3]
 
    fig3.suptitle(
        f'Espectros individuales — escala lineal (V²) — {titulo_ruido}',
        fontsize=12, fontweight='bold'
    )
 
    # ── Espectro 1: xr[n] — senoidal pura ───────────────────────────────────
    ax_r2 = axes3[0]
    ax_r2.plot(freqs, psd_r, color='tomato', lw=1.2)
    ax_r2.set_ylabel('Potencia (V²)')
    ax_r2.set_title(r'Espectro de $x_r[n]$ (senoidal) — delta en $f_0$')
    ax_r2.axvline(ff, color='tomato', lw=0.8, ls=':', alpha=0.7, label=f'f₀ = {ff} Hz')
    ax_r2.legend(fontsize=8)
    ax_r2.grid(True, alpha=0.3)
 
    if snr_db is not None:
        # ── Espectro 2: v[n] — ruido ────────────────────────────────────────
        ax_v2 = axes3[1]
        ax_v2.plot(freqs, psd_v, color='seagreen', lw=0.9, alpha=0.9,
                   label='v[n] (ruido gaussiano)')
        ax_v2.axhline(Vv_teorico, color='orange', lw=1.2, ls='--',
                      label=f'Vv teórico = {Vv_teorico:.2e} V²/Hz  (= 2·Pv/fs)')
        ax_v2.set_ylabel('Potencia (V²)')
        ax_v2.set_title(r'Espectro de $v[n]$ (ruido) — piso constante hasta $f_s/2$')
        ax_v2.legend(fontsize=8)
        ax_v2.grid(True, alpha=0.3)
 
        # ── Espectro 3: x[n] = xr[n] + v[n] ────────────────────────────────
        ax_x2 = axes3[2]
        ax_x2.plot(freqs, psd_x, color='steelblue', lw=0.9,
                   label=r'$x[n] = x_r[n] + v[n]$')
        ax_x2.axhline(Vv_teorico, color='orange', lw=1.0, ls='--',
                      label=f'Vv teórico = {Vv_teorico:.2e} V²/Hz')
        ax_x2.axvline(ff, color='tomato', lw=0.8, ls=':', alpha=0.7, label=f'f₀ = {ff} Hz')
        ax_x2.set_ylabel('Potencia (V²)')
        ax_x2.set_title(r'Espectro de $x[n]$ (señal completa) — $\delta$ + piso de ruido')
        ax_x2.legend(fontsize=8)
        ax_x2.grid(True, alpha=0.3)
 
    axes3[-1].set_xlabel('Frecuencia (Hz)')
    plt.tight_layout()

    plt.show()
 
    # ── Resumen numérico ─────────────────────────────────────────────────────
    print(f'\n[{titulo_ruido}]')
    print(f'  Varianza del error    : {np.var(ee):.6f} V²  |  q²/12 = {qq**2/12:.6f} V²')
    print(f'  Potencia senoidal Ps  : {ps:.4f} V²')
    if snr_db is not None:
        print(f'  Potencia ruido Pv     : {pr:.4f} V²')
        print(f'  Vv teórico (2·Pv/fs) : {Vv_teorico:.6f} V²/Hz  '
              f'→ {10*np.log10(Vv_teorico):.2f} dB')
        print(f'  Ancho de banda        : {fs/2:.0f} Hz  '
              f'(sigma-delta ↑BW → ↓Vv → ↓piso de ruido en banda útil)')
 
# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Tue Jun 23 19:40:01 2026

@author: Mora De La Corte

Descripción: 
    Simulando distintos tipos de filtros(pasa-bajos, pasa-altos, pasa-banda y notch)
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================
# COMPARACIÓN DE FILTROS: Pasa-Bajos, Pasa-Altos, Pasa-Banda, Notch
# Todos diseñados como filtros de segundo orden (Butterworth / RLC)
# Frecuencia central / de corte: w0 = 3 rad/s
# Factor de calidad: Q = 10 (para pasa-banda y notch)
# ==============================================================

w0 = 1.0       # Frecuencia de corte / central [rad/s]
Q  = 5      # Factor de calidad (afecta ancho de banda en pasa-banda y notch)
fs = 10        # Frecuencia de muestreo para discretización [Hz]

# ---------------------------------------------------------------
# Definición de los cuatro filtros analógicos (numerador, denominador)
# ---------------------------------------------------------------

# 1) PASA-BAJOS:  H(s) = w0^2 / (s^2 + (w0/Q)*s + w0^2)
b_lp = [w0**2]
a_lp = [1, w0/Q, w0**2]

# 2) PASA-ALTOS:  H(s) = s^2 / (s^2 + (w0/Q)*s + w0^2)
b_hp = [1, 0, 0]
a_hp = [1, w0/Q, w0**2]

# 3) PASA-BANDA:  H(s) = (w0/Q)*s / (s^2 + (w0/Q)*s + w0^2)
b_bp = [w0/Q, 0]
a_bp = [1, w0/Q, w0**2]

# 4) NOTCH (Rechaza-banda): H(s) = (s^2 + w0^2) / (s^2 + (w0/Q)*s + w0^2)
b_no = [1, 0, w0**2]
a_no = [1, w0/Q, w0**2]

filtros = {
    "Pasa-Bajos (LP)":  (b_lp, a_lp),
    "Pasa-Altos (HP)":  (b_hp, a_hp),
    "Pasa-Banda (BP)":  (b_bp, a_bp),
    "Notch (NF)":       (b_no, a_no),
}

colores = {
    "Pasa-Bajos (LP)":  "tab:blue",
    "Pasa-Altos (HP)":  "tab:orange",
    "Pasa-Banda (BP)":  "tab:green",
    "Notch (NF)":       "tab:red",
}

markers = {
    "Pasa-Bajos (LP)":  "x",
    "Pasa-Altos (HP)":  "x",
    "Pasa-Banda (BP)":  "x",
    "Notch (NF)":       "x",
}

# ==============================================================
# RESPUESTA EN FRECUENCIA ANALÓGICA
# ==============================================================
w_log = np.logspace(-1, 1.5, 2000)   # 0.1 … ~31.6 rad/s

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("Respuesta en frecuencia analógica |H(jw)| y Fase", fontsize=13)

for nombre, (b, a) in filtros.items():
    _, h = signal.freqs(b, a, w_log)
    mag_db = 20 * np.log10(np.abs(h) + 1e-12)
    fase_deg = np.rad2deg(np.unwrap(np.angle(h)))
    c = colores[nombre]
    axes[0].semilogx(w_log, mag_db, label=nombre, color=c, lw=2)
    axes[1].semilogx(w_log, fase_deg, label=nombre, color=c, lw=2)

axes[0].axvline(w0, color="k", ls="--", lw=1, label=f"w0 = {w0} rad/s")
axes[0].set_ylabel("Magnitud [dB]")
axes[0].set_ylim(-60, 10)
axes[0].grid(True, which="both", ls=":")
axes[0].legend(fontsize=8)

axes[1].axvline(w0, color="k", ls="--", lw=1)
axes[1].set_xlabel("w [rad/s]")
axes[1].set_ylabel("Fase [grados]")
axes[1].grid(True, which="both", ls=":")

plt.tight_layout()

# ==============================================================
# POLOS Y CEROS EN EL PLANO S (un subplot por filtro)
# ==============================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle("Polos y ceros — Plano s (analógico)", fontsize=13)

for ax, (nombre, (b, a)) in zip(axs.flat, filtros.items()):
    z, p, _ = signal.tf2zpk(b, a)
    c = colores[nombre]
    mk = markers[nombre]

    ax.plot(np.real(p), np.imag(p), mk, color=c,
            markersize=10, markeredgewidth=2, label="Polos")
    if len(z) > 0:
        ax.plot(np.real(z), np.imag(z), "o", color=c,
                markersize=10, fillstyle="none", markeredgewidth=2, label="Ceros")

    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)

    # Círculo de radio w0 de referencia
    circ = patches.Circle((0, 0), radius=w0, fill=False,
                           color="gray", ls="dotted", lw=1.5)
    ax.add_patch(circ)

    lim = w0 * 1.6
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(nombre, fontsize=10)
    ax.set_xlabel("σ")
    ax.set_ylabel("jω")
    ax.grid(True)
    ax.legend(fontsize=8)

    # Imprimir info en consola
    print(f"\n── {nombre} ──")
    print(f"  Ceros (s): {z}   |z|: {np.abs(z)}")
    print(f"  Polos (s): {p}   |p|: {np.abs(p)}")

plt.tight_layout()

# ==============================================================
# DISCRETIZACIÓN POR TRANSFORMADA BILINEAL
# ==============================================================
digitales = {}
for nombre, (b, a) in filtros.items():
    num, den = signal.bilinear(b, a, fs)
    digitales[nombre] = (num, den)

# --- Respuesta en frecuencia digital ---
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f"Respuesta en frecuencia digital (fs = {fs} Hz)", fontsize=13)

for nombre, (num, den) in digitales.items():
    wz, hz = signal.freqz(num, den, worN=2000)
    mag_db = 20 * np.log10(np.abs(hz) + 1e-12)
    fase_deg = np.rad2deg(np.unwrap(np.angle(hz)))
    c = colores[nombre]

    # Frecuencia normalizada de corte (pre-warped)
    w0_norm = w0 / (np.pi * fs)
    axes[0].plot(wz / np.pi, mag_db, label=nombre, color=c, lw=2)
    axes[1].plot(wz / np.pi, fase_deg, label=nombre, color=c, lw=2)

# Línea vertical en frecuencia digital de w0
w0_dig = w0 / (np.pi * fs)
axes[0].axvline(w0_dig, color="k", ls="--", lw=1,
                label=f"w0 ≈ {w0_dig:.3f}π rad/muestra")
axes[0].set_ylabel("Magnitud [dB]")
axes[0].set_ylim(-60, 10)
axes[0].grid(True, which="both", ls=":")
axes[0].legend(fontsize=8)

axes[1].axvline(w0_dig, color="k", ls="--", lw=1)
axes[1].set_xlabel("θ / π  [rad/muestra]")
axes[1].set_ylabel("Fase [grados]")
axes[1].grid(True, which="both", ls=":")

plt.tight_layout()

# --- Polos y ceros en el plano z ---
fig, axs = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(f"Polos y ceros — Plano z (digital, fs = {fs} Hz)", fontsize=13)

for ax, (nombre, (num, den)) in zip(axs.flat, digitales.items()):
    zz, pz, _ = signal.tf2zpk(num, den)
    c = colores[nombre]
    mk = markers[nombre]

    ax.plot(np.real(pz), np.imag(pz), mk, color=c,
            markersize=10, markeredgewidth=2, label="Polos")
    if len(zz) > 0:
        ax.plot(np.real(zz), np.imag(zz), "o", color=c,
                markersize=10, fillstyle="none", markeredgewidth=2, label="Ceros")

    unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                                  color="gray", ls="dotted", lw=2)
    ax.add_patch(unit_circle)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title(nombre, fontsize=10)
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.grid(True)
    ax.legend(fontsize=8)

    print(f"\n── {nombre} (digital) ──")
    print(f"  Ceros (z): {zz}")
    print(f"  Polos (z): {pz}")
    print(f"  |polos|:   {np.abs(pz)}")

plt.tight_layout()


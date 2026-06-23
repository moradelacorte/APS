# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Mon Jun 22 11:26:46 2026

@author: Mora De La Corte

Descripción: 
    Simulación del filtro dado en el segundo parcial punto 1
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ==============================================================
# FILTRO ANALOGICO T(s) = (3^2 / 2) / (s^2 + s*3/10 + 3^2)
# No tiene ceros finitos (están en el infinito)
# Polos en -0.15 +- 2.996j -> radio 3 (w0 = 3 rad/s)
# ==============================================================
b = [3**2 / 2]           # Numerador: 4.5
a = [1, 3/10, 3**2]      # Denominador: s^2 + 0.3*s + 9

# --- Respuesta en frecuencia analogica ---
w = np.logspace(-1, 1.5, 1000) # de 0.1 a ~31.6 rad/s (centrado en w0=3)
w, h = signal.freqs(b, a, w)

plt.figure()
plt.semilogx(w, 20*np.log10(abs(h)))
plt.title("Magnitud analógica |H(jw)|")
plt.xlabel("w [rad/s]"); plt.ylabel("dB")
plt.grid(True, which="both", ls=":")

# --- Polos y ceros en el plano s ---
z, p, k = signal.tf2zpk(b, a)
print("ceros (s):", z, " radio:", np.abs(z))
print("polos (s):", p, " radio:", np.abs(p))

plt.figure(figsize=(6, 6))
ax = plt.gca()
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.title("Polos y ceros (plano s)")
plt.xlabel("sigma"); plt.ylabel("jw")
plt.axis('equal'); plt.grid(True); plt.legend()

# ==============================================================
# FILTRO DIGITAL por transformada bilineal: s = 2*fs*(z-1)/(z+1)
# ==============================================================
fs = 50 # <-- PROBAR distintos valores: 1, 2, 5 (w0 = 3 rad/s aprox 0.48 Hz)
num, den = signal.bilinear(b, a, fs)

# --- Respuesta en frecuencia digital (theta en [0, pi] rad/muestra) ---
wz, hz = signal.freqz(num, den, worN=1000)

plt.figure()
plt.plot(wz/np.pi, 20*np.log10(abs(hz)))
plt.title("Magnitud digital |H|, fs = %g" % fs)
plt.xlabel("theta / pi [rad/muestra]"); plt.ylabel("dB")
plt.grid(True, which="both", ls=":")

# --- Polos y ceros en el plano z ---
zz, pz, kz = signal.tf2zpk(num, den)
print("ceros (z):", zz, " radio", np.abs(zz), " angulo[rad]", np.angle(zz))
print("polos (z):", pz, " radio", np.abs(pz), " angulo[rad]", np.angle(pz))

plt.figure(figsize=(6, 6))
ax = plt.gca()
plt.plot(np.real(pz), np.imag(pz), 'x', markersize=10, label='Polos')
if len(zz) > 0:
    plt.plot(np.real(zz), np.imag(zz), 'o', markersize=10, fillstyle='none', label='Ceros')
unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
ax.add_patch(unit_circle)
plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title("Polos y ceros (plano z), fs = %g" % fs)
plt.xlabel("Re(z)"); plt.ylabel("Im(z)")
plt.grid(True); plt.legend()


# --- Fase analógica ---
plt.figure()
plt.semilogx(w, np.rad2deg(np.unwrap(np.angle(h))))
plt.title("Fase analógica")
plt.xlabel("w [rad/s]")
plt.ylabel("Fase [grados]")
plt.grid(True, which="both", ls=":")

# --- Fase digital ---
plt.figure()
plt.plot(wz/np.pi, np.rad2deg(np.unwrap(np.angle(hz))))
plt.title("Fase digital, fs = %g" % fs)
plt.xlabel("theta / pi [rad/muestra]")
plt.ylabel("Fase [grados]")
plt.grid(True, which="both", ls=":")

plt.show()
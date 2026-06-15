# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Sun Jun 14 11:24:54 2026

@author: Mora De La Corte

Descripción: 
    Códigos brindados por la catedra para la guia 5
"""
# Punto 1 ---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as spsw
 
fs = 1000   # frecuencia de muestreo [Hz]
N  = 100    # cantidad de muestras
n  = np.arange(N)
 
# La señal: coseno a 205 Hz
# 205 Hz NO cae en un bin exacto (bin 20 = 200 Hz, bin 21 = 210 Hz)
x = np.cos(2 * np.pi * 205 * n / fs)
 
freqs = np.fft.rfftfreq(N, d=1/fs)  # eje de frecuencias: 0 a fs/2
 
# -----------------------------------------------------------
# GRÁFICO 1: qué pasa con cada ventana
# Todo el bloque de plot tiene que estar DENTRO del for
# -----------------------------------------------------------
for nombre, w in [("Rectangular", np.ones(N)),
                  ("Hann",        spsw.hann(N, sym=False))]:
 
    X = np.fft.rfft(x * w) / N          # FFT con la ventana aplicada
 
    plt.figure(figsize=(8, 3))
    plt.plot(freqs, 20 * np.log10(np.abs(X) + 1e-12), label=nombre, marker='o')
    plt.axvline(205, color="red", linestyle="--", label="205 Hz real")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X(f)| [dB]")
    plt.title(f"Espectro con ventana {nombre}")
    plt.ylim(-80, 0)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
 
plt.show()
 
# -----------------------------------------------------------
# ¿QUÉ TENÉS QUE MIRAR?
#
# La línea roja vertical es donde DEBERÍA estar el pico (205 Hz).
#
# Ventana RECTANGULAR:
#   - El pico no está exactamente en 205 Hz sino entre dos bins
#     (200 y 210 Hz). Está "aplastado".
#   - Aparecen "patas" que se extienden por todo el espectro
#     (decaen lento, tipo -6 dB por octava).
#   - Eso es el DESPARRAMO: energía que aparece en frecuencias
#     donde no hay señal real.
#
# Ventana HANN:
#   - El pico también está entre los dos bins, pero es MÁS ANCHO.
#   - Las "patas" casi desaparecen (caen a -80 dB rápido).
#   - La ventana suaviza el desparramo a costa de ensanchar el pico.
#
# CONCLUSIÓN VISUAL:
#   Rectangular → pico estrecho pero patas grandes
#   Hann        → pico ancho pero patas pequeñas
#   Ese es el trade-off fundamental de las ventanas.
# -----------------------------------------------------------

#%% 
# Punto 2 ---------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as spsw

fs = 8000   # frecuencia de muestreo [Hz]
N  = 800    # cantidad de muestras

# -----------------------------------------------------------
# PARTE A: respuesta en frecuencia de cada ventana
# Así podés ver visualmente el ancho del lóbulo principal
# y el nivel de los lóbulos laterales de cada una
# -----------------------------------------------------------

ventanas = {
    "Rectangular":      np.ones(N),
    "Hann":             spsw.hann(N, sym=False),
    "Hamming":          spsw.hamming(N, sym=False),
    "Blackman-Harris":  spsw.blackmanharris(N, sym=False),
    "Flattop":          spsw.flattop(N, sym=False),
}

# Zero-padding para ver bien la forma de los lóbulos
zp = 8 * N
freqs_win = np.fft.rfftfreq(zp, d=1/fs)

plt.figure(figsize=(10, 5))
for nombre, w in ventanas.items():
    W    = np.fft.rfft(w, n=zp)
    W_db = 20 * np.log10(np.abs(W) / (np.abs(W[0]) + 1e-14))
    plt.plot(freqs_win, W_db, label=nombre, linewidth=1.2)

# La resolución mínima teórica de Hamming es 4 * fs/N = 40 Hz
# marcamos esa frecuencia para que se vea dónde termina el lóbulo principal
plt.axvline(40, color="gray", linestyle=":", linewidth=1, label="40 Hz (ref. Hann/Hamming)")
plt.axhline(-40, color="black", linestyle="--", linewidth=0.8, label="-40 dB (señal débil)")
plt.axhline(-13, color="red",   linestyle="--", linewidth=0.8, label="-13 dB (1er lób. Rect.)")

plt.xlim(0, 300)   # enfocamos los primeros lóbulos
plt.ylim(-120, 5)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|W(f)| [dB]  (0 dB = cima del lóbulo principal)")
plt.title("Respuesta en frecuencia de las ventanas\n(con fs=8000 Hz, N=800)")
plt.grid(True, which="both")
plt.legend(fontsize=8)
plt.tight_layout()

# -----------------------------------------------------------
# PARTE B: simulación con dos señales de distinta amplitud
#
# Señal fuerte: 0 dB en f1 = 1000 Hz
# Señal débil:  -40 dB en f2 = 1050 Hz  (50 Hz de separación)
#
# ¿Qué ventanas permiten "ver" las dos?
# -----------------------------------------------------------

n    = np.arange(N)
A1   = 1.0                       # amplitud señal fuerte  (0 dB)
A2   = 10**(-40/20)              # amplitud señal débil   (-40 dB ≈ 0.01)
f1   = 1000                      # [Hz]
f2   = 1050                      # [Hz]
x    = A1 * np.cos(2*np.pi*f1*n/fs) + A2 * np.cos(2*np.pi*f2*n/fs)

freqs = np.fft.rfftfreq(N, d=1/fs)

fig, axes = plt.subplots(len(ventanas), 1, figsize=(10, 14), sharex=True)

for ax, (nombre, w) in zip(axes, ventanas.items()):
    X     = np.fft.rfft(x * w) / N
    X_db  = 20 * np.log10(np.abs(X) + 1e-14)

    ax.plot(freqs, X_db, linewidth=0.9, label=nombre)
    ax.axvline(f1, color="blue",  linestyle="--", linewidth=0.8, alpha=0.6, label=f"f1={f1} Hz (0 dB)")
    ax.axvline(f2, color="green", linestyle="--", linewidth=0.8, alpha=0.6, label=f"f2={f2} Hz (-40 dB)")
    ax.set_xlim(800, 1300)
    ax.set_ylim(-100, 5)
    ax.set_ylabel("[dB]")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True)
    ax.set_title(nombre)

axes[-1].set_xlabel("Frecuencia [Hz]")
fig.suptitle("¿Se pueden ver las dos señales?\n"
             "Línea azul = 0 dB en 1000 Hz  |  Línea verde = −40 dB en 1050 Hz",
             fontsize=10)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------
# ¿QUÉ TENÉS QUE MIRAR?
#
# En cada subplot buscá si aparecen DOS picos (uno en cada línea
# de colores) o si la señal débil queda tapada por las patas
# del pico fuerte.
#
# Rectangular y Hann:
#   La señal débil (-40 dB) queda tapada por los lóbulos
#   laterales del pico fuerte.
#
# Hamming, Blackman-Harris, Flattop:
#   Los lóbulos laterales están por debajo de -40 dB, así que
#   el pico débil sí se puede ver.
#
# BONUS: probá cambiar f2 a 1010 Hz (solo 10 Hz de separación).
#   ¿Qué ventana sigue separando los dos picos?
#   ¿Qué ventana los fusiona en uno solo?
# -----------------------------------------------------------

#%% Punto 6 -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal.windows as spsw

# --- Leer WAV ---
fs, data = wavfile.read("La440.wav")

# A mono si viene estereo
if data.ndim == 2:
    data = data.mean(axis=1)

# A float y quitar DC
x = data.astype(np.float64)
x = x - x.mean()

N = len(x)
Ts = 1.0 / fs

# --- Ventanas ---
ventanas = {
    "Rectangular": np.ones(N),
    "Hann": spsw.hann(N, sym=False),
    "Hamming": spsw.hamming(N, sym=False),
    "Blackman-Harris": spsw.blackmanharris(N, sym=False),
    "Flattop": spsw.flattop(N, sym=False),
}

# --- Eje de frecuencias (unilateral) ---
freqs = np.fft.rfftfreq(N, d=Ts)

# --- Graficar las ventanas en el tiempo ---
tt = np.arange(N) * Ts
plt.figure(figsize=(9,4))
for nombre, w in ventanas.items():
    plt.plot(tt, w, label=nombre, linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("w[n]")
plt.title("Ventanas en el dominio del tiempo")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --- FFT por ventana (con correccion CG) y graficos separados ---
for nombre, w in ventanas.items():
    # Coherent Gain (ganancia a una senoidal "bin-centered")
    CG = w.mean()  # = sum(w)/N

    xw = x * w
    X = np.fft.rfft(xw) / (N * CG)  # corregimos por N y CG para comparar amplitudes
    mag = np.abs(X)

    f0_est = freqs[np.argmax(mag)]

    plt.figure(figsize=(9,4))
    plt.plot(freqs, mag, label=f"{nombre} (pico ~ {f0_est:.2f} Hz)")
    plt.xlim(0, 1000)  # enfocamos cerca de 440 Hz; cambia si queres ver mas
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X(f)| (u.a.)")
    plt.title(f"Espectro con ventana {nombre}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# # --- (Opcional) Respuesta en frecuencia de las ventanas ---
# # Muestra lobulos laterales y ancho del lobulo principal (normalizado a 0 dB en DC)
# zp = 16 * N  # zero-padding para ver bien la forma
# freqs_win = np.fft.rfftfreq(zp, d=Ts)
# plt.figure(figsize=(9,5))
# for nombre, w in ventanas.items():
#     W = np.fft.rfft(w, n=zp)
#     # Normalizamos a 0 dB en DC (|W[0]| = sum(w))
#     W_db = 20*np.log10(np.maximum(np.abs(W) / (np.abs(W[0]) + 1e-12), 1e-12))
#     plt.plot(freqs_win, W_db, label=nombre, linewidth=1)
# plt.xlim(0, 2000)  # para comparar alrededor de 0-2 kHz
# plt.ylim(-120, 3)
# plt.xlabel("Frecuencia [Hz]")
# plt.ylabel("|W(f)| [dB] (normalizado a 0 dB)")
# plt.title("Respuesta en frecuencia de las ventanas (dB)")
# plt.grid(True, which="both", axis="both")
# plt.legend()
# plt.tight_layout()

plt.show()
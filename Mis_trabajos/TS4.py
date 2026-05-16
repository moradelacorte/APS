# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Sat May 16 11:29:24 2026

@author: Mora De La Corte

Descripción: 
    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

N = 1000
R = 200
fs = 1000
ts = 1/fs

omega_0 = fs/4

np.random.seed(3)

fr = np.random.uniform(-2, 2, R).reshape((R,1))

omega_1 = omega_0 + fr * (fs / N) 
omega = np.tile(omega_1, N) 

tt= (np.arange(0,N)*ts).reshape((1, N))

nn = np.tile(tt, (R,1)) 

SNR = 3 # Probar 10 y 3
print(f"SNR: {SNR}")
sigma = 10**(-SNR /20)
mu=0 # Distribucion normal

na = np.random.normal(mu, sigma, (R,N)) 

xx = np.sqrt(2) * np.sin(np.pi*2*omega*nn) + na

plt.figure(1)
plt.plot (xx.T)


# Para la ventana rectangular -------------------------------------------------
print("Ventana rectagular ------------------")
win_rec = np.ones(N)
xx_rec = xx * win_rec
X_rec = np.fft.fft(xx_rec, N, axis=1) / N

k0 = int(N * (omega_0 / fs))  # = N//4 
X_rec_k0 = X_rec[:, k0] # FFT evaluada en N/4

# ---- Estamacion amplitud ----
amp_rec = (np.abs(X_rec_k0)) * 2 # amplitud evaluar la X en N/4
ses_amp_rec = np.mean(amp_rec) - np.sqrt(2) 
var_amp_rec = np.var(amp_rec)

#---- Estamacion frecuencia ----
X_rec_max = np.argmax(np.abs(X_rec[:, :N//2]), axis=1)
frec_rec = X_rec_max * fs/N
ses_frec_rec = np.mean(frec_rec - omega_1.flatten())
var_frec_rec = np.var(frec_rec)

print("Estimadores de amplitud")
print(f"Sesgo: {ses_amp_rec}")
print(f"Varianza: {var_amp_rec}")

print("\nEstimadores de frecuencia")
print(f"Sesgo: {ses_frec_rec}")
print(f"Varianza: {var_frec_rec}")

# Para la ventana flattop -----------------------------------------------------
print("\n\nVentana flattop ------------------")
win_ft = windows.flattop(N)
xx_ft = xx * win_ft
cg_ft = np.sum(win_ft)

X_ft = np.fft.fft(xx_ft, N, axis=1) / cg_ft

k0 = int(N * (omega_0 / fs))
X_ft_k0 = X_ft[:, k0] 

# ---- Estamacion amplitud ----
amp_ft = (np.abs(X_ft_k0)) * 2
ses_amp_ft = np.mean(amp_ft) - np.sqrt(2) 
var_amp_ft = np.var(amp_ft)

#---- Estamacion frecuencia ----
X_ft_max = np.argmax(np.abs(X_ft[:, :N//2]), axis=1)
frec_ft = X_ft_max * fs/N
ses_frec_ft = np.mean(frec_ft - omega_1.flatten())
var_frec_ft = np.var(frec_ft)

print("Estimadores de amplitud")
print(f"Sesgo: {ses_amp_ft}")
print(f"Varianza: {var_amp_ft}")

print("\nEstimadores de frecuencia")
print(f"Sesgo: {ses_frec_ft}")
print(f"Varianza: {var_frec_ft}")

# Para la ventana Blackmanharris ----------------------------------------------
print("\n\nVentana Blackmanharris ------------------")
win_bh = windows.blackmanharris(N)
xx_bh = xx * win_bh
cg_bh = np.sum(win_bh)

X_bh = np.fft.fft(xx_bh, N, axis=1) / cg_bh

k0 = int(N * (omega_0 / fs))
X_bh_k0 = X_bh[:, k0] 

# ---- Estamacion amplitud ----
amp_bh = (np.abs(X_bh_k0)) * 2
ses_amp_bh = np.mean(amp_bh) - np.sqrt(2) 
var_amp_bh = np.var(amp_bh)

#---- Estamacion frecuencia ----
X_bh_max = np.argmax(np.abs(X_bh[:, :N//2]), axis=1)
frec_bh = X_bh_max * fs/N
ses_frec_bh = np.mean(frec_bh - omega_1.flatten())
var_frec_bh = np.var(frec_bh)

print("Estimadores de amplitud")
print(f"Sesgo: {ses_amp_bh}")
print(f"Varianza: {var_amp_bh}")

print("\nEstimadores de frecuencia")
print(f"Sesgo: {ses_frec_bh}")
print(f"Varianza: {var_frec_bh}")

# Para la ventana Hamming -----------------------------------------------------
print("\n\nVentana Hamming ------------------")
win_hm = windows.hamming(N)
xx_hm = xx * win_hm
cg_hm = np.sum(win_hm)

X_hm = np.fft.fft(xx_hm, N, axis=1) / cg_hm

k0 = int(N * (omega_0 / fs))
X_hm_k0 = X_hm[:, k0] 

# ---- Estamacion amplitud ----
amp_hm = (np.abs(X_hm_k0)) * 2
ses_amp_hm = np.mean(amp_hm) - np.sqrt(2) 
var_amp_hm = np.var(amp_hm)

#---- Estamacion frecuencia ----
X_hm_max = np.argmax(np.abs(X_hm[:, :N//2]), axis=1)
frec_hm = X_hm_max * fs/N
ses_frec_hm = np.mean(frec_hm - omega_1.flatten())
var_frec_hm = np.var(frec_hm)

print("Estimadores de amplitud")
print(f"Sesgo: {ses_amp_hm}")
print(f"Varianza: {var_amp_hm}")

print("\nEstimadores de frecuencia")
print(f"Sesgo: {ses_frec_hm}")
print(f"Varianza: {var_frec_hm}")

# Grafico ventanas ------------------------------------------------------------

# Usamos mucho zero-padding para ver la curva espectral suave
N_pad = N * 10
df = fs / N_pad

# 2. Armamos los índices con arange (asumiendo que N_pad es par)
k_pos = np.arange(0, N_pad // 2)      # Índices positivos: 0, 1, 2...
k_neg = np.arange(-N_pad // 2, 0)     # Índices negativos: -N/2, -N/2+1... -1

# 3. Concatenamos y multiplicamos por df
freqs_pad = np.concatenate((k_pos, k_neg)) * df

bins_pad = freqs_pad * N / fs # Eje X convertido a "bines relativos"

# FFT de las ventanas normalizadas
W_rec = np.fft.fftshift(np.abs(np.fft.fft(win_rec, N_pad)) / N)
W_ft = np.fft.fftshift(np.abs(np.fft.fft(win_ft, N_pad)) / cg_ft)
W_bh = np.fft.fftshift(np.abs(np.fft.fft(win_bh, N_pad)) / cg_bh)
W_hm = np.fft.fftshift(np.abs(np.fft.fft(win_hm, N_pad)) / cg_hm)

# Eje X para que quede centrado
bins_pad_shift = np.fft.fftshift(bins_pad)

plt.figure(2, figsize=(10, 6))
plt.plot(bins_pad_shift, W_rec, label='Rectangular')
plt.plot(bins_pad_shift, W_ft, label='Flattop')
plt.plot(bins_pad_shift, W_bh, label='Blackman-Harris')
plt.plot(bins_pad_shift, W_hm, label='Hamming')

# Área donde caen las frecuencias aleatorias (-2 a 2 bines)
plt.axvspan(-2, 2, color='gray', alpha=0.2, label='Desintonía de la señal (±2 bines)')

plt.xlim(-10, 10)
plt.title('Espectro de las Ventanas (Evaluación de Amplitud)')
plt.xlabel('Desvío respecto a N/4 (en bines)')
plt.ylabel('Ganancia de Amplitud')
plt.legend()
plt.grid(True)
plt.show()

# ---- Estamacion amplitud ----
plt.figure(3)
plt.hist(amp_rec, bins=20, color='skyblue', edgecolor='grey', 
         alpha=0.5, label='Rectangular', density=True)

plt.hist(amp_ft, bins=20, color='salmon', edgecolor='grey', 
         alpha=0.5, label='Flat-top', density=True)

plt.hist(amp_bh, bins=20, color='lightgreen', edgecolor='grey', 
         alpha=0.5, label='Blackmanharris', density=True)

plt.hist(amp_hm, bins=20, color='yellow', edgecolor='grey', 
         alpha=0.5, label='Hamming', density=True)

plt.title(f'Comparativa de Amplitudes ({R} realizaciones)')
plt.xlabel('Amplitud estimada')
plt.ylabel('Densidad de probabilidad')
plt.legend() 
plt.grid(axis='y', alpha=0.3)

# ---- Estamacion frecuencia ----
plt.figure(4)
plt.hist(frec_rec, bins=20, color='skyblue', edgecolor='black', 
         alpha=0.5, label='Rectangular', density=True)

plt.hist(frec_ft, bins=20, color='salmon', edgecolor='black', 
         alpha=0.5, label='Flat-top', density=True)

plt.hist(frec_bh, bins=20, color='lightgreen', edgecolor='black', 
         alpha=0.5, label='Blackmanharris', density=True)

plt.hist(frec_hm, bins=20, color='yellow', edgecolor='black', 
         alpha=0.5, label='Hamming', density=True)

plt.title(f'Comparativa de Frecuancias ({R} realizaciones)')
plt.xlabel('Frecuencia estimada')
plt.ylabel('Densidad de probabilidad')
plt.legend() 
plt.grid(axis='y', alpha=0.3)
# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu May  7 18:40:56 2026

@author: Mora De La Corte

Descripción: 
    superponer cada ventana, la amplitud
    segundo estimador --> en frecuencia
        argmax_k(|X(k)|) vas a tener que encontrar el argumento que maximiza, buscar estos maximos 
        
    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

N = 1000
R = 200
fs = 1000
ts = 1/fs

omega_0 = fs/4

np.random.seed(42)

fr = np.random.uniform(-2, 2, R).reshape((R,1))

omega_1 = omega_0 + fr * (fs / N) 
omega = np.tile(omega_1, N) 

tt= (np.arange(0,N)*ts).reshape((1, N))

nn = np.tile(tt, (R,1)) 

SNR = 10
sigma = 10**(-SNR /20)
mu=0 # Distribucion normal

na = np.random.normal(mu, sigma, (R,N)) 

xx = np.sqrt(2) * np.sin(np.pi*2*omega*nn) + na

plt.figure(1)
plt.plot (xx.T)

# N_pad = N * 10
# k0_pad = int(N_pad * (omega_0/fs))

# Para la ventana rectangular ------------------------------------------------
print("Ventana rectagular")
win_rec = np.ones(N)
xx_rec = xx * win_rec
X_rec = np.fft.fft(xx_rec, N, axis=1) 

k0 = int(N * (omega_0 / fs))  # = N//4 
X_rec_k0 = X_rec[:, k0] # FFT evaluada en N/4
# Grafica de la FFT con ventana rectangular 

plt.figure(2)
plt.plot(np.abs(X_rec.T))

plt.figure(3)
plt.hist(fr, bins=20, color="orange")

# Estamacion amplitud 
amp_rec = (np.abs(X_rec_k0)/N) * 2 # amplitud evaluar la X en N/4

plt.figure(4)
plt.hist(amp_rec, bins=20, color='skyblue', edgecolor='black', density=True)
plt.title(f'Histograma de la amplitud en $k_0$ ({R} realizaciones)')

ses_amp_rec = np.mean(amp_rec) - np.sqrt(2) # sesgo
plt.figure(5)
plt.hist(ses_amp_rec, bins=20, color='green', edgecolor='black', density=True)
plt.title(f'Histograma de la sesgo en $k_0$ ({R} realizaciones)')

var_amp_rec = np.var(amp_rec) # varianza
plt.figure(6)
plt.hist(var_amp_rec, bins=20, color='pink', edgecolor='black', density=True)
plt.title(f'Histograma de la varianza en $k_0$ ({R} realizaciones)')

# Estamacion frecuencia

frec_rec = np.argmax(amp_rec)
ses_frec_rec = np.mean(np.mean(frec_rec) - omega_1)
var_frec_rec = np.var(frec_rec)

print("Estimadores de amplitud", ses_amp_rec, var_amp_rec)
print("Estimadores de frecuencia", ses_frec_rec, var_frec_rec)

#%% DSP
psd_k0 = (np.abs(X_rec_k0)**2) / N
plt.figure(4)
plt.hist(psd_k0, bins=30, color='skyblue', edgecolor='black', density=True)

# Añadir detalles estéticos
plt.title(f'Histograma de la DSP en $k_0$ ({R} realizaciones)')
plt.xlabel('Potencia $|X(k_0)|^2 / N$')
plt.ylabel('Densidad de probabilidad')
plt.grid(alpha=0.3)

# Calcular la DSP de toda la matriz (R x N)
psd_completa = (np.abs(X_rec)**2) / N

# Promediar sobre las R realizaciones para reducir la varianza
psd_promedio = np.mean(psd_completa, axis=0)

# Graficar el espectro de potencia promedio
plt.figure(5)
frecuencias = np.fft.fftfreq(N, ts)
plt.plot(np.fft.fftshift(frecuencias), np.fft.fftshift(psd_promedio))
plt.title('DSP Promediada (Periodograma de Welch simplificado)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia')
plt.grid()

plt.show()






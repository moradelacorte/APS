# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Tue Apr 14 10:05:57 2026

@author: Mora De La Corte

Descripción: 
    Algunos códigos que usé para entender mejor los videos de youtube.
"""

#%% DFT 1° video

import numpy as np
import matplotlib.pyplot as plt

# 1. Parámetros iniciales
N = 8                  
n = np.arange(N)       
k = np.arange(N)       

# 2. Generamos las tres señales en el tiempo
x1 = np.sin(2 * np.pi * 1.0 * n / N)  # 1.0 ciclo
x2 = np.sin(2 * np.pi * 1.5 * n / N)  # 1.5 ciclos
x3 = np.sin(2 * np.pi * 2.5 * n / N)  # NUEVO: 2.5 ciclos

# 3. Calculamos los módulos de las DFT
X1_magnitud = np.abs(np.fft.fft(x1))
X2_magnitud = np.abs(np.fft.fft(x2))
X3_magnitud = np.abs(np.fft.fft(x3))  # NUEVO

# 4. Configuración de los gráficos (3 filas, 1 columna)
plt.figure(figsize=(10, 12)) # Hicimos la figura más alta

# --- Gráfico 1: El caso ideal (1 ciclo) ---
plt.subplot(3, 1, 1) 
plt.stem(k, X1_magnitud, basefmt="b-")
plt.title("Módulo de DFT: Exactamente 1 ciclo (Sin Fuga)", fontsize=12)
plt.ylabel("Magnitud |X[k]|", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(k)

# --- Gráfico 2: Fuga con 1.5 ciclos ---
plt.subplot(3, 1, 2) 
plt.stem(k, X2_magnitud, linefmt='C1-', markerfmt='C1o', basefmt="C1-")
plt.title("Módulo de DFT: 1.5 ciclos (Con Fuga)", fontsize=12)
plt.ylabel("Magnitud |X[k]|", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(k)

# --- Gráfico 3: NUEVO - Fuga con 2.5 ciclos ---
plt.subplot(3, 1, 3) 
plt.stem(k, X3_magnitud, linefmt='C2-', markerfmt='C2o', basefmt="C2-")
plt.title("Módulo de DFT: 2.5 ciclos (Con Fuga)", fontsize=12)
plt.xlabel("Índice k (Bins de frecuencia)", fontsize=11)
plt.ylabel("Magnitud |X[k]|", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(k)

# Ajustamos el diseño y mostramos
plt.tight_layout()
plt.show()

# Imprimimos los valores exactos de k=0 por consola para comprobar
print(f"Valor en k=0 para 1.0 ciclo : {X1_magnitud[0]:.3f}")
print(f"Valor en k=0 para 1.5 ciclos: {X2_magnitud[0]:.3f}")
print(f"Valor en k=0 para 2.5 ciclos: {X3_magnitud[0]:.3f}")

#%% DFT 2° video

import scipy.signal as sig

N = 1000
kk = np.arange(start=-10, stop=10, step=1/10)

kernel_dirchlet = np.sin(np.pi * kk) / np.sin(np.pi * kk / N)

plt.figure(2)
plt.plot(kk, np.abs(kernel_dirchlet))
plt.plot(np.array([-10,10]), np.array([0,0]), ':k')  
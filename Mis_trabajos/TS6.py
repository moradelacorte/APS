# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Jun  4 13:25:04 2026

@author: Mora De La Corte

Descripción: 
    
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Definir los coeficientes del sistema a) 
# y(n) = x(n) + x(n-1) + x(n-2) + x(n-3)
b = [1, 1, 1, 1]  # Coeficientes del numerador (X)
a = [1]           # Coeficientes del denominador (Y)

# 2. Calcular la respuesta en frecuencia
# worN=8000 genera 8000 puntos entre 0 y pi radianes/muestra
w, H = signal.freqz(b, a, worN=8000)

# 3. Calcular Módulo y Fase
modulo = np.abs(H)
fase = np.angle(H)  # En radianes

# 4. Graficar los resultados
plt.figure(figsize=(10, 6))

# Gráfico del Módulo
plt.subplot(2, 1, 1)
plt.plot(w / np.pi, modulo, color='tomato', linewidth=2)
plt.title('Respuesta en Frecuencia - Sistema a)', fontsize=14)
plt.ylabel('Módulo |H(e^{j\omega})|', fontsize=12)
plt.grid(True)

# Gráfico de la Fase
plt.subplot(2, 1, 2)
plt.plot(w / np.pi, fase, color='teal', linewidth=2)
plt.xlabel('Frecuencia Normalizada (x \pi rad/muestra)', fontsize=12)
plt.ylabel('Fase \phi(\omega) (rad)', fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
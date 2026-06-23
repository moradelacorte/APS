# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Tue Jun 23 13:45:29 2026

@author: Mora De La Corte

Descripción: 
    Simulacion para los FIR del punto 2 del 2° Parcial
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =============================================================================
# 1. CONFIGURACIÓN DE LOS FILTROS
# =============================================================================

# Filtro 1 (Inciso a - Tipo I - Normalizado)
# T1(z) = 1 - 2*z^-1 + z^-2
b1 = [1.0, -2.0, 1.0]
a1 = [1.0]

# Filtro 2 (Inciso b - Tipo IV)
# T2(z) = 1 - 3*z^-1 + 3*z^-2 - z^-3
b2 = [1.0, -3.0, 3.0, -1.0]
a2 = [1.0]

filtros = [
    {"b": b1, "a": a1, "nombre": "Filtro 1 (Tipo I - Pasa Altos)"},
    {"b": b2, "a": a2, "nombre": "Filtro 2 (Tipo IV - Pasa Altos)"}
]

# =============================================================================
# 2. CREACIÓN DE LA FIGURA
# =============================================================================
fig, axs = plt.subplots(2, 4, figsize=(18, 10))
fig.suptitle("Análisis Comparativo de Filtros FIR", fontsize=16, fontweight='bold')

for idx, f in enumerate(filtros):
    b = f["b"]
    a = f["a"]
    
    # Calcular Respuesta en Frecuencia (Módulo y Fase)
    w, h = signal.freqz(b, a, worN=8000)
    
    # Calcular Retardo de Grupo
    w_gd, gd = signal.group_delay((b, a), w=8000)
    
    # Calcular Polos y Ceros
    ceros, polos, _ = signal.tf2zpk(b, a)
    
    # -------------------------------------------------------------------------
    # Columna 1: Módulo (Lineal / Absoluto)
    # -------------------------------------------------------------------------
    ax_mag = axs[idx, 0]
    ax_mag.plot(w, np.abs(h), color='tab:orange', linewidth=2)
    ax_mag.set_title(f"{f['nombre']}\nRespuesta en Módulo")
    ax_mag.set_xlabel("Frecuencia Digital ($\Omega$ rad/muestra)")
    ax_mag.set_ylabel("|T($\Omega$)|")
    ax_mag.set_xlim(0, np.pi)
    ax_mag.grid(True, linestyle='--', alpha=0.7)
    
    # -------------------------------------------------------------------------
    # Columna 2: Fase (Radianes)
    # -------------------------------------------------------------------------
    ax_phase = axs[idx, 1]
    # Desenvolvemos la fase para ver la linealidad perfecta sin saltos de 2*pi
    fase_desenvolvente = np.unwrap(np.angle(h))
    ax_phase.plot(w, fase_desenvolvente, color='tab:blue', linewidth=2)
    ax_phase.set_title("Respuesta de Fase")
    ax_phase.set_xlabel("Frecuencia Digital ($\Omega$ rad/muestra)")
    ax_phase.set_ylabel("Fase (rad)")
    ax_phase.set_xlim(0, np.pi)
    ax_phase.grid(True, linestyle='--', alpha=0.7)
    
# -------------------------------------------------------------------------
    # Columna 3: Retardo de Grupo
    # -------------------------------------------------------------------------
    ax_gd = axs[idx, 2]
    ax_gd.plot(w_gd, gd, color='tab:green', linewidth=2)
    ax_gd.set_title("Retardo de Grupo")
    ax_gd.set_xlabel("Frecuencia Digital ($\Omega$ rad/muestra)")
    ax_gd.set_ylabel("Retardo (muestras)")
    ax_gd.set_xlim(0, np.pi)
    ax_gd.set_ylim(np.min(gd) - 1, np.max(gd) + 1) 
    ax_gd.grid(True, linestyle='--', alpha=0.7)
    
    # -------------------------------------------------------------------------
    # Columna 4: Diagrama de Polos y Ceros (Plano Z)
    # -------------------------------------------------------------------------
    ax_z = axs[idx, 3]
    # Dibujar la circunferencia unitaria
    circulo_u = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--', alpha=0.5)
    ax_z.add_artist(circulo_u)
    
    # Graficar ceros (o) y polos (x)
    if len(ceros) > 0:
        ax_z.scatter(np.real(ceros), np.imag(ceros), s=60, marker='o', 
                     facecolors='none', edgecolors='tab:blue', linewidth=2, label='Ceros')
    if len(polos) > 0:
        ax_z.scatter(np.real(polos), np.imag(polos), s=60, marker='x', 
                     color='tab:red', linewidth=2, label='Polos')
        
    ax_z.set_title("Plano Z (Polos y Ceros)")
    ax_z.set_xlabel("Real")
    ax_z.set_ylabel("Imaginario")
    ax_z.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax_z.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax_z.grid(True, linestyle='--', alpha=0.5)
    ax_z.axis('equal')
    ax_z.set_xlim(-1.5, 1.5)
    ax_z.set_ylim(-1.5, 1.5)
    ax_z.legend(loc='upper right')

# Ajustar diseño para que no se superpongan las etiquetas
plt.tight_layout()
plt.show()
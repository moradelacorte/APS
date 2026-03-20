# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Wed Mar 18 20:55:38 2026

@author: Mora De La Corte

Descripción: 
    
"""

# Bloque para crear la función pedida
import numpy as np

def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):

    # tiempo de muestreo
    ts = 1/fs
    
    # vector de tiempo
    tt = np.linspace(0, (nn-1)*ts, nn).reshape(nn,1) # El reshape es para que sean vectores de Nx1
    
    # señal senoidal
    xx = vmax * np.sin(2*np.pi*ff*tt + ph) + dc
    
    return tt, xx


vmax = np.sqrt(2)
ff = 1
N = 100
fs = 100
snr = 10 #SNR=20dB: señal 100 veces más potente que el ruido

tt, xx = mi_funcion_sen(vmax=vmax, dc=0, ff=ff, ph=0, nn=N, fs=fs)

ps = np.var(xx) # Uso la variancia para calcular la potencia de la señal
pr = 10**-(snr/10) # Despejando de la ecuacion para SNR_dB

ruido = np.random.normal(0,np.sqrt(pr),(N,1))

yy = xx + ruido

import matplotlib.pyplot as plt

plt.plot(tt,xx,label="señal limpia")
plt.plot(tt,yy,label="señal con ruido")

plt.xlabel("tiempo")
plt.ylabel("Voltaje")
plt.title("Senoidal con ruido")
plt.legend()

plt.show()

print('Potencia de la señal: ', ps)
print('Potencia del ruido: ', pr)


from scipy import signal as sig

n0 = 10 #muestras
dd = np.zeros(N)
dd[n0] = 1.

xx = xx.flatten() 

ee = sig.convolve(xx, dd)
hh = sig.convolve(ruido, np.flip(ruido)) # autocorrelacion

#cuatizacion

B = 3 #bits # mas bits se nota menos la cuantizacion
Vfs = 3 #volts

qq = Vfs/2**B

xx_in = yy

xxq = np.round(xx_in/qq) * qq

# Continuo clase 19/03 ---------------------------------------------

# secuencia de error
ee = xxq - xx_in

plt.figure(2)
plt.plot(xx_in, label = 'xx')
plt.plot(xxq, label = 'xxq')
plt.legend()

# --- Graficación con subplots ---
fig, (ax2, ax3) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle(f'Cuantización uniforme — B={B} bits, Vfs={Vfs}V, q={qq:.4f}V', fontsize=13)



# Subplot 2: señal cuantizada
ax2.plot(tt, xx_in,  color='steelblue', linewidth=1,   alpha=0.4, label='xx (original)')
ax2.plot(tt, xxq, color='tomato',    linewidth=1.2, linestyle='--', label='xxq (cuantizada)')
ax2.set_ylabel('Amplitud (V)')
ax2.set_title('Señal cuantizada xxq')
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color='black', linewidth=0.5)

# Subplot 3: error de cuantización
ax3.plot(tt, ee, color='seagreen', linewidth=1)
ax3.axhline( qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'+q/2 = +{qq/2:.4f}V')
ax3.axhline(-qq/2, color='red',   linewidth=0.8, linestyle='--', label=f'-q/2 = -{qq/2:.4f}V')
ax3.set_ylabel('Error (V)')
ax3.set_xlabel('Tiempo (s)')
ax3.set_title('Error de cuantización ee = xxq − xx')
ax3.legend(loc='upper right', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()

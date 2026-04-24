# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Fri Apr 24 11:15:34 2026

@author: Mora De La Corte

Descripción: 
     plaaylist de youtube: Clase 3 - Aproximacion de Butterworth
     
     El código no es excatamente el mismo que el que se ve en la playlist. 
     https://pytc2.readthedocs.io/en/latest/notebooks/ejnb_ComparativaOrdenes.html
"""
# Importacion de modulos generales
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

# Apariencia de figures
fig_sz_x = 13
fig_sz_y = 7
fig_dpi = 80 # dpi

fig_font_size = 11

mpl.rcParams['figure.figsize'] = (fig_sz_x, fig_sz_y)
mpl.rcParams['figure.dpi'] = fig_dpi
plt.rcParams.update({'font.size':fig_font_size})

# Importacion de modulos de PyTC2
from pytc2.sistemas_lineales import analyze_sys, pretty_print_bicuad_omegayq, tf2sos_analog, pretty_print_SOS
from pytc2.general import print_subtitle

def sim_aprox(aproxs, orders2analyze, ripple, attenuation):

    all_sys = []
    filter_names = []

    for (this_aprox, this_order, this_ripple, this_att) in zip(aproxs, orders2analyze, ripple, attenuation):

        if this_aprox == 'Butterworth':

            z,p,k = sig.buttap(this_order) # Aproximacion de Butterworth de orden this_order

            eps = np.sqrt( 10**(this_ripple/10) - 1 )
            num, den = sig.zpk2tf(z,p,k) # Obtener la funcion de transferencia
            num, den = sig.lp2lp(num, den, eps**(-1/this_order))

            z,p,k = sig.tf2zpk(num, den)

        elif this_aprox == 'Chebyshev1':

            z,p,k = sig.cheb1ap(this_order, this_ripple)

        elif this_aprox == 'Chebyshev2':

            z,p,k = sig.cheb2ap(this_order, this_att)

        elif this_aprox == 'Bessel':

            z,p,k = sig.besselap(this_order, norm='delay')

        elif this_aprox == 'Cauer':

            z,p,k = sig.ellipap(this_order, this_ripple, this_att)


        num, den = sig.zpk2tf(z,p,k)

        
        all_sys.append(sig.TransferFunction(num,den))

        this_label = this_aprox + '_ord_' + str(this_order) + '_rip_' + str(this_ripple)+ '_att_' + str(this_att)
        
        print_subtitle(this_label)
        # factorizamos en SOS's
        this_sos = tf2sos_analog(num, den)
        
        pretty_print_SOS(this_sos, mode='omegayq')
        
        filter_names.append(this_label)
        
    # el caracter "_" descarta la salida de la función
    _ = analyze_sys( all_sys, filter_names )

    return( all_sys, filter_names )

aprox_name = 'Butterworth'
#aprox_name = 'Chebyshev1'
#aprox_name = 'Chebyshev2'
#aprox_name = 'Bessel'
#aprox_name = 'Cauer'

# parametrizamos el orden para cada aproximación
orders2analyze = [2, 3, 4]

# Mismo requerimiento de ripple y atenuación
aproxs = [aprox_name] * len(orders2analyze)
ripple = [3] * len(orders2analyze) # dB \alpha_{max} <-- Sin parametrizar, lo dejo en Butterworth
attenuation = [40] * len(orders2analyze) # dB \alpha_{min} <-- Sin parametrizar, att fija


print_subtitle('Aproximaciones de Butterworth')

( all_sys, filter_names ) = sim_aprox(aproxs, orders2analyze, ripple, attenuation)

# %% Detalle a banda de paso

asy_axes = analyze_sys( all_sys, filter_names )

plt.close(2)
plt.close(3)
plt.close(4)

plt.sca(asy_axes[0][1][0])
max_ripple = np.max(np.array(ripple))
plt.ylim(np.array([-2*max_ripple, 2.]) )
plt.xlim(np.array([3. * 10.**-1, 3. * 10.**-0]) )
plt.title('Detalle de la banda de paso')

# %% Maxima planicidad

aprox_name = 'Butterworth'

# parametrizamos el ripple
ripple = [0.5, 1, 3]  # dB \alpha_{max} 

# Mismo requerimiento de ripple y atenuación
aproxs = [aprox_name] * len(orders2analyze)
orders2analyze = [4] * len(orders2analyze)
attenuation = [40] * len(orders2analyze) # dB \alpha_{min} <-- Sin parametrizar, att fija


print_subtitle('Aproximaciones de Máxima Planicidad')

( all_sys, filter_names ) = sim_aprox(aproxs, orders2analyze, ripple, attenuation)

# %% Mismo orden distinto ripple

aprox_name = 'Butterworth'

# parametrizamos el ripple
ripple = [1, 3, 6]  # dB \alpha_{max} 

# Mismo requerimiento de ripple y atenuación
aproxs = [aprox_name] * len(orders2analyze)
orders2analyze = [2] * len(orders2analyze)
attenuation = [40] * len(orders2analyze) # dB \alpha_{min} <-- Sin parametrizar, att fija


print_subtitle('Mismo orden distinto ripple')

( all_sys, filter_names ) = sim_aprox(aproxs, orders2analyze, ripple, attenuation)
# -*- coding: utf-8 -*-
"""
Materia: Análisis y Procesamiento de Señales
Fecha: Thu Apr 16 21:02:10 2026

@author: Mora De La Corte

Descripción: 
    Clase 16/04/26 
    
    
    pytc2.readthedocs.io/en/latest/notebooks/ejnb_ComparativaOrdenes.html
"""



import numpy as np
import scipy.signal as sig

this_order = 5

z,p,k = sig.buttap(this_order)

# esp = np.sqrt(10**(this_ripple/10)-1)
esp = 1

num,dem = sig.zpk2tf(z,p,k)
# z,p,k = sig.zpk2tf(num,dem)
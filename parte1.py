'''
Este script
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)

def func_modelo(params, d):
    # Modelo que utilizó Hubble.
    Ho = params
    return Ho * d

def func_a_minimizar(params, d_data, v_data):
    # Función chi-cuadrado o función error.
    return (v_data - func_modelo(params, d_data))


# Main
#Cargar datos
datos = np.loadtxt("data/hubble_original.dat")
d = datos[:, 0] # Distancia [Mpc]
v = datos[:, 1] # Velocidad [km/s]

# Setup
# Adivinanza para el valor de Ho
a0 = 4
# Minimizacion del chi-cuadrado
resultado = leastsq(func_a_minimizar, a0, args=(d, v))
print "Status: ", resultado[1]
print "mejor fit para Ho: ", resultado[0]

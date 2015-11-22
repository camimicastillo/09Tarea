'''
Este script
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)

def func_modelo(params, x):
    # Modelo que utilizó Hubble: v = Ho * d (Caso a)
    # d = (1 / Ho) * v (Caso b)
    Ho = params
    return Ho * x


def func_a_minimizar(params, x_data, y_data):
    # Función chi-cuadrado o función error = v - Ho * d (Caso a)
    # Función chi-cuadrado o función error = d - v / Ho (Caso b)
    return (y_data - func_modelo(params, x_data))


# Main
# Cargar datos
datos = np.loadtxt("data/hubble_original.dat")
d = datos[:, 0] # Distancia [Mpc]
v = datos[:, 1] # Velocidad [km/s]

# Setup
# Adivinanza para el valor de Ho, caso a
a0 = 4
# Minimizacion del chi-cuadrado caso a
resultado_a = leastsq(func_a_minimizar, a0, args=(d, v))
print "Status para a: ", resultado_a[1]
print "mejor fit para Ho, caso a: ", resultado_a[0]

#Adivinanza para el valor de 1/Ho, caso b
a1 = 5
# Minimizacion del chi-cuadrado caso b
resultado_b = leastsq(func_a_minimizar, a1, args=(v, d))
print "Status para b: ", resultado_b[1]
# En este caso el parámetro que optimizamos fue 1/Ho, pero Ho es el
# valor que buscamos
Ho_b = 1 / resultado_b[0]
print "mejor fit para Ho, caso b: ", Ho_b

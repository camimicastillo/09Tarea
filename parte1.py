'''
Este script
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)

def func_modelo_a(params, d):
    # Modelo que utilizó Hubble: v = Ho * d (Caso a)
    Ho = params
    return Ho * d

def func_modelo_b(params, v):
    # Modelo que utilizó Hubble: d = v / Ho (Caso b)
    Ho = params
    return v / Ho


def func_a_minimizar_a(params, d_data, v_data):
    # Función chi-cuadrado o función error = v - Ho * d
    return (v_data - func_modelo_a(params, d_data))

def func_a_minimizar_b(params, v_data, d_data):
    # Función chi-cuadrado o función error = d - v / Ho
    return (d_data - func_modelo_b(params, v_data))


# Main
#Cargar datos
datos = np.loadtxt("data/hubble_original.dat")
d = datos[:, 0] # Distancia [Mpc]
v = datos[:, 1] # Velocidad [km/s]

# Setup
# Adivinanza para el valor de Ho
a0 = 4
# Minimizacion del chi-cuadrado caso a
resultado_a = leastsq(func_a_minimizar_a, a0, args=(d, v))
print "Status para a: ", resultado_a[1]
print "mejor fit para Ho, caso a: ", resultado_a[0]

#Minimizacion del chi-cuadrado caso b
resultado_b = leastsq(func_a_minimizar_b, a0, args=(v, d))
print "Status para b: ", resultado_b[1]
print "mejor fit para Ho, caso b: ", resultado_b[0]

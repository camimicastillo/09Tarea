'''
Este script
'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)

np.random.seed(9102)

# Main
# Cargar datos
datos = np.loadtxt("data/DR9Q.dat", usecols=(80,81,82,83))
flujo_i = datos[:, 0] # Distancia [Mpc]
error_i = datos[:, 1] # Velocidad [km/s]
flujo_z = datos[:, 2] # Distancia [Mpc]
error_z = datos[:, 3] # Velocidad [km/s]

# Setup
# Cambiar unidades
Flujo_i = flujo_i * 3.631
Error_i = error_i * 3.631
Flujo_z = flujo_z * 3.631
Error_z = error_z * 3.631


# Metodo de Monte Carlo
N_mc = 10000
# Ajuste lineal de la forma y = a * x + b
a = np.zeros(N_mc)
b = np.zeros(N_mc)

for i in range(N_mc):
    r = np.random.normal(0, 1, size=len(Flujo_i))
    falsos_i = Flujo_i + Error_i * r
    falsos_z = Flujo_z + Error_z * r
    a[i], b[i] = np.polyfit(falsos_i, falsos_z, 1)

a = np.sort(a)
b = np.sort(b)
lim_inf_a = a[int(N_mc * 0.025)]
lim_sup_a = a[int(N_mc * 0.975)]
lim_inf_b = b[int(N_mc * 0.025)]
lim_sup_b = b[int(N_mc * 0.975)]

print "El intervalo de confianza para " \
          "la pendiente a al 95% es: [{}:{}]".format(lim_inf_a, lim_sup_a)
print "El intervalo de confianza para el " \
          "coef de posicion b al 95% es: [{}:{}]".format(lim_inf_b, lim_sup_b)

# Polyfit con los datos reales
a_real, b_real= np.polyfit(Flujo_i, Flujo_z, 1)
print a_real, 'Coeficiente a del ajuste lineal a*x+b'
print b_real, 'Coeficiente b del ajuste lineal a*x+b'

# Graficos
x = np.linspace(-100, 500, 10000)
y = a_real * x + b_real

fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)

ax1.plot(x, y, 'r', label='Ajuste')
plt.errorbar(Flujo_i, Flujo_z, xerr=Error_i, yerr=Error_z, fmt='o',
             label='Datos experimentales')

ax1.set_xlabel("Flujo Banda i [$10^{-6} Jy$]")
ax1.set_ylabel("Flujo Banda z [$10^{-6} Jy$]")
ax1.set_title("Grafico de Flujo de Banda en i vs Flujo de Banda en z")

plt.legend(loc='upper left')
plt.draw()
plt.show()

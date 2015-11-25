'''
Este script calcula la constante de Hubble con un intervalo de confianza
del 95% (algoritmo Bootstrap), para los datos del archivo hubble_original.dat
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import (leastsq, curve_fit)

np.random.seed(500)


def func_modelo(params, x):
    # Modelo que utilizó Hubble: v = Ho * d (Caso a)
    # d = (1 / Ho) * v (Caso b)
    Ho = params
    return Ho * x


def func_a_minimizar(params, x_data, y_data):
    # Función chi-cuadrado o función error = v - Ho * d (Caso a)
    # Función chi-cuadrado o función error = d - v / Ho (Caso b)
    return (y_data - func_modelo(params, x_data))


def bootstrap(data, x_data):
    # Simulacion de bootstrap para encontrar el intervalo de confianza al 95%
    N = data.shape[0]
    N_boot = int(len(x_data)) ** 2
    H = np.zeros(N_boot)
    for i in range(N_boot):
        s = np.random.randint(low=0, high=N, size=N)
        datos_falsos = data[s][s]
        x = datos_falsos[:, 0]
        y = datos_falsos[:, 1]
        H1, S1 = leastsq(func_a_minimizar, a0, args=(x, y))
        casi_H2, S2 = leastsq(func_a_minimizar, a0, args=(y, x))
        H2 = 1 / casi_H2
        Hprom = (H1 + H2) / 2
        H[i] = Hprom
    H_ord = np.sort(H)
    lim_inf = H_ord[int(N_boot * 0.025)]
    lim_sup = H_ord[int(N_boot * 0.975)]
    print "El intervalo de confianza " \
          "al 95% es: [{}:{}]".format(lim_inf, lim_sup)
    return H, lim_inf, lim_sup


# Main
# Cargar datos
datos = np.loadtxt("data/hubble_original.dat")
d = datos[:, 0]  # Distancia [Mpc]
v = datos[:, 1]  # Velocidad [km/s]

# Setup
# Adivinanza para el valor de Ho, caso a
a0 = 4
# Minimizacion del chi-cuadrado caso a
resultado_a = leastsq(func_a_minimizar, a0, args=(d, v))
print "Status para a: ", resultado_a[1]
print "mejor fit para Ho, caso a: ", resultado_a[0]
Ho_a = resultado_a[0]

# Adivinanza para el valor de 1/Ho, caso b
a1 = 5
# Minimizacion del chi-cuadrado caso b
resultado_b = leastsq(func_a_minimizar, a1, args=(v, d))
print "Status para b: ", resultado_b[1]
# En este caso el parámetro que optimizamos fue 1/Ho, pero Ho es el
# valor que buscamos
Ho_b = 1 / resultado_b[0]
print "mejor fit para Ho, caso b: ", Ho_b

# Como se busca una alternativa que sea simétrica se propone promediar los
# valores obtenidos y utilizar este promedio como el Ho óptimo
Ho_prom = (Ho_a + Ho_b) / 2
print Ho_prom


# Se grafican los valores experimentales y el ajuste con la minimización
# de la función chi-cuadrado (usando el Ho óptimo encontrado)
fig = plt.figure()
fig.clf()
ax1 = fig.add_subplot(111)

ax1.plot(d, v, '*', label="Datos experimentales")
ax1.plot(d, Ho_prom * d, label="Ajuste usando $H_o$ optimo")

ax1.set_xlim([-0.5, 2.5])
ax1.set_xlabel("Distancia $[Mpc]$")
ax1.set_ylabel("Velocidad $[km/s]$")
ax1.set_title("Grafico de distancia $[Mpc]$ versus velocidad $[km/s]$")

plt.legend(loc='upper left')
plt.draw()
plt.show()

# Intervalo de confianza
interv_confianza, lim_inf, lim_sup = bootstrap(datos, d)

# Histograma
fig2 = plt.figure(2)
fig2.clf()
plt.hist(interv_confianza, bins=50, facecolor='g', alpha=0.5)
plt.axvline(Ho_prom, color='r', label="Valor optimo encontrado")
plt.axvline(lim_inf, color='b',
            label="Intervalo de confianza al 95$\%$")
plt.axvline(lim_sup, color='b')
plt.title("Histograma $H_0$")
plt.legend(fontsize=11)
plt.draw()
plt.show()

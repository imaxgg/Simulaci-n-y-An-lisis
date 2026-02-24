# ================================================================#
# Simulación y análisis de señales con la transformada de Fourier #
# ================================================================#

"""
Este programa:
1) Genera señales en dominio del tiempo
2) Calcula su Transformada de Fourier
3) Grafica magnitud y fase
4) Verifica propiedades de Fourier
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# DEFINICIÓN DEL TIEMPO
# =========================
# Se define frecuencia de muestreo y vector de tiempo
fs = 1000
t = np.arange(-1, 1, 1/fs)

# =========================
# DEFINICIÓN DE SEÑALES
# =========================
# Pulso rectangular
rect = np.where(np.abs(t) < 0.2, 1, 0)

# Escalón unitario
step = np.heaviside(t, 1)

# Señal senoidal de 5 Hz
sine = np.sin(2*np.pi*5*t)

# =========================
# CÁLCULO FFT
# =========================
def calc_fft(x):
    """
    Calcula la Transformada de Fourier de una señal
    y genera el vector de frecuencias
    """
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1/fs))
    return f, X

# FFT de cada señal
f_rect, X_rect = calc_fft(rect)
f_step, X_step = calc_fft(step)
f_sine, X_sine = calc_fft(sine)

# =========================
# VISUALIZACIÓN TIEMPO
# =========================
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(t,rect); plt.title("Pulso rectangular")
plt.subplot(3,1,2); plt.plot(t,step); plt.title("Escalón")
plt.subplot(3,1,3); plt.plot(t,sine); plt.title("Senoidal")
plt.tight_layout()
plt.show()

# =========================
# VISUALIZACIÓN MAGNITUD
# =========================
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_rect,np.abs(X_rect)); plt.title("FFT Pulso")
plt.subplot(3,1,2); plt.plot(f_step,np.abs(X_step)); plt.title("FFT Escalón")
plt.subplot(3,1,3); plt.plot(f_sine,np.abs(X_sine)); plt.title("FFT Seno")
plt.tight_layout()
plt.show()

# =========================
# VISUALIZACIÓN FASE
# =========================
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_rect,np.angle(X_rect)); plt.title("Fase Pulso")
plt.subplot(3,1,2); plt.plot(f_step,np.angle(X_step)); plt.title("Fase Escalón")
plt.subplot(3,1,3); plt.plot(f_sine,np.angle(X_sine)); plt.title("Fase Seno")
plt.tight_layout()
plt.show()

# =========================
# PROPIEDADES FOURIER
# =========================

# Linealidad: suma de señales
lin = rect + sine
f_lin, X_lin = calc_fft(lin)

# Desplazamiento temporal
shift = np.roll(sine, 200)
f_shift, X_shift = calc_fft(shift)

# Escalamiento temporal (mayor frecuencia)
scale = np.sin(2*np.pi*10*t)
f_scale, X_scale = calc_fft(scale)

plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_lin,np.abs(X_lin)); plt.title("Linealidad")
plt.subplot(3,1,2); plt.plot(f_shift,np.abs(X_shift)); plt.title("Desplazamiento")
plt.subplot(3,1,3); plt.plot(f_scale,np.abs(X_scale)); plt.title("Escalamiento")
plt.tight_layout()
plt.show()
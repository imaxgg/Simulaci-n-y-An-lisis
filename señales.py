# ================================================================#
# Actividad 2: Simulación y análisis de señales con Fourier     #
# ================================================================#

"""
Este programa:
1) Genera señales elementales en el dominio del tiempo
2) Calcula la Transformada de Fourier
3) Grafica magnitud y fase
4) Verifica propiedades de Fourier: linealidad, desplazamiento y escalamiento
"""

import numpy as np
import matplotlib.pyplot as plt

# =========================
# DEFINICIÓN DEL TIEMPO
# =========================
fs = 1000                  # Frecuencia de muestreo (Hz)
t = np.arange(-1, 1, 1/fs) # Vector de tiempo

# =========================
# DEFINICIÓN DE SEÑALES
# =========================
rect = np.where(np.abs(t) < 0.2, 1, 0)  # Pulso rectangular
step = np.heaviside(t, 1)              # Escalón unitario
sine = np.sin(2*np.pi*5*t)             # Senoidal 5 Hz

# =========================
# FUNCIÓN PARA FFT
# =========================
def calc_fft(x):
    """
    Calcula la Transformada de Fourier de una señal
    y genera el vector de frecuencias centrado en cero.
    """
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1/fs))
    return f, X

# FFT de cada señal
f_rect, X_rect = calc_fft(rect)
f_step, X_step = calc_fft(step)
f_sine, X_sine = calc_fft(sine)

# =========================
# VISUALIZACIÓN EN EL DOMINIO DEL TIEMPO
# =========================
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(t, rect); plt.title("Pulso rectangular")
plt.subplot(3,1,2); plt.plot(t, step); plt.title("Escalón unitario")
plt.subplot(3,1,3); plt.plot(t, sine); plt.title("Señal senoidal 5 Hz")
plt.tight_layout()
plt.show()

# =========================
# VISUALIZACIÓN DE MAGNITUD FFT
# =========================
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_rect, np.abs(X_rect)); plt.title("FFT Pulso rectangular")
plt.subplot(3,1,2); plt.plot(f_step, np.abs(X_step)); plt.title("FFT Escalón unitario")
plt.subplot(3,1,3); plt.plot(f_sine, np.abs(X_sine)); plt.title("FFT Senoidal 5 Hz")
plt.tight_layout()
plt.show()

# =========================
# VISUALIZACIÓN DE FASE FFT
# =========================
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_rect, np.angle(X_rect)); plt.title("Fase Pulso rectangular")
plt.subplot(3,1,2); plt.plot(f_step, np.angle(X_step)); plt.title("Fase Escalón unitario")
plt.subplot(3,1,3); plt.plot(f_sine, np.angle(X_sine)); plt.title("Fase Senoidal 5 Hz")
plt.tight_layout()
plt.show()

# =========================
# PROPIEDADES DE FOURIER
# =========================
# Linealidad: suma de pulso + seno
lin = rect + sine
f_lin, X_lin = calc_fft(lin)

# Desplazamiento temporal: desplazamiento de la senoidal
shift = np.roll(sine, 200)
f_shift, X_shift = calc_fft(shift)

# Escalamiento temporal: frecuencia aumentada (10 Hz)
scale = np.sin(2*np.pi*10*t)
f_scale, X_scale = calc_fft(scale)

plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_lin, np.abs(X_lin)); plt.title("Linealidad (Pulso + Seno)")
plt.subplot(3,1,2); plt.plot(f_shift, np.abs(X_shift)); plt.title("Desplazamiento temporal (Senoidal 5 Hz)")
plt.subplot(3,1,3); plt.plot(f_scale, np.abs(X_scale)); plt.title("Escalamiento temporal (Senoidal 10 Hz)")
plt.tight_layout()
plt.show()
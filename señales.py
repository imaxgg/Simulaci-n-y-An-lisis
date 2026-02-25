# ================================================================#
# Simulación, análisis de señales y filtrado con Fourier y filtros #
# ================================================================#

"""
Este programa:
1) Genera señales en dominio del tiempo
2) Calcula su Transformada de Fourier
3) Grafica magnitud y fase
4) Verifica propiedades de Fourier
5) Diseña y aplica filtros FIR e IIR (Pasa Bajos, Pasa Altos, Pasa Banda)
6) Compara señales antes y después del filtrado
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =========================
# DEFINICIÓN DEL TIEMPO
# =========================
fs = 1000                  # Frecuencia de muestreo (Hz)
t = np.arange(-1, 1, 1/fs) # Vector de tiempo

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
# FUNCION FFT
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
plt.subplot(3,1,3); plt.plot(t,sine); plt.title("Senoidal 5 Hz")
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
lin = rect + sine
f_lin, X_lin = calc_fft(lin)
shift = np.roll(sine, 200)
f_shift, X_shift = calc_fft(shift)
scale = np.sin(2*np.pi*10*t)
f_scale, X_scale = calc_fft(scale)

plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_lin,np.abs(X_lin)); plt.title("Linealidad")
plt.subplot(3,1,2); plt.plot(f_shift,np.abs(X_shift)); plt.title("Desplazamiento")
plt.subplot(3,1,3); plt.plot(f_scale,np.abs(X_scale)); plt.title("Escalamiento")
plt.tight_layout()
plt.show()

# =========================
# SEÑAL COMPUESTA + RUIDO
# =========================
comp = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*50*t)
noise = 0.5*np.random.randn(len(t))
noisy = comp + noise

# =========================
# FILTROS IIR (PASA BAJOS, PASA ALTOS, PASA BANDAS)
# =========================
# Pasa Bajos: deja pasar < 10 Hz
b_low, a_low = signal.butter(4, 10/(fs/2), btype='low')
low_filtered = signal.filtfilt(b_low, a_low, noisy)

# Pasa Altos: deja pasar > 20 Hz
b_high, a_high = signal.butter(4, 20/(fs/2), btype='high')
high_filtered = signal.filtfilt(b_high, a_high, noisy)

# Pasa Bandas: deja pasar 4-6 Hz (seno 5 Hz)
b_band, a_band = signal.butter(4, [4/(fs/2), 6/(fs/2)], btype='bandpass')
band_filtered = signal.filtfilt(b_band, a_band, noisy)

# =========================
# FILTRO FIR PASA BAJOS
# =========================
fir_coeff = signal.firwin(101, 10/(fs/2))
fir_filtered = signal.filtfilt(fir_coeff, [1], noisy)

# =========================
# VISUALIZACIÓN SEÑALES FILTRADAS (TIEMPO)
# =========================
plt.figure(figsize=(12,8))
plt.subplot(4,1,1); plt.plot(t,noisy); plt.title("Señal ruidosa")
plt.subplot(4,1,2); plt.plot(t,low_filtered); plt.title("Filtro IIR Pasa Bajos")
plt.subplot(4,1,3); plt.plot(t,high_filtered); plt.title("Filtro IIR Pasa Altos")
plt.subplot(4,1,4); plt.plot(t,band_filtered); plt.title("Filtro IIR Pasa Bandas")
plt.tight_layout()
plt.show()

# =========================
# VISUALIZACIÓN FFT SEÑALES FILTRADAS
# =========================
f_noisy, X_noisy = calc_fft(noisy)
f_low, X_low = calc_fft(low_filtered)
f_high, X_high = calc_fft(high_filtered)
f_band, X_band = calc_fft(band_filtered)

plt.figure(figsize=(12,8))
plt.subplot(4,1,1); plt.plot(f_noisy,np.abs(X_noisy)); plt.title("FFT Señal ruidosa")
plt.subplot(4,1,2); plt.plot(f_low,np.abs(X_low)); plt.title("FFT Filtro IIR Pasa Bajos")
plt.subplot(4,1,3); plt.plot(f_high,np.abs(X_high)); plt.title("FFT Filtro IIR Pasa Altos")
plt.subplot(4,1,4); plt.plot(f_band,np.abs(X_band)); plt.title("FFT Filtro IIR Pasa Bandas")
plt.tight_layout()
plt.show()
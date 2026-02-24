import numpy as np
import matplotlib.pyplot as plt

# TIEMPO
fs = 1000
t = np.arange(-1, 1, 1/fs)

# SEÑALES
rect = np.where(np.abs(t) < 0.2, 1, 0)
step = np.heaviside(t, 1)
sine = np.sin(2*np.pi*5*t)

# FFT
def calc_fft(x):
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1/fs))
    return f, X

f_rect, X_rect = calc_fft(rect)
f_step, X_step = calc_fft(step)
f_sine, X_sine = calc_fft(sine)

# TIEMPO
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(t,rect); plt.title("Pulso rectangular")
plt.subplot(3,1,2); plt.plot(t,step); plt.title("Escalón")
plt.subplot(3,1,3); plt.plot(t,sine); plt.title("Senoidal")
plt.tight_layout()
plt.show()

# MAGNITUD
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_rect,np.abs(X_rect)); plt.title("FFT Pulso")
plt.subplot(3,1,2); plt.plot(f_step,np.abs(X_step)); plt.title("FFT Escalón")
plt.subplot(3,1,3); plt.plot(f_sine,np.abs(X_sine)); plt.title("FFT Seno")
plt.tight_layout()
plt.show()

# FASE
plt.figure(figsize=(12,6))
plt.subplot(3,1,1); plt.plot(f_rect,np.angle(X_rect)); plt.title("Fase Pulso")
plt.subplot(3,1,2); plt.plot(f_step,np.angle(X_step)); plt.title("Fase Escalón")
plt.subplot(3,1,3); plt.plot(f_sine,np.angle(X_sine)); plt.title("Fase Seno")
plt.tight_layout()
plt.show()

# PROPIEDADES
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
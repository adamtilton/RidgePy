import numpy as np
import matplotlib.pyplot as plt

MODE_COUNT = 6

mode_numbers     = [m for m in range(1, MODE_COUNT + 1)]
sample_frequency = 16000.0
time             = np.arange(0, 0.2, 1./sample_frequency)
F                = 150.0
A                = [1.0, 0.25, 0.0, 0.7, -0.4, -0.8]
B                = [0.5, 0.0, -0.2, -0.8, 0.9, 1.0]

data = np.zeros((len(time), 16))
data[:,0] = time
data[:,1] = np.mod(2*np.pi*F*time, 2*np.pi)
for mode in mode_numbers:
    data[:,2] += A[mode-1]*np.cos(mode*data[:,1]) + B[mode-1]*np.sin(mode*data[:,1])
data[:,3] = F*np.ones_like(time)

data[:,4] = A[0]*np.ones_like(time)
data[:,5] = A[1]*np.ones_like(time)
data[:,6] = A[2]*np.ones_like(time)
data[:,7] = A[3]*np.ones_like(time)
data[:,8] = A[4]*np.ones_like(time)
data[:,9] = A[5]*np.ones_like(time)

data[:,10] = B[0]*np.ones_like(time)
data[:,11] = B[1]*np.ones_like(time)
data[:,12] = B[2]*np.ones_like(time)
data[:,13] = B[3]*np.ones_like(time)
data[:,14] = B[4]*np.ones_like(time)
data[:,15] = B[5]*np.ones_like(time)

np.savetxt(
    "bin/simulation/multi_mode_example.txt",
    data,
    header="time, phase, observation, F, A1, A2, A3, A4, A5, A6, B1, B2, B3, B4, B5, B6"
)

width  = 12.0
height = width / 1.618 / 2.


fig, ax  = plt.subplots(nrows=MODE_COUNT+1, ncols=1, sharex=True, sharey=True, figsize=(width, 3*height))
ax[0].set_title("Multi-mode Input signal")
ax[0].plot(data[:,0], data[:,2], linestyle='-', label=r'$Y(t)$')
for mode_ndx in range(MODE_COUNT):
    true_mode_observation = data[:,mode_ndx+4]*np.cos(mode_numbers[mode_ndx] * data[:,1]) + data[:,mode_ndx+10]*np.sin(mode_numbers[mode_ndx] * data[:,1])
    ax[mode_ndx+1].set_title(f"Mode {mode_ndx} contribution.")
    ax[mode_ndx+1].plot(data[:,0], true_mode_observation, linestyle='-')
ax[-1].set_xlabel('time [seconds]')

plt.show()
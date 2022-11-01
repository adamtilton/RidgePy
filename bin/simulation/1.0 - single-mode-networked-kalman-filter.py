import numpy as np
import matplotlib.pyplot as plt

from ridgepy.libridgewrapper.config import MAIN_CONFIG
from ridgepy.libridgewrapper.kalman_filter_mode import KalmanFilterMode
from ridgepy.libridgewrapper.kalman_filter_network import KalmanFilterNetwork

# This example processes the single_mode_example file created by the
# bin/simulate/0.0-simulate-single-mode.py file.
data       = np.loadtxt("bin/simulation/single_mode_example.txt")
data_count = len(data)

# The C library is compiled for a specific number of modes. For this example
# file the library should be compiled for 1 mode
MODE_COUNT = MAIN_CONFIG['MODE_COUNT']
assert MODE_COUNT == 1, "Recompile the C library for MODE_COUNT = 1"
mode_numbers = [m for m in range(1, MODE_COUNT + 1)]

# We assume knowledge of the frequency of the single, but no knowledge of the
# phase or the coefficients.
FREQUENCY  = 150.0
PHASE      = 0.0
COEFFICIENTS = np.zeros((MODE_COUNT,2))

frequency_sample = 16000.0
time_delta = 1./frequency_sample

kf_network = KalmanFilterNetwork(
    mode_numbers,
    0.001,
    COEFFICIENTS,
    [
        [0.0, 0.1, 0.1],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    1.0
)

estimate     = np.zeros((len(data), 1))
error        = np.zeros((len(data), 1))
predictions  = np.zeros((len(data), MODE_COUNT))
phase        = np.zeros((len(data), 1))
frequency    = np.zeros((len(data), 1))
gain         = np.zeros((len(data), 3))
coefficients = np.zeros((len(data), 2))

power        = np.zeros((MODE_COUNT, len(data)))

for ndx, row in enumerate(data):
    observation = row[2]

    if ndx == 0:
        frequency[ndx] = FREQUENCY
        phase[ndx] = PHASE
    elif ndx > 0:
        frequency[ndx]  = frequency[ndx-1]
        phase[ndx]      = phase[ndx-1]
        phase[ndx]     += 2*np.pi*frequency[ndx]*time_delta

    phase[ndx]  = np.mod(phase[ndx], 2*np.pi)

    kf_network.prior_update(phase[ndx])
    kf_network.posterior_update(observation)

    for mode_ndx in range(MODE_COUNT):
        frequency[ndx] += 0.005*kf_network.kf_modes[mode_ndx].phase_update / MODE_COUNT

    estimate[ndx,0] = kf_network.prediction
    error[ndx,0] = kf_network.error
    gain[ndx, 0] = kf_network.kf_modes[true_mode_ndx].gain[0]
    gain[ndx, 1] = kf_network.kf_modes[true_mode_ndx].gain[1]
    gain[ndx, 2] = kf_network.kf_modes[true_mode_ndx].gain[2]
    coefficients[ndx, 0] = kf_network.kf_modes[true_mode_ndx].coefficients[0]
    coefficients[ndx, 1] = kf_network.kf_modes[true_mode_ndx].coefficients[1]


    for mode_ndx in range(MODE_COUNT):
        power[mode_ndx, ndx] = np.sqrt(
            kf_network.kf_modes[mode_ndx].coefficients[0]**2 +
            kf_network.kf_modes[mode_ndx].coefficients[1]**2
        )

width  = 12.0
height = width / 1.618 / 2.

fig, ax  = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False, figsize=(width, height))
plot_line(ax[0], data[:,0], gain[:,0], color=colors['teal'], linestyle='-', label=r'$V_0(t)$')
plot_line(ax[0], data[:,0], error[:,0], color=colors['pink'], linestyle='-', label=r'$e(t)$')
plot_line(ax[1], data[:,0], gain[:,1], color=colors['green'], linestyle='-', label=r'$V_1(t)$')
plot_line(ax[1], data[:,0], error[:,0], color=colors['pink'], linestyle='-', label=r'$e(t)$')
plot_line(ax[2], data[:,0], gain[:,2], color=colors['yellow'], linestyle='-', label=r'$V_2(t)$')
plot_line(ax[2], data[:,0], error[:,0], color=colors['pink'], linestyle='-', label=r'$e(t)$')
fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.85))
ax[2].set_xlabel('time [seconds]')

fig, ax  = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(width, height))
plot_line(ax[0], data[:,0], coefficients[:,0], color=colors['teal'], linestyle='-', label=r'$\hat{A}(t)$')
plot_line(ax[1], data[:,0], coefficients[:,1], color=colors['yellow'], linestyle='-', label=r'$\hat{B}(t)$')
fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.85))
ax[1].set_xlabel('time [seconds]')


phase_error = np.mod(data[:,1].flatten() - mode_numbers[true_mode_ndx]*phase.flatten(), 2*np.pi)
fig, ax  = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(width, height))
plot_phase(ax[0], data[:,0], data[:,1], color=colors['pink'], marker='o', label=r'$\theta(t)$')
plot_phase(ax[0], data[:,0], np.mod(mode_numbers[true_mode_ndx]*phase, 2*np.pi), color=colors['teal'], marker='o', label=r'$\hat{\theta}(t)$')
plot_phase(ax[1], data[:,0], phase_error, color=colors['green'], marker='o', label=r'$\theta(t) - \hat{\theta}(t)$')
fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.85))
ax[1].set_xlabel('Time [seconds]')

frequency_error = data[:,3].flatten() - frequency.flatten()
fig, ax  = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(width, height))
plot_line(ax[0], data[:,0], data[:,3], color=colors['pink'], linestyle='-', label=r'$f(t)$')
plot_line(ax[0], data[:,0], frequency, color=colors['teal'], linestyle='--', label=r'$\hat{f}(t)$')
plot_line(ax[1], data[:,0], frequency_error, color=colors['green'], linestyle='-', label=r'$f(t)-\hat{f}(t)$')
fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.85))
ax[1].set_xlabel('time [seconds]')


fig, ax  = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=(width, height))
plot_line(ax[0], data[:,0], data[:,2], color=colors['pink'], linestyle='-', label=r'$Y(t)$')
plot_line(ax[0], data[:,0], estimate[:,0], color=colors['teal'], linestyle='--', label=r'$\hat{h}(t)$')
plot_line(ax[1], data[:,0], error[:,0], color=colors['green'], linestyle='-', label=r'$e(t)$')
fig.legend(ncol=3, bbox_to_anchor=(0.5, 0.85))
ax[1].set_xlabel('time [seconds]')

fig, ax  = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=True, figsize=(width, height))
ax[0].imshow(power, aspect='auto', cmap='cool', extent=[0, data[-1,0], 0, FREQUENCY_MAX], origin='lower')
ax[1].imshow(power, aspect='auto', cmap='cool', extent=[0, data[-1,0], 0, 1200], origin='lower')

fig, ax  = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(width, height))
fft = sp.fft.rfft(data[:,2])
freq = sp.fft.rfftfreq(data_count, d=time_delta)
ax[0, 0].set_title("Calculated DTFT Amplitude")
ax[0, 0].stem(freq, 2.0/data_count * np.abs(fft))
ax[0, 1].set_title("Calculated KF Amplitude")
ax[0, 1].stem(frequency[-1] * np.array(mode_numbers), power[:,-1])

print(freq[1])

ax[1, 0].set_title("Zoomed DTFT Amplitude")
ax[1, 0].stem(freq, 2.0/data_count * np.abs(fft))
ax[1, 0].set_xlim([0, 1200])
ax[1, 1].set_title("Zoomed KF Amplitude")
ax[1, 1].stem(frequency[-1] * np.array(mode_numbers), power[:,-1])
ax[1, 1].set_xlim([0, 1200])

plt.show()
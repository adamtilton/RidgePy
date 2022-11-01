import numpy as np
import matplotlib.pyplot as plt

sample_frequency = 16000.0
time             = np.arange(0, 0.2, 1./sample_frequency)
F                = 150.0
A                = 1.0
B                = 0.5

data = np.zeros((len(time), 6))
data[:,0] = time
data[:,1] = np.mod(2*np.pi*F*time, 2*np.pi)
data[:,2] = A*np.cos(data[:,1]) + B*np.sin(data[:,1])
data[:,3] = F*np.ones_like(time)
data[:,4] = A*np.ones_like(time)
data[:,5] = B*np.ones_like(time)

np.savetxt(
    "bin/simulation/single_mode_example.txt",
    data,
    header="time, phase, observation, A, B, F"
)

width  = 12.0
height = width / 1.618 / 2.

fig, ax  = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False, figsize=(width, height))
ax.plot(data[:,0], data[:,2], linestyle='-', label=r'$Y(t)$')
fig.legend(ncol=1, bbox_to_anchor=(0.5, 0.85))
ax.set_xlabel('time [seconds]')

plt.show()
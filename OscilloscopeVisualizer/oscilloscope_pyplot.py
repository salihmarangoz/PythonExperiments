
# ref:
# - https://python-sounddevice.readthedocs.io/en/0.4.3/examples.html#real-time-text-mode-spectrogram
# - https://soundcard.readthedocs.io/en/latest/

import soundcard as sc
import numpy as np
import matplotlib.pyplot as plt

mics = sc.all_microphones(include_loopback=True)

plt.ion()
plt.figure(figsize=(4,4))
for m in mics:
    if m.isloopback:
        with m.recorder(samplerate=44100) as mic: 
            while True:
                data = mic.record(numframes=44100//20)
                print(np.max(data), np.min(data), data.shape)
                plt.cla()
                plt.axis("off")
                plt.xlim(-0.3, 0.3)
                plt.ylim(-0.3, 0.3)
                plt.plot(data[:, 0], data[:, 1], '.', alpha=0.2, markersize=1.5)
                plt.draw()
                plt.pause(0.0001)

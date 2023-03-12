# Description: Plot the waveform of a channel of a sample audio file

import matplotlib.pyplot as plt
import numpy as np
from audiohandler import AudioWave

audio = AudioWave.from_file("../../tests/test_sound/test.wav")

t_audio = len(audio) / audio.samplerate

signal_array = audio.get_channel(0).array / 2**(audio.bit - 1)

times = np.linspace(0, t_audio, num=len(audio))

plt.figure(figsize=(15, 5))
plt.plot(times, signal_array)
plt.title("Signal Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.xlim(0, t_audio)
plt.ylim(-1, 1)

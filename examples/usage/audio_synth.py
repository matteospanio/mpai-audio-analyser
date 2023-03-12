# Description: generate a sine wave and save it as a wav file
# create a sine wave with a frequency of 440 Hz, 44100 Hz sample rate, 16 bit depth and 1 channel

import numpy as np
from audiohandler import AudioWave

freq = 440
sr = 44100
bit = 16
channels = 1
time = np.arange(0, 1, 1 / sr)

signal = np.sin(2 * np.pi * freq * time)

signal *= 2**(bit - 1)
signal = np.int16(signal)

audio = AudioWave(signal, bit, channels, sr)
audio.save("signal.wav")

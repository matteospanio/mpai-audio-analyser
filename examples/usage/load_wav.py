# Description: Load a wav file and get the signal

from audiohandler import AudioWave

audio_file = AudioWave.from_file("signal.wav")

sample_rate = audio_file.samplerate
channels = audio_file.channels
bit = audio_file.bit
signal = audio_file.array

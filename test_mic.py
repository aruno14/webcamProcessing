import time
import numpy as np
import pyaudio
import struct

CHUNK = 44100
FORMAT = pyaudio.paFloat32
format_max = 32767
CHANNELS, RATE = 1, 44100
decoded_data = []

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("* recording")

start_time = time.time()
while True:
    data = stream.read(CHUNK)
    end_time = time.time()
    elapsed_ms = (end_time - start_time) * 1000
    print("elapsed_ms", elapsed_ms)
    start_time = end_time

    count = len(data)/2
    format = "%dh"%(count)
    shorts = struct.unpack(format, data)
    newData = np.asarray(shorts)/32767#Normalized between -1~1
    meanNoise = np.mean(np.square(newData))
    print("Sound:", meanNoise)
    decoded_data = np.concatenate([decoded_data, newData], axis=-1)

print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()

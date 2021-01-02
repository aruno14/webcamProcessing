import time
import numpy as np
import tensorflow as tf
import pyaudio
import struct

labelsGender = ['male', 'female']
model_gender = "models/model_gender_voice.tflite"
model_age = "models/model_age_voice.tflite"

interpreter_gender = tf.lite.Interpreter(model_path=model_gender)
interpreter_gender.allocate_tensors()

interpreter_age = tf.lite.Interpreter(model_path=model_age)
interpreter_age.allocate_tensors()

frame_length = 1024
image_width = 128
def audioToTensor(audioArray, audioSR:int):
    audio = tf.convert_to_tensor(audioArray, dtype=tf.float32)
    frame_step = int(audioSR * 0.008)
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step)
    spect_real = tf.math.real(spectrogram)
    spect_real = tf.abs(spect_real)
    partsCount = len(spect_real)//image_width
    parts = np.zeros((partsCount, image_width, int(frame_length/2+1)))
    for i, p in enumerate(range(0, len(spectrogram)-image_width, image_width)):
        parts[i] = spect_real[p:p+image_width]
    return parts, audioSR

def toAge(predictions, step=5, start=2.5):
    return np.sum(predictions * np.arange(start, len(predictions)*step+start, step, dtype="float32"))

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    output = classify_image_simple(interpreter, image, top_k)
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def classify_image_simple(interpreter, image, top_k=1):
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))
    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)
    return output

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

    if len(decoded_data) > image_width * RATE * 0.008 + frame_length:
        start_time = time.time()
        decoded_data = decoded_data[-int((image_width) * RATE * 0.008 + frame_length):]
        test_audio, _ = audioToTensor(decoded_data, RATE)

        gender = classify_image(interpreter_gender, test_audio, top_k=1)
        print("audio gender:", gender[0][1], labelsGender[gender[0][0]])

        results_age = classify_image_simple(interpreter_age, test_audio)
        age = toAge(results_age, start=15, step=10)
        print("audio age:", results_age, age)

        elapsed_ms = (time.time() - start_time) * 1000
        print("Aelapsed_ms", elapsed_ms)
        decoded_data = []

print("* done recording")
stream.stop_stream()
stream.close()
p.terminate()

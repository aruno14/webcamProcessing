import time
import numpy as np
import tensorflow as tf
import pyaudio
import struct
import threading
import cv2

labelsGender = ['male', 'female']
model_gender_voice = "models/model_gender_voice.tflite"
model_age_voice = "models/model_age_voice.tflite"

model_gender_face ="models/model_gender_grayscale.tflite"
model_age_face ="models/model_age.tflite"
model_emotion_face ="models/model_emotion.tflite"
labelsEmotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

interpreter_gender_voice = tf.lite.Interpreter(model_path=model_gender_voice)
interpreter_gender_voice.allocate_tensors()

interpreter_age_voice = tf.lite.Interpreter(model_path=model_age_voice)
interpreter_age_voice.allocate_tensors()

interpreter_gender_face = tf.lite.Interpreter(model_path=model_gender_face)
interpreter_gender_face.allocate_tensors()

interpreter_age_face = tf.lite.Interpreter(model_path=model_age_face)
interpreter_age_face.allocate_tensors()

interpreter_emotion_face = tf.lite.Interpreter(model_path=model_emotion_face)
interpreter_emotion_face.allocate_tensors()


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


audio_data = []
stopAudio = False
CHANNELS, RATE = 1, 44100
def audioCapture():
    global audio_data, stopAudio
    CHUNK = 44100
    FORMAT = pyaudio.paFloat32
    format_max = 32767

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("* recording")

    start_time = time.time()
    while not stopAudio:
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
        audio_data = np.concatenate([audio_data, newData], axis=-1)

    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

current_gender_voice = [0, 0]
current_age_voice = -1
def audioAnalysis():
    global audio_data, stopAudio, current_age_voice, current_gender_voice
    while not stopAudio:
        if len(audio_data) > image_width * RATE * 0.008 + frame_length:
            start_time = time.time()
            audio_data = audio_data[-int((image_width) * RATE * 0.008 + frame_length):]
            test_audio, _ = audioToTensor(audio_data, RATE)

            gender = classify_image(interpreter_gender_voice, test_audio, top_k=1)
            current_gender_voice = gender[0]
            print("audio gender:", current_gender_voice[1], labelsGender[current_gender_voice[0]])

            results_age = classify_image_simple(interpreter_age_voice, test_audio)
            current_age_voice = toAge(results_age, start=15, step=10)
            print("audio age:", results_age, current_age_voice)

            elapsed_ms = (time.time() - start_time) * 1000
            print("audio elapsed_ms", elapsed_ms)
            audio_data = []

faceSize = 150
faceMean = np.zeros((faceSize, faceSize, 3), np.uint8)
current_gender_face = [0, 0]
current_emotion_face = [0, 0]
current_age_face = -1
stopFace = False
def faceCapture():
    global faceMean, current_gender_face, current_emotion_face, current_age_faces, faceSize, stopFace, stopAudio
    video = cv2.VideoCapture(0)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(fps, width, height)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    textColor = (255, 255, 255)
    frameCount = 0
    faceMeanCount = 0

    while not stopFace:
        grabbed, frame = video.read()
        frameCount+=1
        outputFrame = np.zeros((height, width + faceSize, 3), np.uint8)
        frame = cv2.resize(frame, (width, height))

        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
        for (x,y,w,h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (faceSize, faceSize))
            faceMean = np.average([faceMean, face], axis=0, weights=[min(faceMeanCount, 100), 1])
            faceMeanCount+=1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)

        outputFrame[0:height,0:width] = frame
        outputFrame[0:faceSize, width:width+faceSize] = faceMean

        cv2.putText(outputFrame, 'Face', (width, faceSize + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
        cv2.putText(outputFrame, '{}: {:0.4f}'.format(labelsGender[current_gender_face[0]], current_gender_face[1]), (width, faceSize + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
        cv2.putText(outputFrame, '{}: {:0.4f}'.format(labelsEmotion[current_emotion_face[0]], current_emotion_face[1]), (width, faceSize + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
        cv2.putText(outputFrame, '{}: {:0.4f}'.format('age', current_age_face), (width, faceSize + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)

        cv2.putText(outputFrame, 'Voice', (width, faceSize + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
        cv2.putText(outputFrame, '{}: {:0.4f}'.format(labelsGender[current_gender_voice[0]], current_gender_voice[1]), (width, faceSize + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
        cv2.putText(outputFrame, '{}: {:0.4f}'.format('age', current_age_voice), (width, faceSize + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)

        cv2.imshow("Video", outputFrame)
        if frameCount%100 == 0:
            cv2.imwrite("video.jpg", outputFrame)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            stopFace = True
            stopAudio = True
            break

def faceAnalysis():
    global faceMean, current_gender_face, current_emotion_face, current_age_face, stopFace
    while not stopFace:
        start_time = time.time()
        faceMean48 = cv2.resize(faceMean, (48, 48))
        faceMeanGray = cv2.cvtColor(faceMean48.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        results_gender = classify_image(interpreter_gender_face, np.expand_dims(faceMeanGray, -1), top_k=1)
        current_gender_face = results_gender[0]
        print('{}: {:0.4f}'.format(labelsGender[current_gender_face[0]], current_gender_face[1]))

        results_emotion = classify_image(interpreter_emotion_face, np.expand_dims(faceMeanGray, -1), top_k=1)
        current_emotion_face = results_emotion[0]
        print('{}: {:0.4f}'.format(labelsEmotion[current_emotion_face[0]], current_emotion_face[1]))

        results_age = classify_image_simple(interpreter_age_face, np.expand_dims(faceMeanGray, -1))
        current_age_face = toAge(results_age)
        print('{}: {:0.4f}'.format('age', current_age_face))
        elapsed_ms = (time.time() - start_time) * 1000
        print("face elapsed_ms", elapsed_ms)

audioCaptureThread = threading.Thread(target=audioCapture, args=())
audioCaptureThread.start()
videoCaptureThread = threading.Thread(target=faceCapture, args=())
videoCaptureThread.start()

audioAnalysisThread = threading.Thread(target=audioAnalysis, args=())
audioAnalysisThread.start()
videoAnalysisThread = threading.Thread(target=faceAnalysis, args=())
videoAnalysisThread.start()

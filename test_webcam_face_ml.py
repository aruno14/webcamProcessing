import cv2
import time
import numpy as np
import tensorflow as tf

gender_model="models/model_gender_grayscale.tflite"
labels_gender = ['male', 'female']
emotion_model="models/model_emotion.tflite"
labels_emotion = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
age_model="models/model_age.tflite"

interpreter_gender = tf.lite.Interpreter(model_path=gender_model)
interpreter_gender.allocate_tensors()
interpreter_emotion = tf.lite.Interpreter(model_path=emotion_model)
interpreter_emotion.allocate_tensors()
interpreter_age = tf.lite.Interpreter(model_path=age_model)
interpreter_age.allocate_tensors()

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
  
video = cv2.VideoCapture(0)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))#//2
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))#//2
print(fps, width, height)

faceSize = 150
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
textColor = (255, 255, 255)
frameCount = 0
faceMean = np.zeros((faceSize, faceSize, 3), np.uint8)
faceMeanCount = 0

current_gender = [0, 0]
current_emotion = [0, 0]
current_age = -1

while True:
    grabbed, frame = video.read()
    frameCount+=1
    outputFrame = np.zeros((height, width + faceSize, 3), np.uint8)
    frame = cv2.resize(frame, (width, height))
    
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
    for (x,y,w,h) in faces:
        face = cv2.resize(frame[y:y+h, x:x+w], (faceSize, faceSize))
        faceMean = np.average([faceMean, face], axis=0, weights=[faceMeanCount, 1])
        faceMeanCount+=1
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
    
    outputFrame[0:height,0:width] = frame
    outputFrame[0:faceSize, width:width+faceSize] = faceMean
    
    if frameCount%100 == 0:
        print("====Frame", frameCount, "====")
        start_time = time.time()
        faceMean48 = cv2.resize(faceMean, (48, 48))
        faceMeanGray = cv2.cvtColor(faceMean48.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        
        results_gender = classify_image(interpreter_gender, np.expand_dims(faceMeanGray, -1), top_k=1)
        current_gender = results_gender[0]
        print('{}: {:0.4f}'.format(labels_gender[current_gender[0]], current_gender[1]))
     
        results_emotion = classify_image(interpreter_emotion, np.expand_dims(faceMeanGray, -1), top_k=1)
        current_emotion = results_emotion[0]
        print('{}: {:0.4f}'.format(labels_emotion[current_emotion[0]], current_emotion[1]))
        
        results_age = classify_image_simple(interpreter_age, np.expand_dims(faceMeanGray, -1))
        current_age = toAge(results_age)
        print('{}: {:0.4f}'.format('age', current_age))
        elapsed_ms = (time.time() - start_time) * 1000
        print(frameCount, "elapsed_ms", elapsed_ms)
        
        cv2.imwrite("video.jpg", outputFrame)
    
    cv2.putText(outputFrame, '{}: {:0.4f}'.format(labels_gender[current_gender[0]], current_gender[1]), (width, faceSize + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
    cv2.putText(outputFrame, '{}: {:0.4f}'.format(labels_emotion[current_emotion[0]], current_emotion[1]), (width, faceSize + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
    cv2.putText(outputFrame, '{}: {:0.4f}'.format('age', current_age), (width, faceSize + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
    cv2.imshow("Video", outputFrame)
    
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

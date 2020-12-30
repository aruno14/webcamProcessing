import cv2
import time
import numpy as np

video = cv2.VideoCapture(0)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))//2
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
print(fps, width, height)

faceSize = 150
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

frameCount = 0
faceMean = np.zeros((faceSize, faceSize, 3), np.uint8)
faceMeanCount = 0

while True:
    grabbed, frame = video.read()
    frameCount+=1
    print("====Frame", frameCount, "====")
    outputFrame = np.zeros((height, width + faceSize, 3), np.uint8)
    frame = cv2.resize(frame, (width, height))
    start_time = time.time()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
            face = cv2.resize(frame[y:y+h, x:x+w], (faceSize, faceSize))
            faceMean = np.average([faceMean, face], axis=0, weights=[faceMeanCount, 1])
            faceMeanCount+=1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
    elapsed_ms = (time.time() - start_time) * 1000
    print(frameCount, "elapsed_ms", elapsed_ms)
    
    outputFrame[0:height,0:width] = frame
    outputFrame[0:faceSize, width:width+faceSize] = faceMean
    
    cv2.imshow("Video", outputFrame)
    if frameCount%100 == 0:
        cv2.imwrite("video.jpg", outputFrame)
    
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

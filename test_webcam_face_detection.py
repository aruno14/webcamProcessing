import cv2
import time
import face_detection

print(face_detection.available_detectors)
detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.5, nms_iou_threshold=.3)

video = cv2.VideoCapture(0)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))//2
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))//2
print(fps, width, height)

frameCount = 0
while True:
    grabbed, frame = video.read()
    frameCount+=1
    print("====Frame ", frameCount, "====")
    frame = cv2.resize(frame, (width, height))
    
    start_time = time.time()
    detections = detector.detect(frame)
    print("detections", detections)
    for d in detections:
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (255, 0, 0), 2)
    elapsed_ms = (time.time() - start_time) * 1000
    print(frameCount, "elapsed_ms", elapsed_ms)
    
    cv2.imshow("Video", frame)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

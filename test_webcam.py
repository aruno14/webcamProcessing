import cv2

video = cv2.VideoCapture(0)
fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)

frameCount = 0
while True:
    grabbed, frame = video.read()
    frameCount+=1
    print("====Frame ", frameCount, "====")
    frame = cv2.resize(frame, (width//2, height//2))
    cv2.imshow("Video", frame)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

import cv2
from imutils.video import VideoStream,FPS
from yolo import predict

vs = VideoStream(src="http://192.168.1.50:8000/stream.mjpg", usePiCamera=False).start()


while True:
  image = vs.read()
  if (image is not None):
    detected_labels = predict(image)
    print(detected_labels)

    cv2.imshow('frame', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# release the file pointers
print("[INFO] cleaning up...")
vs.release()
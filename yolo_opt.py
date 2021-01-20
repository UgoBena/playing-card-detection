import numpy as np
import time
import cv2
from imutils.video import VideoStream,FPS

USE_PICAMERA = True
INPUT_FILE = "poker.mp4"
LABELS_FILE='cards.names'
CONFIG_FILE='yolo-tiny-cards.cfg'
WEIGHTS_FILE='yolo-tiny-cards_best.weights'
CONFIDENCE_THRESHOLD=0.75
NMS_THRESHOLD = 0.4

H=None
W=None

fps = FPS().start()

LABELS = open(LABELS_FILE).read().strip().split("\n")

np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
  dtype=int)

model = cv2.dnn_DetectionModel(WEIGHTS_FILE,CONFIG_FILE)
model.setInputParams(size=(416,416),scale=1/255,swapRB=True, crop=False)

# test_area=np.array([[230,190],[300,180],[300,210],[240,220]])
# test_area = test_area.reshape((-1, 1, 2)) 
vs = VideoStream(src="192.168.1.30:8000/stream.mjpg").start()


image = vs.read()
while image is not None:
  try:
    image = vs.read()
  except:
    break
  #cv2.polylines(image, [test_area],isClosed=True,color=[0,0,255],thickness=2)
  # initialize our lists of detected bounding boxes, confidences, and
  # class IDs, respectively
  classes,scores,boxes = model.detect(image,CONFIDENCE_THRESHOLD,NMS_THRESHOLD)
  
  for (classid_array,score_array,box) in zip(classes,scores,boxes):
      (x, y , w, h) = (box[0], box[1],box[2],box[3])

      classid = classid_array[0]
      score = score_array[0]
      color = [int(c) for c in COLORS[classid]]
      cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
      text = "{}: {:.4f}".format(LABELS[classid], score)
      cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 2)



  # show the output image
  cv2.imshow("output", cv2.resize(image,(800, 600)))
  fps.update()
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break

fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()

# release the file pointers
print("[INFO] cleaning up...")
vs.stop()
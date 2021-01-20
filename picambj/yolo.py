import numpy as np
import time
import cv2

LABELS_FILE='cards.names'
CONFIG_FILE='yolo-tiny-cards.cfg'
WEIGHTS_FILE='yolo-tiny-cards_best.weights'
CONFIDENCE_THRESHOLD=0.75
NMS_THRESHOLD = 0.4


LABELS = open(LABELS_FILE).read().strip().split("\n")


np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype=int)

model = cv2.dnn_DetectionModel(WEIGHTS_FILE,CONFIG_FILE)
model.setInputParams(size=(416,416),scale=1/255,swapRB=True, crop=False)

def predict(image):
    classes,scores,boxes = model.detect(image,CONFIDENCE_THRESHOLD,NMS_THRESHOLD)
    unique_classes = np.unique(classes)
    unique_labels = []
    for unique_class in unique_classes:
        unique_labels.append(LABELS[unique_class])
    return unique_labels
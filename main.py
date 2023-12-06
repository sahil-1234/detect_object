import cv2
from object_detect import perform_object_detection

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Can't open the video")

perform_object_detection(cap)

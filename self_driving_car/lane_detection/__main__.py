import cv2
import sys
from .lane_detection import LaneDetection


if (len(sys.argv) < 2):
    print("usage: python test.py <filename>")
    exit()

lane_detection = LaneDetection()

image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

def apply(x):
    img, lines, thetas = lane_detection.get_lanes(image)

    cv2.imshow("EdgeDetection", img)
    

cv2.namedWindow("EdgeDetection")
cv2.createTrackbar("Min", "EdgeDetection", 162, 500, apply)
cv2.createTrackbar("Max", "EdgeDetection", 81, 500, apply)
cv2.createTrackbar("Kernel", "EdgeDetection", 7, 100, apply)

cv2.waitKey(0)
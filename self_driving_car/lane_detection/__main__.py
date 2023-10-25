import cv2
import sys
from .lane_detection import LaneDetection

if (len(sys.argv) < 2):
    print("usage: python test.py <filename>")
    exit()

lane_detection = LaneDetection()

image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

def apply(x):
    min_threshold = cv2.getTrackbarPos("Min", "EdgeDetection")
    max_threshold = cv2.getTrackbarPos("Max", "EdgeDetection")
    max_lane_gap = cv2.getTrackbarPos("Max L Gap", "EdgeDetection")
    threshold = cv2.getTrackbarPos("Threshold", "EdgeDetection")
    rho = cv2.getTrackbarPos("Rho", "EdgeDetection")

    img, lines, thetas = lane_detection.get_lanes(image, min_threshold, max_threshold, max_lane_gap, threshold, rho)

    cv2.imshow("EdgeDetection", img)

cv2.namedWindow("EdgeDetection")
cv2.createTrackbar("Min", "EdgeDetection", 111, 500, apply)
cv2.createTrackbar("Max", "EdgeDetection", 71, 500, apply)
cv2.createTrackbar("Max L Gap", "EdgeDetection", 50, 100, apply)
cv2.createTrackbar("Threshold", "EdgeDetection", 180, 250, apply)
cv2.createTrackbar("Rho", "EdgeDetection", 1, 100, apply)

# Chame a função apply uma vez para exibir a imagem inicial
apply(0)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Tecla Esc para sair
        break

cv2.destroyAllWindows()

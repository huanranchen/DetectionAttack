import sys
import cv2

img = cv2.imread(sys.argv[1])
img = cv2.resize(img, (416, 416))
cv2.imwrite('./test.png', img)
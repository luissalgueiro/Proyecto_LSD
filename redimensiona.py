import cv2 
import numpy as np
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required = True, help="path to input image")

args= vars(ap.parse_args())

image = cv2.imread(args["image"])

resized = cv2.resize(image, (300,300), interpolation = cv2.INTER_CUBIC)


cv2.imwrite("resized_"+args["image"],resized)
cv2.imshow("resized",resized)
cv2.waitKey()
cv2.destroyAllWindows()

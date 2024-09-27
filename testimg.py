import cv2
from PIL import Image
from IPython.display import display
import numpy as np

img_path = "pexels-samuel-reis-355265419-14262633.jpg"

# Read the image
img = cv2.imread(img_path)

# todo: resize the image
img = cv2.resize(img, (0,0),None,0.1,0.1)

def getContours(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (11,11), 0)
    
    # Apply canny edge detection
    edges = cv2.Canny(blurred, 100, 100)
    
    return gray, blurred, edges

gray, blurred, edges = getContours(img)

#show the image
cv2.imshow("origin",img)
cv2.imshow("gray",gray)

cv2.imshow("blurred", blurred)

cv2.imshow("edges", edges)

cv2.waitKey(0)

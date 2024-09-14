import cv2
import numpy as np



def get_limits(color):
    '''Returns the upper and lower limits for a given color in HSV'''
    c = np.uint8([[color]])  # BGR values
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hue = hsvC[0][0][0]  # Get the hue value

    if hue >= 165:  # Upper limit for divided yellow hue
        lowerLimit = np.array([hue - 7, 150, 150])
        upperLimit = np.array([180, 255, 255])
    elif hue <= 15:  # Lower limit for divided yellow hue
        lowerLimit = np.array([0, 150, 150])
        upperLimit = np.array([hue + 7, 255, 255])
    else:
        lowerLimit = np.array([hue - 7, 150, 150])
        upperLimit = np.array([hue + 7, 255, 255])

    return lowerLimit, upperLimit


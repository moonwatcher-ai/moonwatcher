import cv2
import numpy as np


def compute_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])


def compute_contrast(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale.std()


def compute_saturation(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 1])


def compute_resolution(image):
    height, width = image.shape[:2]
    return height * width


ATTRIBUTE_FUNCTIONS = {
    "brightness": compute_brightness,
    "contrast": compute_contrast,
    "saturation": compute_saturation,
    "resolution": compute_resolution,
}

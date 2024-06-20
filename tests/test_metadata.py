import cv2
import numpy as np

from moonwatcher.dataset.metadata import (
    compute_brightness,
    compute_contrast,
    compute_saturation,
    compute_resolution,
    ATTRIBUTE_FUNCTIONS,
)


def test_compute_brightness():
    white_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert (
        compute_brightness(white_image) == 255
    ), "Brightness of a white image should be maximum"


def test_compute_contrast():
    gray_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert (
        compute_contrast(gray_image) == 0
    ), "Contrast of a uniform image should be zero"


def test_compute_saturation():
    bgr_image = np.zeros((100, 100, 3), dtype=np.uint8)
    bgr_image[:, :, 2] = 255
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    computed_saturation = compute_saturation(hsv_image)
    assert (
        computed_saturation == 255
    ), f"Saturation of a fully red image should be maximum, got {computed_saturation}"


def test_compute_resolution():
    test_image = np.zeros((200, 300, 3), dtype=np.uint8)
    assert (
        compute_resolution(test_image) == 60000
    ), "Resolution should be width multiplied by height"


def test_attribute_functions():
    image = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert ATTRIBUTE_FUNCTIONS["brightness"](image) == compute_brightness(image)
    assert ATTRIBUTE_FUNCTIONS["contrast"](image) == compute_contrast(image)
    assert ATTRIBUTE_FUNCTIONS["saturation"](image) == compute_saturation(image)
    assert ATTRIBUTE_FUNCTIONS["resolution"](image) == compute_resolution(image)

import numpy as np

import mmcv, cv2

def get_skin_image(image):
    lower = np.array([0, 23, 88])
    upper = np.array([46, 132, 255])

    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skinImg = cv2.bitwise_and(image, image, mask=skinMask)

    return skinImg, skinMask
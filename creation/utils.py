import cv2
from PIL import Image
import numpy as np


def complexity(img: np.ndarray) -> float:
    """
    Calculates the complexity of an image as defined by:
    :param img: numpy array containing the image data in opencv format.
    :return: the complexity of the image.
    """
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    M = lab_img.shape[0]
    N = lab_img.shape[1]

    ksize = -1
    gX = cv2.Sobel(lab_img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=ksize)
    gY = cv2.Sobel(lab_img, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=ksize)
    gX = cv2.convertScaleAbs(gX)
    gY = cv2.convertScaleAbs(gY)

    combined = abs(cv2.addWeighted(gX, 0.5, gY, 0.5, 0))
    X, Y, Z = combined.shape
    reshaped = combined.reshape((X * Y, Z))

    total_sum = np.sum(np.max(reshaped, axis=1))
    complexity = (1 / (N * M)) * total_sum

    return complexity


def PIL_to_opencv(image_path: str) -> np.ndarray:
    """
    Opens an image using PIL and converts it to an opencv numpy array.
    :param image_path: path of the image.
    :return: numpy array containing the image data in opencv format.
    """
    pil_image = Image.open(image_path).convert('RGB')
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def average_hue(img: np.ndarray) -> float:
    """
    Calculates the average hue of the given image.
    :param img: numpy array containing the image data in opencv format.
    :return: The average hue of the image.
    """
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    avg_colour = np.mean(hsv_img[:, :, 0])
    return avg_colour


def motion_blur(img: np.ndarray) -> float:
    """
    Calculates the motion blur of the given image.
    :param img: numpy array containing the image data in opencv format.
    :return: The motion blue of the image.
    """
    return cv2.Laplacian(img, cv2.CV_64F).var()

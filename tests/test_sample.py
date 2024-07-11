from PIL import Image
import cv2 
import numpy as np


def func(x):
    return x + 1


def test_answer():
    assert func(4) == 5


def test_random():
    img1 = Image.open("tests/img1.jpg")
    img2 = Image.open("tests/img2.jpg")
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)

    assert gray1.shape[0] == 480

     

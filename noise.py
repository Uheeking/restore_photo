import cv2
import numpy as np
from matplotlib import pyplot as plt

def main(src):
    img = cv2.imread(src)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.show()
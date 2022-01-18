import cv2
import numpy as np
from matplotlib import pyplot as plt

# 이미지 잡음제거(디노이징) - 컬러(정확도가 높음)
def main(src, name):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(dst)
    plt.savefig("/Users/gidaehyeon/Projects/venv/static/img_invert/"+name+'.jpg', facecolor='#bec5bd')
    
# 평균 블러링
def main2(src, name):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    k = np.array([[1,1,1],
                [1,1,1],
                [1,1,1]]) * (1/9)
    blur = cv2.filter2D(img, -1, k)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(blur)
    plt.savefig("/Users/gidaehyeon/Projects/venv/static/img_invert/"+name+'.jpg', facecolor='#bec5bd')
    
# 가우시안 블러링
def main3(src, name):
    img = cv2.imread(src)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    k = np.array([[1,2,1],
                [2,4,2],
                [1,2,1]]) * (1/16)
    blur = cv2.filter2D(img, -1, k)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(blur)
    plt.savefig("/Users/gidaehyeon/Projects/venv/static/img_invert/"+name+'.jpg', facecolor='#bec5bd')
    
# 선명도가 높음   
def main4(src, name):
    img = cv2.imread(src)
    blur = cv2.bilateralFilter(img, 5, 75, 75)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(blur)
    plt.savefig("/Users/gidaehyeon/Projects/venv/static/img_invert/"+name+'.jpg', facecolor='#bec5bd')
    
#소금 페퍼에 관련
def main5(src, name):
    img = cv2.imread(src)
    blur = cv2.medianBlur(img, 5)
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(blur)
    plt.savefig("/Users/gidaehyeon/Projects/venv/static/img_invert/"+name+'.jpg', facecolor='#bec5bd')
    
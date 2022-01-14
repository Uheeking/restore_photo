# 영상 노이즈 처리
## 노이즈

: 영상의 잡음(Noise)는 영상의 픽셀 값에 추가되는 원치 않는 형태의 신호를 의미합니다.

> f(x,y) = s(x,y) + n(x,y)
> 

여기서 f(x,y)는 획득된 영상, s(x,y)는 원본 신호, n(x,y)는 잡음을 의미합니다.

대부분의 경우 센서에서 잡음이 추가됩니다. 노이즈는 국지적이고 무작위적으로 생겨날수 있으며, 확률을 따르기에 노이즈가 생겨나는 방식이 여러가지 있습니다. 이를 노이즈 모델이라 합니다. 

## **잡음의 종류**

**(1) 가우시안 잡음(Gaussian noise)**

![https://blog.kakaocdn.net/dn/ctWW6j/btqJ4GDfCZZ/AxpyAxhAWDxSHFf6uO8tsk/img.png](https://blog.kakaocdn.net/dn/ctWW6j/btqJ4GDfCZZ/AxpyAxhAWDxSHFf6uO8tsk/img.png)

대부분의 잡음 형태는 가우시안 형태입니다. 이는 픽셀값에서 조금 더 어두워지거나 밝아지게 됩니다. 보통 평균이 0인 가우시안 분포를 따르는 잡음을 의미합니다. 

지지직 거리는 느낌의 이미지를 보면 거의 잡음이 가우시안 노이즈입니다. 이름이 가우시안 노이즈인 이유는, 이름처럼 가우스 함수에 따른 분포(가우시안 분포)를 따르고 있기 때문에 가우시안 노이즈라고 이름 붙여졌습니다. 

가우시안 잡음은 보통 이미지의 압축, 전송 등의 과정에서 일어납니다. 이미지가 압축되면서 이미지가 줄어들게 되고, 이후 다시 복구하는 과정에서 여러 가지 원인으로 인해 원래의 화소 값이 아닌, 오차가 생긴 값이 들어갈 수가 있습니다.

**(2) 소금$후추 잡음(Salt&Pepper)**

![https://blog.kakaocdn.net/dn/DnhtY/btqJ4HWuDAs/YFbI0rQrqfLpuTKKJLX6l1/img.png](https://blog.kakaocdn.net/dn/DnhtY/btqJ4HWuDAs/YFbI0rQrqfLpuTKKJLX6l1/img.png)

요즘 소금 후추 잡음은 거의 없습니다. 

일정 확률로 픽셀 값이 0 또는 255로 변하는 것을 말합니다.

즉, 원본에 대해 일정 값의 추가나 감소가 아니라, 일정 확률로 픽셀이 흰색, 검은색으로 존재하게 되는 것입니다. 0 또는 255값으로 픽셀이 변하는 것이기에, 마치 소금(흰색)과, 후추(검은색)를 뿌린 것과 같기에 소금 & 후추 노이즈라고 불립니다.

## 기본 잡음 추가/ 제거 함수

1) **가우시안 잡음 배열 구하기 함수**

```python
import cv2
import numpy as np

# std = 잡음의 크기, gray = 그레이 스케일 영상
def make_noise(std, gray):
	height, width = gray.shape
	img_noise = np.zeros((height, width), dtype=np.float)
	for i in range(height): 
		for a in range(width): 
			make_noise = np.random.normal() 
			set_noise = std * make_noise 
			img_noise[i][a] = gray[i][a] + set_noise
	return img_noise

```

→ 중첩 for문을 이용해서 가우시안 노이즈를 만들고, make_noise에 정규분포를 따르는(normal) 랜덤(random)한 숫자를 넣어줍니다. 그리고 set_noise에 make_noise와 외부에서 입력받은 std값을 곱하고, 조금 전에 만들었던 img_noise라는 빈 이미지에 원래 화소 + set_noise값을 하여 넣어줍니다.

이렇게 하면 노이즈가 추가된 이미지가 만들어집니다.

가우시안 잡음 제거 함수

```python
def run(): 
	# 이미지 불러오기, 그레이 스케일로 변환, 크기를 구해준다.
	img = cv2.imread('lenna.png') 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
	height, width = gray.shape 
	
	# 표준편차 설정
	std = 15 
	img_noise = make_noise(std, gray) 
	img_noise2 = make_noise(std, gray) 
	img_noise3 = make_noise(std, gray) 
	img_noise4 = make_noise(std, gray) 
	# 가우시안 노이즈가 첨가된 이미지를 네개를 만들어준다. 
	# 평균으로 노이즈를 제거하기 위함
	out2 = np.zeros((height, width), dtype=np.float) 
	out3 = np.zeros((height, width), dtype=np.float) 
	out4 = np.zeros((height, width), dtype=np.float) 

	# 평균 계산 
	for i in range(height): 
		for j in range(width): 
			if (img_noise[i][j] + img_noise2[i][j]) / 2 > 255: 
				out2[i][j] = 255 
			else: out2[i][j] = (img_noise[i][j] + img_noise2[i][j]) / 2 
			if (img_noise[i][j] + img_noise2[i][j] + img_noise3[i][j]) / 3 > 255: 
				out3[i][j] = 255 
			else: out3[i][j] = (img_noise[i][j] + img_noise2[i][j] + img_noise3[i][j]) / 3 
			if (img_noise[i][j] + img_noise2[i][j] + img_noise3[i][j] + img_noise4[i][j]) / 4 > 255: 
				out4[i][j] = 255 
			else: out4[i][j] = (img_noise[i][j] + img_noise2[i][j] + img_noise3[i][j] + img_noise4[i][j]) / 4 
	cv2.imshow("original", gray) 
	cv2.imshow('noise', img_noise.astype(np.uint8)) 
	cv2.imshow('avr2', out2.astype(np.uint8)) 
	cv2.imshow('avr3', out3.astype(np.uint8)) 
	cv2.imshow('avr4', out4.astype(np.uint8)) 
	cv2.waitKey(0) run() 
```

## **미디언 필터 - cv2.medianBlur**

미디언 필터(Median filter)는 주변 픽셀들의 값들을 정렬하여 그 중앙값(median)으로 픽셀 값을 대체합니다. 소금-후추 잡음 제거에 효과적 입니다. 

![https://blog.kakaocdn.net/dn/sbp1H/btqJ0oDpD6S/L6JIjEML0rDsne9zj9Q3Zk/img.png](https://blog.kakaocdn.net/dn/sbp1H/btqJ0oDpD6S/L6JIjEML0rDsne9zj9Q3Zk/img.png)

미디언 필터는 마스크 모양만 지정합니다. 사각형 행렬을 1열로 나열하고 정렬합니다. 중앙값에 있는 값을 이용해서 셋팅합니다. 입력값에 있는 값을 결과값으로 반환합니다.

OpenCV에서는 미디언 필터링 함수로 cv2.medianBlur 명령어를 제공하고 있습니다. 약간 블러링 되는 효과가 있으며 픽셀들이 뭉치는 형태를 띄어 보기 좋은 결과가 아닙니다.

> cv2.medianBlur(src, ksize, dst=None) -> dst
> 

src : 입력 영상. 각 채널 별로 처리됨

ksize : 커널 크기. 1보다 큰 홀수를 지정. 

dst : 출력 영상, src와 같은 크기, 같은 타입

### **미디언 필터링 예제**

```
src = cv2.imread('noise.bmp', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

dst = cv2.medianBlur(src, 3)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()
```

![https://blog.kakaocdn.net/dn/dxE7ud/btqJZnEK5mJ/SrWPqnCJ4uKFzMZwqFRp00/img.png](https://blog.kakaocdn.net/dn/dxE7ud/btqJZnEK5mJ/SrWPqnCJ4uKFzMZwqFRp00/img.png)

![https://blog.kakaocdn.net/dn/bAtr7H/btqJZnkoi9b/2oTtHss92c5A4r5S2xNEL1/img.png](https://blog.kakaocdn.net/dn/bAtr7H/btqJZnkoi9b/2oTtHss92c5A4r5S2xNEL1/img.png)

# **양방향 필터 - cv2.bilateralFIlter**

가우시안 잡음 제거에는 가우시안 필터가 효과적입니다. 가우시안 블러를 심하게 적용하면 영상에 있는 엣지 부분에 훼손이 생깁니다. 이 단점을 극복하기 위해 양방향 필터라는 기법이 생겼습니다. 이는 가우시안 필터를 양쪽 방향으로 두번 한다고해서 이름이 붙여졌습니다. 양방향 필터는 기준 픽셀과 이웃 픽셀과의 거리, 그리고 픽셀 값의 차이를 함께 고려하여 블러링 정도를 조절합니다.

## **양방향 필터의 작동 원리**

양방향 필터는 에지가 아닌 부분에서만 블러링을 합니다. 평탄한 부분은 가우시안 필터를 사용하고, 엣지 부분이면 가우시안의 일부분만 가져와 필터링을 합니다. 따라서 에지를 보존할 수 있습니다.

![https://blog.kakaocdn.net/dn/4kIeL/btqJ6caTggf/CUbuG9rqEdeFHHjRKK2ne1/img.png](https://blog.kakaocdn.net/dn/4kIeL/btqJ6caTggf/CUbuG9rqEdeFHHjRKK2ne1/img.png)

픽셀 값의 차이가 크면 0으로 채워넣고, 가우시안 필터의 고정점과 픽셀차이가 비슷하면 필터값을 가져옵니다.

OpenCV에서 제공하는 cv2.bilaterFIlter를 이용해서 양방향 필터링을 적용할 수 있습니다.

### **함수 설명**

> cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, dst=None, borderType=None) -> dst
> 
- src: 입력 영상. 8비트 또는 실수형, 1채널 또는 3채널.
- d: 필터링에 사용될 이웃 픽셀의 거리(지름), 음수(-1)를 입력하면 sigmaSpace 값에 의해 자동 결정(권장)
- sigmaColor: 색 공간에서 필터의 표준 편차(엣지냐, 아니냐, 만약 이를 100으로 주면 가우시안 필터 적용하는 것과 같음)
- sigmaSpace: 좌표 공간에서 필터의 표준 편차
- dst: 출력 영상. src와 같은 크기, 같은 타입.
- borderType: 가장자리 픽셀 처리 방식

### **양방향 필터링 예제**

```python
src = cv2.imread('lenna.bmp')

if src is None:
    print('Image load failed!')
    sys.exit()

dst = cv2.bilateralFilter(src, -1, 10, 5)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()
```

![https://blog.kakaocdn.net/dn/mLM3C/btqJ6dgz4EQ/e9XODTO316MzrZZWlit3Jk/img.png](https://blog.kakaocdn.net/dn/mLM3C/btqJ6dgz4EQ/e9XODTO316MzrZZWlit3Jk/img.png)

![https://blog.kakaocdn.net/dn/0nIm4/btqJ1Mqs8qS/GECvg0PHI2oxOULnoSPk4K/img.png](https://blog.kakaocdn.net/dn/0nIm4/btqJ1Mqs8qS/GECvg0PHI2oxOULnoSPk4K/img.png)

import cv2
import numpy
import os


baseDirectory = os.path.abspath(os.path.dirname(__file__)) # 현재 해당 파일 위치

imageDirectory = f'{baseDirectory}/Image/ESTP/m' # 이미지 폴더 위치

haarcascade = cv2.CascadeClassifier(f'{baseDirectory}/haarcascade_frontalface_alt2.xml') # 검출 대상 : 정면 얼굴 검출 

savePath = f'{baseDirectory}/Cropped'
os.makedirs(savePath, exist_ok = True)

filenameList = os.listdir(imageDirectory)

for filename in filenameList:

    image = numpy.fromfile(f'{imageDirectory}/{filename}', numpy.uint8) # 한글경로 읽기
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

    cv2.imshow('Original', image)

    # scaleFactor : 이미지 피라미드 스케일, minNeighbors : 인접 객체 최소 거리 픽셀, minSize : 탐지 객체 최소 크기
    boundingBoxes = haarcascade.detectMultiScale(image, scaleFactor =  1.1, minNeighbors = 3, minSize = (20,20))

    for boundingBox in boundingBoxes:
            
        x, y, width, height = boundingBox

        croppedImage = image[y : y + height, x : x + width,  : ]
        # 얼굴 인식 후 크로핑 실행 

        cv2.imshow(f'Cropped', croppedImage)

        cv2.imwrite(f'{savePath}/{filename}', croppedImage)






        
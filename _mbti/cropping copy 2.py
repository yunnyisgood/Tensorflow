import cv2
import numpy
import os


# 한글경로 쓰기

baseDirectory = os.path.abspath(os.path.dirname(__file__)) # d:\Tensorflow\_mbti\cropping.py

imageDirectory = f'{baseDirectory}/Image/INFP/w' # 계속 다르게 하기 

# face_cascade = cv2.CascadeClassifier(f'{baseDirectory}/haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier(f'{baseDirectory}/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(f'{baseDirectory}/haarcascade_eye.xml')

savePath = f'{baseDirectory}/Cropped'
os.makedirs(savePath, exist_ok = True)

filenameList = os.listdir(imageDirectory)

for filename in filenameList:
    
    image = cv2.imread(f'{imageDirectory}/{filename}')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(image, scaleFactor =  1.1, minNeighbors = 1, minSize = (150, 150))
    imgNum  = 0

    for (x,y,w,h) in faces:
        cropped = image[y - int(h / 1):y + h + int(h / 1), x - int(w / 1):x + w + int(w / 1)]
        # 이미지를 저장
        cv2.imwrite(f'{savePath}/INFP/w/{filename}', cropped)
        imgNum += 1

        cv2.imshow('Cropped', cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


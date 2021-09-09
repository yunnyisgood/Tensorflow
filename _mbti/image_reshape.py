from PIL import Image
import os.path
import os

targerdir = r"D:\Tensorflow\_mbti\Cropped\ESFP\m"  # 크롭된 이미지 파일 위치 

newpath = r"D:\mbti\ESFP\m"  #저장팔 폴더 위치 

files = os.listdir(targerdir)

format = [".jpg",".png",".jpeg","bmp",".JPG",".PNG","JPEG","BMP"] #지원하는 파일 형태의 확장자들

for (path,dirs,files) in os.walk(targerdir):
    for file in files:
         if file.endswith(tuple(format)):  
             # endswith는 특정 문자로 끝이나는 문자열을 찾는 메소드 -> 확장자로 조건문 설정
             
             image = Image.open(path+"\\"+file)

             image=image.resize((150, 150))
             
             image.save(newpath+"\\"+file)
         else:
             print(path)
             print("InValid",file)










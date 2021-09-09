import cv2
import numpy
import os

# 한글경로 쓰기
def imwrite(image, savePath, parameters = None):
    
    try: 
        
        result, array = cv2.imencode(os.path.splitext(savePath)[1], image, parameters) 
        
        if result: 
            
            with open(savePath, mode = 'w+b') as f:
                
                array.tofile(f)

                return True 
                
        else: 
            
            return False 
            
    except Exception as exception: 
        
        print(exception) 
        
        return False

if __name__ == '__main__':
    
    baseDirectory = os.path.abspath(os.path.dirname(__file__))

    imageDirectory = f'{baseDirectory}/Image'

    haarcascade = cv2.CascadeClassifier(f'{baseDirectory}/haarcascade_frontalface_alt.xml')

    savePath = f'{baseDirectory}/GrabCut'
    os.makedirs(savePath, exist_ok = True)

    filenameList = os.listdir(imageDirectory)

    for filename in filenameList:

        image = numpy.fromfile(f'{imageDirectory}/{filename}', numpy.uint8) # 한글경로 읽기
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

        cv2.imshow('Original', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        boundingBoxes = haarcascade.detectMultiScale(image, scaleFactor =  1.5, minNeighbors = 5, minSize = (20,20))

        for boundingBox in boundingBoxes:
                    
            x, y, width, height = boundingBox

            mask = numpy.zeros(image.shape[ : 2], dtype = numpy.uint8)

            bgdModel = numpy.zeros((1, 65), dtype = numpy.float64)
            fgdModel = numpy.zeros((1, 65), dtype = numpy.float64)

            cv2.grabCut(img = image, mask = mask, rect = boundingBox, bgdModel = bgdModel, fgdModel = fgdModel, iterCount = 1, mode = cv2.GC_INIT_WITH_RECT)

            mask = numpy.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')

            # image = image * mask[ : ,  : , numpy.newaxis]
            image[mask == 0] = 255
            croppedImage = image[y : y + height, x : x + width,  : ]

            cv2.imshow('GrabCut', croppedImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            imwrite(image = croppedImage, savePath = f'{savePath}/{filename}')
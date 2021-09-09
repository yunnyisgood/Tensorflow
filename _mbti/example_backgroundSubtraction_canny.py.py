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

    savePath = f'{baseDirectory}/Canny'
    os.makedirs(savePath, exist_ok = True)

    filenameList = os.listdir(imageDirectory)

    for filename in filenameList:

        image = numpy.fromfile(f'{imageDirectory}/{filename}', numpy.uint8) # 한글경로 읽기
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        
        cv2.imshow('Original', image)
        cv2.waitKey(0)

        grayScaledImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale
       
        # edges = cv2.Canny(image = grayScaledImage, threshold1 = 18, threshold2 = 28) # Canny edge detection
        edges = cv2.Canny(image = grayScaledImage, threshold1 = 70, threshold2 = 150) # Canny edge detection
        
        cv2.imshow('Canny', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        edges = cv2.dilate(edges, None)
        
        cv2.imshow('Dilate', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        edges = cv2.erode(edges, None)

        cv2.imshow('Erode', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contours, _ = cv2.findContours(image = edges, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_NONE)
    
        contourInformation = []
        
        for contour in contours:

            contourInformation.append((contour, cv2.isContourConvex(contour), cv2.contourArea(contour)))

        contourInformation = sorted(contourInformation, key = lambda x : x[2], reverse = True)

        mask = numpy.zeros(edges.shape)
        cv2.fillConvexPoly(mask, contourInformation[0][0], (255))

        # mask = cv2.dilate(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 10)
        '''
        cv2.imshow('Maks dilate', mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        image = image.astype('float32') / 255.0
        mask  = numpy.dstack([mask] * 3).astype('float32') / 255.0
        
        maskedImage = (mask * image) + ((1 - mask) * (1.0, 1.0, 1.0))
        maskedImage = (maskedImage * 255).astype('uint8')

        cv2.imshow('Maksed', maskedImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        imwrite(image = maskedImage, savePath = f'{savePath}/{filename}')
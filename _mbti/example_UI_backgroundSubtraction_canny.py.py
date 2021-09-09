import cv2
import numpy
import sys

def onChange(x):

    pass

def main(filePath):
    
    image = numpy.fromfile(filePath, numpy.uint8) # 한글경로 읽기
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

    grayScaledImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Grayscale
    
    cv2.namedWindow('Background subtraction')
    
    cv2.createTrackbar('High threshold', 'Background subtraction', 0, 255, onChange)
    cv2.createTrackbar('Low threshold', 'Background subtraction', 0, 255, onChange)
    cv2.imshow('Background subtraction', image)
    
    while True:

        # k = cv2.waitKey(0) & 0xFF
        k = cv2.waitKey(1) & 0xFF
        
        if k == 27:

            break
            
        threshold_low = cv2.getTrackbarPos('Low threshold', 'Background subtraction')
        threshold_high = cv2.getTrackbarPos('High threshold', 'Background subtraction')
        
        if threshold_low > threshold_high:

            print('Low threshold must be low than High threshold')
        
        elif ((threshold_low == 0) and (threshold_high == 0)):

            cv2.imshow('Background subtraction', image)
        
        else:

            edges = cv2.Canny(grayScaledImage, threshold_low, threshold_high)
            edges = cv2.dilate(edges, None)
            edges = cv2.erode(edges, None)
            
            contours, _ = cv2.findContours(image = edges, mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_NONE)

            contourInformation = []
            
            for contour in contours:

                contourInformation.append((contour, cv2.isContourConvex(contour), cv2.contourArea(contour)))

            contourInformation = sorted(contourInformation, key = lambda x : x[2], reverse = True)

            mask = numpy.zeros(edges.shape)
            cv2.fillConvexPoly(mask, contourInformation[0][0], (255))

            mask = cv2.dilate(mask, None, iterations = 10)
            # mask = cv2.erode(mask, None, iterations = 2)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
            image_float = image.astype('float32') / 255.0
            mask  = numpy.dstack([mask] * 3).astype('float32') / 255.0
            
            maskedImage = (mask * image_float) + ((1 - mask) * (1.0, 1.0, 1.0))
            maskedImage = (maskedImage * 255).astype('uint8')

            cv2.imshow('Background subtraction', maskedImage)

            print(threshold_low, threshold_high)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':

    main(filePath = '/Users/taehwankim/Downloads/Code/OpenCV/Cropped/Test.jpg')
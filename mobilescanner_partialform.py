from panorama import Stitcher
#http://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
#https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/
import cv2, imutils
from mobilescanner_final_curr import displayImg

def partialform(first, second):
	imageA = cv2.imread(first)
	imageB = cv2.imread(second)
	imageA = imutils.resize(imageA, width=800)
	imageB = imutils.resize(imageB, width=800)
	stitcher = Stitcher()
	(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
	displayImg(imageA, True)
	displayImg(imageB, True)
	displayImg(vis, True)
	displayImg(result, True)
	cv2.imwrite('\\'.join(first.split('\\')[:-1]) + '\\stitched_.png', result)




#partialform('.\\newimgs\\form2\\filledhalf1.jpg', '.\\newimgs\\form2\\filledhalf2.jpg')
partialform('.\\newimgs\\form2\\emptyhalf1.jpg', '.\\newimgs\\form2\\emptyhalf2.jpg')


#cd ..\..\Ananya\umd\acads\sem2\ENEE631\project\code\python1
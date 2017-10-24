from mobilescanner_final_curr import *
import cv2, pytesseract
from PIL import Image

def checkFullness(bwimg):
	#print (np.sum(bwimg == 255), float(np.prod(bwimg.shape)), 'fff')
	return (np.sum(bwimg == 255)/float(np.prod(bwimg.shape)))  #check percentage of white pixels

def skeleton(img):  #http://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
	element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
	done = False
	size = np.size(img)
	skel = np.zeros(img.shape, np.uint8)
	while (not done):
		eroded = cv2.erode(img, element)
		temp = cv2.dilate(eroded, element)
		temp = cv2.subtract(img, temp)
		skel = cv2.bitwise_or(skel, temp)
		img = eroded.copy()

		zeros = size - cv2.countNonZero(img)
		if zeros == size:
			done = True

	return skel

def checkClosenessToBoundary(box, imgshape):
	v1 = box[0]; v2 = box[1]
	if v1[0] == 0 or v1[1] == 0 or v2[0] == 0 or v2[1] == 0:
		return True
	if v1[0] == imgshape[1] or v2[0] == imgshape[1]:
		return True
	if v1[1] == imgshape[0] or v2[1] == imgshape[0]:
		return True
	return False

def getTextBoxes(empty1, empty2, dim1):
	#note: non square structuring element to join characters together
	tmpp = 255 - empty2
	#displayImg(tmpp, True)
	#tmpp = skeleton(tmpp)
	displayImg(tmpp, True)
	#tmpp = cv2.erode(tmpp, np.ones((4,4)))
	#displayImg(tmpp, True)
	#print ('hsdkjshsd')
	dilatedEmpty = cv2.dilate(tmpp, np.ones((1, int(np.round(dim1/90.)))))  #int(np.round(dim0/500.))
	displayImg(dilatedEmpty, True)
	#print ('eghtdhg')

	textregionsInEmpty = empty1.copy()
	ret, markers, stats, centroids = cv2.connectedComponentsWithStats(dilatedEmpty)
	textrects = []; otherrects = []
	for k in range(ret):
		if stats[k][2]>0.75*stats[k][3]: #if width>height
		#if not (stats[k][3])
			if stats[k][2]>dim1/40:  #if sufficiently wide
				if stats[k][-1] > 0.00005*np.prod(dilatedEmpty.shape):  #area > a certain threshold // 0.00004
					v1 = (stats[k][0], stats[k][1])  #top left corner of rectangle enclosing this connected component
					v2 = (v1[0]+stats[k][2], v1[1]+stats[k][3])  #bottom right corner
					x = checkFullness(dilatedEmpty[v1[1]:v2[1], v1[0]:v2[0]])#, stats[k][-1]
					if not checkClosenessToBoundary([v1, v2], empty1.shape):  #is the box too close to the boundary..then its probably edges of the paper
						if x > 0.5:
							cv2.rectangle(textregionsInEmpty, v1, v2, (0, 255, 0), 5)
							textimg = empty1[v1[1]:v2[1], v1[0]:v2[0], :]
							#displayImg(textimg, True)
							#try:
							text = pytesseract.image_to_string(Image.fromarray(textimg, 'RGB'))
							print (text)
							#except ValueError:
								#print ('failed')
							#print(text)
							textrects += [(v1, v2)]
						else:
							cv2.rectangle(textregionsInEmpty, v1, v2, (255, 0, 0), 5)
							otherrects += [(v1, v2)]

	displayImg(textregionsInEmpty, True)
	return textrects, otherrects


def area(box):
	return abs(box[1][0]-box[0][0])*abs(box[1][1]-box[0][1])

'''
def findMatch(box, textrectsE, dims):
	areaTh = None
	cornerTh = None
	for b in textrectsE:
		if abs(area(box)-area(b)) < areaTh:
			if abs(box[0][0]-b[0][0]) < cornerTh:
				if abs(box[0][1] - b[0][1]) < cornerTh:
					if abs(box[1][0] - b[1][0]) < cornerTh:
						if abs(box[1][1] - b[1][1]) < cornerTh:
							return b
'''

def iou(boxA, boxB):  #http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0][0], boxB[0][0])
	yA = max(boxA[0][1], boxB[0][1])
	xB = min(boxA[1][0], boxB[1][1])
	yB = min(boxA[1][1], boxB[1][1])

	if xB - xA + 1>0 and yB - yA + 1:
		interArea = (xB - xA + 1) * (yB - yA + 1)  # compute the area of intersection rectangle
	else:
		interArea = 0

	# compute the area of both rectangles
	boxAArea = (boxA[0][0] - boxA[1][0] + 1) * (boxA[0][1] - boxA[1][1] + 1)
	boxBArea = (boxB[0][0] - boxB[1][0] + 1) * (boxB[0][1] - boxB[1][1] + 1)

	return interArea / float(boxAArea + boxBArea - interArea)  #AUB = A + B - AIB


def findMatch(box, textrectsE):
	maxiou = -1
	for b in textrectsE:
		iouval = iou(b, box)
		if iouval > maxiou:
			maxiou = iouval
			bestbox = b
	return bestbox, maxiou

def matchboxes(boxesFromEmptyForm, boxesFromFullForm):
	th = 0.1; newboxlist = []
	for boxID in range(len(boxesFromFullForm)):
		matchbox, matchboxscore = findMatch(boxesFromFullForm[boxID], boxesFromEmptyForm)
		#print (matchboxscore)
		a1 = area(matchbox)
		a2 = area(boxesFromFullForm[boxID])
		#print(abs(a1 - a2),float(a1),abs(a1 - a2)/float(a1))
		if matchboxscore < th or abs(a1 - a2)/float(a1)>1:  #did not find good matches (not enough overlap, or the box areas dont match)
			newboxlist += [boxID]
	return newboxlist

def regionOfInterest(empty1, textrectsF, nomatch1, otherrectsF, nomatch2):
	textregionsInEmpty = empty1.copy()
	for k in nomatch1:
		cv2.rectangle(textregionsInEmpty, textrectsF[k][0], textrectsF[k][1], (0, 0, 255), 5)
	for k in nomatch2:
		cv2.rectangle(textregionsInEmpty, otherrectsF[k][0], otherrectsF[k][1], (0, 0, 255), 5)
	displayImg(textregionsInEmpty, True)

#run 'chcp 65001' on terminal
def formFilling(empty, filled):
	if False:
		dim1 = 1500; dim0 = 2000
		emptyNorm1, emptyNorm2 = normalizeImg(empty, outsize=[dim1, dim0])
		filledNorm1, filledNorm2 = normalizeImg(filled, outsize=[dim1, dim0])
	else:
		emptyNorm1, emptyNorm2 = normalizeImg(empty)
		filledNorm1, filledNorm2 = normalizeImg(filled)
		dim0 = max(filledNorm2.shape[0], emptyNorm2.shape[0])
		dim1 = max(filledNorm2.shape[1], emptyNorm2.shape[1])
		filledNorm2 = cv2.resize(filledNorm2, (dim1, dim0))
		filledNorm2 = (filledNorm2 > 128).astype(np.uint8) * 255  #resizing introduces some non 0 and non 255 values, so thresholding again
		emptyNorm2 = cv2.resize(emptyNorm2, (dim1, dim0))
		emptyNorm2 = (emptyNorm2 > 128).astype(np.uint8) * 255
		filledNorm1 = cv2.resize(filledNorm1, (dim1, dim0))
		emptyNorm1 = cv2.resize(emptyNorm1, (dim1, dim0))
		#pdb.set_trace()


	displayImg(emptyNorm1, True)
	#print ('xxxxxxxxxxxxx')

	displayImg(filledNorm1, True)


	#dilatedFull = cv2.dilate(255 - filled2, np.ones((1, int(np.round(dim1 / 120.)))))
	#displayImg(dilatedFull, True)

	textrectsE, otherrectsE = getTextBoxes(emptyNorm1, emptyNorm2, dim1)
	textrectsF, otherrectsF = getTextBoxes(filledNorm1, filledNorm2, dim1)

	#pdb.set_trace()
	#pdb.set_trace()
	nomatch1 = matchboxes(textrectsE, textrectsF)  #get those textboxes in filled that have no matches in empty
	nomatch2 = matchboxes(otherrectsE, otherrectsF)

	regionOfInterest(filledNorm1, textrectsF, nomatch1, otherrectsF, nomatch2)

	#pdb.set_trace()



formFilling('.\\newimgs\\form4\\fullempty.jpg', '.\\newimgs\\form4\\fullfilled.jpg')
#formFilling('.\\newimgs\\form3\\fullempty1.jpg', '.\\newimgs\\form3\\fullfilled1.jpg')
#formFilling('.\\newimgs\\form2\\testempty.jpg', '.\\newimgs\\form2\\testfilled.jpg')

#x,y = normalizeImg('.\\newimgs\\form2\\testempty.jpg'); displayImg(x, True); displayImg(y, True)
#x,y = normalizeImg('.\\newimgs\\form2\\testfilled.jpg'); displayImg(x, True); displayImg(y, True)
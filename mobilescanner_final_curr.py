from __future__ import division
#from pyimagesearch import imutils
from skimage.filters import threshold_adaptive, threshold_otsu, rank
from skimage.morphology import disk
import numpy as np
import argparse, math
import cv2, pdb, imutils, random
import matplotlib.pyplot as plt
#import thinning, time
from scipy import optimize
from sklearn.cluster import KMeans
from sklearn import linear_model

display=False

def drawline(img, coeff, rad=5):
	for x in range(img.shape[0]):
		y = coeff[0]*x+coeff[1]
		cv2.circle(img, (int(y),int(x)), rad, (255,), -1)


def displayImg(img, display=display):
	if display:
		fig, ax = plt.subplots()
		plt.axis('off')
		#plt.axis('tight')
		if len(img.shape)==2:
			plt.imshow(img, cmap='gray')
		else:
			plt.imshow(img[...,::-1])
		for item in [fig, ax]:
			item.patch.set_visible(False)
		plt.show()

		#cv2.imshow('image', img)
		#cv2.waitKey(0)
		#cv2.destroyAllWindows()

def connectedCompSizeFilter(edges, percentile=0.02, maxcomps=30, percentile2=0.02, maxcomps2=30):
	ret, markers, stats, centroids = cv2.connectedComponentsWithStats(edges)

	compsize = {}
	for i in range(markers.shape[0]):
		for j in range(markers.shape[1]):
			compsize[markers[i,j]] = compsize.get(markers[i,j],0)+1

	componentsSizes = sorted(compsize.values(),reverse=True)[1:]  #leave the first one out, it belongs to the background probably

	cutoff = int(min(int(len(componentsSizes)*percentile), maxcomps))  #keep top percentile% lengths


	#the component should be long
	#tt = ([max(i[2],i[3]) for i in stats])
	tt = ([i[2]**2 + i[3]**2 for i in stats])
	ttt = sorted(tt, reverse=True)[1:]
	cutoff2 = int(min((len(ttt)*percentile2), maxcomps2))
	#print cutoff, cutoff2, 'dfdfd'
	markedForDeletion = []
	for k in compsize:   #markedForDeletion = [k for k in compsize if compsize[k] < cutoff]
		if compsize[k] < componentsSizes[cutoff] or tt[k] < ttt[cutoff2]:
			markedForDeletion += [k]

	markedForDeletion = set(markedForDeletion)  #'in' is much faster for set than list
	for i in range(edges.shape[0]):
		for j in range(edges.shape[1]):
			if markers[i,j] in markedForDeletion:
				edges[i,j] = 0

	return edges

def thinme(edges):  ##https://pypi.python.org/pypi/thinning  #thinning algorithm by Guo and Hall
	return None#thinning.guo_hall_thinning(edges)

def reducebits(gray, numbits):
	maxnum = 2**numbits
	#pdb.set_trace()
	return (np.round((maxnum/255)*gray.astype(np.float32)) * 2**(8-numbits)).astype(np.uint8)


def w_choice(seq, prob):
	total_prob = sum(prob)
	chosen = random.uniform(0, total_prob)
	cumulative = 0
	for ii in range(len(seq)):
		item = seq[ii]; probality = prob[ii]
		cumulative += probality
		if cumulative > chosen:
			return item

def resample(linecoeffs):
	totlen = sum([i[3] for i in linecoeffs])
	prob = [i[3]/float(totlen) for i in linecoeffs]
	#pdb.set_trace()
	return [w_choice(linecoeffs, prob) for jj in range(len(linecoeffs))]

#assumptions:
#high contrast between paper edge and table
#quadrilateral paper (straight edges)
#paper is the largest quadrilateral
#all the paper edges are visible

#should handle:
#no corner markings required
#any angle and orientation
def findCorners(img, stitched=None):
	origshape = img.shape
	img = cv2.resize(img, (500, 500))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#gray = reducebits(gray, 4)
	#gray = cv2.equalizeHist(gray)
	displayImg(gray)
	gray = cv2.bilateralFilter(gray,5,35,35)
	displayImg(gray)
	#25, 35
	edges = cv2.Canny(gray,25,35)  #low thresholds give noisy edges, but we shall clean up edges later using connected components
	displayImg(edges)

	if stitched:
		th = 2
		aa = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) == 0).astype(np.uint8) * 255
		#hacky... this is known from image stitching code
		displayImg(aa)
		rr = int(np.floor((449/784)*500))  #these constants are for C:\Ananya\umd\acads\sem2\ENEE631\project\code\python1\newimgs\form2\stitched_.png
		cc = int(np.floor((799/863)*500))  #run mobilescanner_partialform to get these numbers
		#rr = int(np.floor((449/899)*500)) 
		#cc = int(np.floor((799/924)*500))
		
		#aa[rr+5,:cc]=255
		#aa[rr-5,:cc]=255
		
		#aa[:rr, cc+2] = 255
		#aa[:rr, cc-2] = 255
		aa[rr,:cc]=255
		aa[:rr, cc] = 255
		aa[:rr, cc] = 255
		displayImg(aa)
		ttt = np.where(aa == 255)
		tt = np.where(edges == 255)
		cc =0
		for i,j in zip(tt[0], tt[1]):  #this loop is very timeconsuming
			print (cc, len(tt[0])); cc+=1
			for ii, jj in zip(ttt[0], ttt[1]):
				if abs(i - ii) < th and abs(j - jj) < th:
					edges[i, j] = 0
		displayImg(edges)


	edges = connectedCompSizeFilter(edges)
	displayImg(edges)


	cleanedgesmore = True
	if cleanedgesmore:
		#pdb.set_trace()
		edges = cv2.dilate(edges, np.ones((5, 5)))
		#displayImg(edges)
		#ret, markers, stats, centroids = cv2.connectedComponentsWithStats(edges)
		#print ret
		#plt.imshow(markers)
		#plt.show()
		#edges = thinme(edges)
		#displayImg(edges)

	if False:
		ret, markers, stats, centroids = cv2.connectedComponentsWithStats(edges)
		print (ret)
		plt.imshow(markers)
		plt.show()

		minLineLength = 30 #50
		maxLineGap = 5
		lines = cv2.HoughLinesP(edges,1,np.pi/45,10,minLineLength,maxLineGap)
		pp = img.copy()
		for x in range(0, len(lines)):
			for x1,y1,x2,y2 in lines[x]:
				t = random.randint(0,2)
				aa = [0,0,0]; aa[t]=255
				cv2.line(pp,(x1,y1),(x2,y2),tuple(aa),2)
		displayImg(pp, True)
		#pdb.set_trace()


	cnts,_ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[-2:]
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

	periTh = 0.05
	c = cnts[0]

	p = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, periTh * p, True)
	pp = img.copy()
	cv2.drawContours(pp,c,-1,(255,0,0),2)
	displayImg(pp)
	
	if len(approx) == 4:  #and area is large enuf
		corners = np.squeeze(approx)
	else:
		print ('didnt find contour')
		#method 1: just find bounding box
		rect = cv2.minAreaRect(c)
		corners = cv2.boxPoints(rect)
		corners1 = np.int0(corners)


		#method 2: using hough lines, and clustering and line fitting
		minLineLength = 30  # 50
		maxLineGap = 5
		contourimg = np.zeros((500,500), np.uint8)
		#pdb.set_trace()
		for i in range(c.shape[0]):
			contourimg[c[i,0,1], c[i,0,0]] = 255
		displayImg(contourimg)
		lines = cv2.HoughLinesP(contourimg, 1, np.pi / 90, 5, minLineLength, maxLineGap)
		linecoeffs = []; lineids = []
		#pdb.set_trace()
		for x in range(0, len(lines)):  #lines is of shape numlines,1,4
			for x1, y1, x2, y2 in lines[x]:
				if x1==x2:
					m = float('inf'); intrcpt = x1 #intercept is technically inf
					angl = math.atan2(1,0)
					length = math.sqrt((y2-y1)**2 + (x2-x1)**2)
				else:
					m = (y2-y1)/(x2-x1)
					intrcpt = y2 - m*x2
					angl = math.atan2(y2-y1, x2-x1)
					length = math.sqrt((y2-y1)**2 + (x2-x1)**2)
				linecoeffs += [(m, intrcpt, angl, length)]
				cv2.line(pp, (x1, y1), (x2, y2), (0,255,0), 2)
		displayImg(pp)
		
		#pdb.set_trace()
		#linecoeffs = resample(linecoeffs)

		mlist = [(kk[0],1000.0)[kk[0]==float('inf')] for kk in linecoeffs]
		mlist = cutlist(mlist, 0.1)
		mlist = [[kk] for kk in mlist]
		kmeans_m = KMeans(n_clusters=2, init='k-means++', max_iter=300).fit(mlist)
		m_clusters = [[], []]
		for x in range(0, len(linecoeffs)):
			tmpp = linecoeffs[x][0]
			if kmeans_m.predict((tmpp,1000.0)[tmpp==float('inf')])[0]==0:
				m_clusters[0] += [x]
			else:
				m_clusters[1] += [x]

		
			
		finalLines = {}
		for m_clusterid, m_cluster in enumerate(m_clusters):
			#now find 2 clusters of c
			clist = [linecoeffs[lineid][1] for lineid in m_cluster]
			clist = cutlist(clist, 0.1)
			clist = [[kk] for kk in clist]
			kmeans_c = KMeans(n_clusters=2, init='k-means++', max_iter=300).fit(clist)
			c_clusters = [[], []]
			for lineid in m_cluster:
				if kmeans_c.predict([[linecoeffs[lineid][1]]])[0]==0:
					c_clusters[0] += [lineid]
				else:
					c_clusters[1] += [lineid]
			finalLines[(m_clusterid, 0)] = c_clusters[0]
			finalLines[(m_clusterid, 1)] = c_clusters[1]
		#pdb.set_trace()

		cnt = 0
		coll = [(0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 0)]
		lineeqdict = {}
		for finalline in finalLines:
			xlist = []; ylist = []
			for lineid in finalLines[finalline]:
				x1, y1, x2, y2 = lines[lineid][0]
				xlist += [x1, x2]
				ylist += [y1, y2]
				cv2.circle(pp, (x1, y1), 5, coll[cnt])
				cv2.circle(pp, (x2, y2), 5, coll[cnt])
			cnt += 1
			lineeqdict[finalline] = ransacfit(ylist, xlist)  #np.polyfit(ylist, xlist, 1)...ransac fits noisy data better
			#plt.plot(xlist, ylist, 'b*');plt.show()
			drawline(pp, lineeqdict[finalline], 2)
		p1 = solvelineq(lineeqdict[0,0],lineeqdict[1,0])
		p2 = solvelineq(lineeqdict[1,0], lineeqdict[0,1])
		p3 = solvelineq(lineeqdict[0,1], lineeqdict[1,1])
		p4 = solvelineq(lineeqdict[1,1], lineeqdict[0,0])
		corners2 = np.array([p1, p2, p3, p4])

		displayImg(pp)
		
		#select which one to use, corners1 or corners2?
		inbounds = lambda n : n>=0 and n<500
		checkCorners = lambda cornerslist : all([inbounds(i[0]) and inbounds(i[1]) for i in cornerslist])
		if not(checkCorners(corners1) and cv2.isContourConvex(corners1)):
			if not(checkCorners(corners2) and cv2.isContourConvex(corners2)):
				print ('both are bad')   #both are bad
				corners = corner1
			else:
				corners = corners2  #1 is bad, 2 is good
		else:
			if not(checkCorners(corners2) and cv2.isContourConvex(corners2)):
				corners = corners1   #1 is good, 2 is bad
			else:
				corners = corners1 if cv2.contourArea(corners1) > cv2.contourArea(corners1) else corners2 #both are good
			
	for i in corners:
		cv2.circle(pp, (i[0], i[1]), 5, (255,255,0), -1)
	displayImg(pp)

	scx = origshape[1]/500
	scy = origshape[0]/500
	return np.array([(int(scx*i[0]), int(scy*i[1])) for i in corners])

def cutlist(clist, perccut):
	clist = sorted(clist); lenclist = len(clist)
	clistnew = clist[int(perccut*lenclist) : int((1-perccut)*lenclist)]
	return clist if len(clistnew)==0 else clistnew
			
def solvelineq(lineeq1, lineeq2):  #lineq2 = [m,c]
	m1,c1 = lineeq1; m2,c2 = lineeq2
	a = [[m1,-1], [m2,-1]]
	b = [-c1, -c2]
	return [int(i) for i in np.linalg.solve(a, b)][-1::-1]  #reversing the list

def dist(p1, p2):
	return np.linalg.norm(np.array(p1) - np.array(p2))

def ransacfit(X, y):
	model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
	model_ransac.fit(np.array(X).reshape(-1,1), y)
	#pdb.set_trace()
	return [model_ransac.estimator_.coef_[0], model_ransac.estimator_.intercept_]

def order_points(pts):  #from here http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")

	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# return the ordered coordinates
	return rect
	
def histeq_colimg(img, histeq=True):
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	if histeq:
		yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
	else:
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		yuv[:,:,0] = clahe.apply(yuv[:,:,0])
	return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


def gammaTransform(img, gamma):
	img = cv2.pow(img/255.0, gamma)
	return np.uint8(img*255)

def postprocess(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(img,5,35,35)
	#eqimg = cv2.equalizeHist(img)
	#displayImg(img, True)
	# create a CLAHE object (Arguments are optional).
	#clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	#cl1 = clahe.apply(img)
	#cv2.imwrite('clahe_2.jpg',cl1)

	#img = gammaTransform(img, 0.75)
	#displayImg(img, True)

	#http://scikit-image.org/docs/dev/auto_examples/xx_applications/plot_thresholding.html
	##method 1
	block_size = 35
	tmp = threshold_adaptive(gray, block_size, offset=10)
	th2 = 255*tmp.astype(np.uint8)
	#medfil = cv2.medianBlur(th2,3)
	#method 2 (slow)
	#local_otsu = rank.otsu(img, disk(15))
	#displayImg(local_otsu, True)
	#th2 = 255*(img > local_otsu).astype(np.uint8)

	#method 3
	#th2 = cv2.adaptiveThreshold(eqimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)

	#pdb.set_trace()
	thtmp = th2.copy()
	thtmp = 255 - thtmp
	thtmp = cv2.dilate(thtmp, np.ones((4, 4)))
	thtmp = 255 - thtmp

	#displayImg(thtmp, True)

	#pdb.set_trace()

	th1 = thtmp.copy()
	idxs = np.where(th1 == 0)
	for i,j in zip(idxs[0], idxs[1]):
		th1[i,j] = gray[i,j]

	#cleanupborder()
	#medfil = cv2.medianBlur(th2,3)
	#medfil = cv2.medianBlur(th1,3)
	return th2, th1

def normalizeImg(imgname, disp=False, stitched=False, outsize=None):
	img = cv2.imread(imgname) if type(imgname)==type('a') else imgname
	corners = order_points(findCorners(img, stitched))
	pp = img.copy()
	for i in corners:
		cv2.circle(pp, (i[0], i[1]), 10, (255,255,0), -1)
	displayImg(pp, disp)



	if outsize is None:
		maxWidth = int(dist(corners[0], corners[1]))
		maxHeight = int(dist(corners[1], corners[2]))
	else:
		maxWidth, maxHeight = outsize
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype="float32")
	#pdb.set_trace()
	M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
	#pdb.set_trace()
	warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
	img = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	eqimg = cv2.equalizeHist(img)
	#print ("entering here")
	displayImg(eqimg, disp)
	#displayImg(eqimg, True)
	#print ("entering here1")
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	#print ("entering here2")
	#displayImg(cl1, disp)
	displayImg(cl1, disp)
	#print ("entering here3")
	finalimg1, finalimg2 = postprocess(warped)
	displayImg(finalimg1, disp)
	displayImg(finalimg2, disp)
	#displayImg(finalimg1, True)
	#displayImg(finalimg2, True)

	
	#print ('xxxx')
	displayImg((warped), disp)
	#print ('xxxxyyyy')
	#pdb.set_trace()
	displayImg(histeq_colimg(warped), disp)
	#print ('xxxxyyyyzzzz')
	displayImg(histeq_colimg(warped, False), disp)
	#pdb.set_trace()

	return warped, finalimg1

	
#TODO: try dbscan iinstead of kmeans
#please comment the below lines when running the files  mobilescanner_formfilling.py, mobilescanner_partialform.py
'''
x,y = normalizeImg('sample1.jpg'); displayImg(x, True); displayImg(y, True)
x,y = normalizeImg('sample4.jpg'); displayImg(x, True); displayImg(y, True)  #full contour
x,y = normalizeImg('sample5.jpg'); displayImg(x, True); displayImg(y, True)  #full contour
x,y = normalizeImg('.\\newimgs\\form2\\fullempty.jpg'); displayImg(x, True); displayImg(y, True)
x,y = normalizeImg('.\\newimgs\\form2\\fullfilled.jpg'); displayImg(x, True); displayImg(y, True)

x,y = normalizeImg('.\\newimgs\\form1\\fullempty.jpg'); displayImg(x, True); displayImg(y, True)
x,y = normalizeImg('.\\newimgs\\form1\\fullfilled.jpg'); displayImg(x, True); displayImg(y, True)
#normalizeImg('form.jpg')

#normalizeImg('.\\newimgs\\form2\\fullempty.jpg')

if True:
	aa = normalizeImg('.\\newimgs\\form2\\stitched_.png', stitched=True)
	displayImg(aa[0], True)
	displayImg(aa[1], True)
	#pdb.set_trace()

'''
#for partial form... to detect quadrilateral contour ...  add a border on the image?




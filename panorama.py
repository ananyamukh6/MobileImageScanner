# import the necessary packages
import numpy as np
import imutils
import cv2, pdb
from mobilescanner_final_curr import displayImg

import sys
import os
import numpy as np
import cv2
import scipy
from scipy.stats import norm
from scipy.signal import convolve2d
import math
from skimage.restoration import inpaint

'''split rgb image to its channels'''
#https://compvisionlab.wordpress.com/2013/05/13/image-blending-using-pyramid/

def split_rgb(image):
	red = None
	green = None
	blue = None
	(blue, green, red) = cv2.split(image)
	return red, green, blue


'''generate a 5x5 kernel'''


def generating_kernel(a):
	w_1d = np.array([0.25 - a / 2.0, 0.25, a, 0.25, 0.25 - a / 2.0])
	return np.outer(w_1d, w_1d)


'''reduce image by 1/2'''


def ireduce(image):
	out = None
	kernel = generating_kernel(0.4)
	outimage = scipy.signal.convolve2d(image, kernel, 'same')
	out = outimage[::2, ::2]
	return out


'''expand image by factor of 2'''


def iexpand(image):
	out = None
	kernel = generating_kernel(0.4)
	outimage = np.zeros((image.shape[0] * 2, image.shape[1] * 2), dtype=np.float64)
	outimage[::2, ::2] = image[:, :]
	out = 4 * scipy.signal.convolve2d(outimage, kernel, 'same')
	return out


'''create a gaussain pyramid of a given image'''


def gauss_pyramid(image, levels):
	output = []
	output.append(image)
	tmp = image
	for i in range(0, levels):
		tmp = ireduce(tmp)
		output.append(tmp)
	return output


'''build a laplacian pyramid'''


def lapl_pyramid(gauss_pyr):
	output = []
	k = len(gauss_pyr)
	for i in range(0, k - 1):
		gu = gauss_pyr[i]
		egu = iexpand(gauss_pyr[i + 1])
		if egu.shape[0] > gu.shape[0]:
			egu = np.delete(egu, (-1), axis=0)
		if egu.shape[1] > gu.shape[1]:
			egu = np.delete(egu, (-1), axis=1)
		output.append(gu - egu)
	output.append(gauss_pyr.pop())
	return output


'''Blend the two laplacian pyramids by weighting them according to the mask.'''


def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
	blended_pyr = []
	k = len(gauss_pyr_mask)
	for i in range(0, k):
		p1 = gauss_pyr_mask[i] * lapl_pyr_white[i]
		p2 = (1 - gauss_pyr_mask[i]) * lapl_pyr_black[i]
		blended_pyr.append(p1 + p2)
	return blended_pyr


'''Reconstruct the image based on its laplacian pyramid.'''


def collapse(lapl_pyr):
	output = None
	output = np.zeros((lapl_pyr[0].shape[0], lapl_pyr[0].shape[1]), dtype=np.float64)
	for i in range(len(lapl_pyr) - 1, 0, -1):
		lap = iexpand(lapl_pyr[i])
		lapb = lapl_pyr[i - 1]
		if lap.shape[0] > lapb.shape[0]:
			lap = np.delete(lap, (-1), axis=0)
		if lap.shape[1] > lapb.shape[1]:
			lap = np.delete(lap, (-1), axis=1)
		tmp = lap + lapb
		lapl_pyr.pop()
		lapl_pyr.pop()
		lapl_pyr.append(tmp)
		output = tmp
	return output


def blenderfunc(image1, image2, mask):
	r1 = None
	g1 = None
	b1 = None
	r2 = None
	g2 = None
	b2 = None
	rm = None
	gm = None
	bm = None

	(r1, g1, b1) = split_rgb(image1)
	(r2, g2, b2) = split_rgb(image2)
	(rm, gm, bm) = split_rgb(mask)

	r1 = r1.astype(float)
	g1 = g1.astype(float)
	b1 = b1.astype(float)

	r2 = r2.astype(float)
	g2 = g2.astype(float)
	b2 = b2.astype(float)

	rm = rm.astype(float) / 255
	gm = gm.astype(float) / 255
	bm = bm.astype(float) / 255

	# Automatically figure out the size
	min_size = min(r1.shape)
	depth = int(math.floor(math.log(min_size, 2))) - 4  # at least 16x16 at the highest level.

	gauss_pyr_maskr = gauss_pyramid(rm, depth)
	gauss_pyr_maskg = gauss_pyramid(gm, depth)
	gauss_pyr_maskb = gauss_pyramid(bm, depth)

	gauss_pyr_image1r = gauss_pyramid(r1, depth)
	gauss_pyr_image1g = gauss_pyramid(g1, depth)
	gauss_pyr_image1b = gauss_pyramid(b1, depth)

	gauss_pyr_image2r = gauss_pyramid(r2, depth)
	gauss_pyr_image2g = gauss_pyramid(g2, depth)
	gauss_pyr_image2b = gauss_pyramid(b2, depth)

	lapl_pyr_image1r = lapl_pyramid(gauss_pyr_image1r)
	lapl_pyr_image1g = lapl_pyramid(gauss_pyr_image1g)
	lapl_pyr_image1b = lapl_pyramid(gauss_pyr_image1b)

	lapl_pyr_image2r = lapl_pyramid(gauss_pyr_image2r)
	lapl_pyr_image2g = lapl_pyramid(gauss_pyr_image2g)
	lapl_pyr_image2b = lapl_pyramid(gauss_pyr_image2b)

	outpyrr = blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr)
	outpyrg = blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg)
	outpyrb = blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb)

	outimgr = collapse(blend(lapl_pyr_image2r, lapl_pyr_image1r, gauss_pyr_maskr))
	outimgg = collapse(blend(lapl_pyr_image2g, lapl_pyr_image1g, gauss_pyr_maskg))
	outimgb = collapse(blend(lapl_pyr_image2b, lapl_pyr_image1b, gauss_pyr_maskb))
	# blending sometimes results in slightly out of bound numbers.
	outimgr[outimgr < 0] = 0
	outimgr[outimgr > 255] = 255
	outimgr = outimgr.astype(np.uint8)

	outimgg[outimgg < 0] = 0
	outimgg[outimgg > 255] = 255
	outimgg = outimgg.astype(np.uint8)

	outimgb[outimgb < 0] = 0
	outimgb[outimgb > 255] = 255
	outimgb = outimgb.astype(np.uint8)

	result = np.zeros(image1.shape, dtype=image1.dtype)
	tmp = []
	tmp.append(outimgb)
	tmp.append(outimgg)
	tmp.append(outimgr)
	result = cv2.merge(tmp, result)
	return result

class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()

	def stitch(self, images, ratio=0.75, reprojThresh=4.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)

		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)

		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]+imageB.shape[0]))
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		print (imageB.shape)
		#pdb.set_trace()

		#get rid of black regions
		blackstartrow = 0
		for k in range((result.shape[0])):
			#pdb.set_trace()
			if not np.all(result[k,:,:]==0):
				blackstartrow = k
		blackstartcol = 0
		for k in range((result.shape[1])):
			if not np.all(result[:, k, :] == 0):
				blackstartcol = k
		result = result[:blackstartrow, :blackstartcol, :]
		
		print (result.shape, imageB.shape)

		#inpaint black regions
		#mask = (np.sum(result, axis=-1)==0).astype(np.uint8)*255
		#result = cv2.inpaint(result, mask, 3, cv2.INPAINT_TELEA)

		#print ('gegeg')
		#mask = (np.sum(result, axis=-1) == 0).astype(np.uint8) * 255
		#result = inpaint.inpaint_biharmonic(result, mask, multichannel=True)
		#print ('gegegbdf')


		'''
		for k in range((result.shape[0])):
			flag = False
			for l in range(result.shape[1]-1,result.shape[1]//2,-1):
				if np.sum(result[k, l, :]) != 0: #first non black pixel found
					flag = True
					break
			if flag:
				result[k, l:, :] = np.mean(result[max(k-1,0):k+1, max(l-5,0):l, :], axis=(0,1)).astype(np.uint8)

		for k in range((result.shape[0])):
			flag=False
			for l in range(result.shape[1]//2):
				if np.sum(result[k, l, :]) != 0: #first non black pixel found
					flag=True
					break
			#pdb.set_trace()
			if flag:
				result[k, :l, :] = np.mean(result[max(k-1,0):k+1, l:l+5, :], axis=(0,1)).astype(np.uint8)

		'''

		'''
		mn = np.mean(result, axis=(0,1)).astype(np.uint8)
		for k in range((result.shape[0])):
			for l in range((result.shape[1])):
				if sum(result[k,l,:])==0:
					result[k,l,:] = mn
		'''





		#blending
		#imgBExpanded = np.zeros(result.shape)
		#imgBExpanded[0:imageB.shape[0], 0:imageB.shape[1], :] = imageB
		#pdb.set_trace()
		#mask = np.zeros(result.shape); mask[0:imageB.shape[0], 0:imageB.shape[1], :] = 255; mask = mask.astype(np.uint8)
		#result = blenderfunc(result, imgBExpanded, mask)  #np.zeros(imageB.shape)


		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)

			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)

		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# check to see if we are using OpenCV 3.X
		if self.isv3:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)

		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)

			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)

		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])

		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])

			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)

			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)

		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB

		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

		# return the visualization
		return vis
"""
author : Peb Ruswono Aryan

metric for evaluating binarization algorithms
implemented : 

 * F-Measure
 * pseudo F-Measure (as in H-DIBCO 2010 & 2012)
 * Peak Signal to Noise Ratio (PSNR)
 * Negative Rate Measure (NRM)
 * Misclassification Penaltiy Measure (MPM)
 * Distance Reciprocal Distortion (DRD)

usage:
	python metric.py test-image.png ground-truth-image.png
"""
import numpy as np 
import cv2
# uses https://gist.github.com/pebbie/c2cec958c248339c8537e0b4b90322da for skeletonization
from bwmorph_thin import bwmorph_thin as bwmorph
import os.path as path
import sys

def drd_fn(im, im_gt):
	height, width = im.shape
	neg = np.zeros(im.shape)
	neg[im_gt!=im] = 1
	y, x = np.unravel_index(np.flatnonzero(neg), im.shape)
	
	n = 2
	m = n*2+1
	W = np.zeros((m,m), dtype=np.uint8)
	W[n,n] = 1.
	W = cv2.distanceTransform(1-W, cv2.CV_DIST_L2, cv2.CV_DIST_MASK_PRECISE)
	W[n,n] = 1.
	W = 1./W
	W[n,n] = 0.
	W /= W.sum()
	
	nubn = 0.
	block_size = 8
	for y1 in xrange(0, height, block_size):
		for x1 in xrange(0, width, block_size):
			y2 = min(y1+block_size-1,height-1)
			x2 = min(x1+block_size-1,width-1)
			block_dim = (x2-x1+1)*(y1-y1+1)
			block = 1-im_gt[y1:y2, x1:x2]
			block_sum = np.sum(block)
			if block_sum>0 and block_sum<block_dim:
				nubn += 1

	drd_sum= 0.
	tmp = np.zeros(W.shape)
	for i in xrange(min(1,len(y))):
		tmp[:,:] = 0 

		x1 = max(0, x[i]-n)
		y1 = max(0, y[i]-n)
		x2 = min(width-1, x[i]+n)
		y2 = min(height-1, y[i]+n)

		yy1 = y1-y[i]+n
		yy2 = y2-y[i]+n
		xx1 = x1-x[i]+n
		xx2 = x2-x[i]+n

		tmp[yy1:yy2+1,xx1:xx2+1] = np.abs(im[y[i],x[i]]-im_gt[y1:y2+1,x1:x2+1])
		tmp *= W

		drd_sum += np.sum(tmp)
	return drd_sum/nubn

if __name__=="__main__":
	if len(sys.argv)<3:
		print( sys.argv[0],"input-image ground-truth-image")
		sys.exit(1)
	if not (path.exists(sys.argv[1]) and path.exists(sys.argv[2])):
		print( "file not found")
		sys.exit(1)
	im = cv2.imread(sys.argv[1],0)
	im_gt = cv2.imread(sys.argv[2], 0)

	height, width = im.shape
	npixel = height*width

	im[im>0] = 1
	gt_mask = im_gt==0
	im_gt[im_gt>0] = 1

	sk = bwmorph(1-im_gt)
	im_sk = np.ones(im_gt.shape)
	im_sk[sk] = 0
	
	kernel = np.ones((3,3), dtype=np.uint8)
	im_dil = cv2.erode(im_gt, kernel)
	im_gtb = im_gt-im_dil
	im_gtbd = cv2.distanceTransform(1-im_gtb, cv2.CV_DIST_L2, 3)
	
	nd = im_gtbd.sum()

	ptp = np.zeros(im_gt.shape)
	ptp[(im==0) & (im_sk==0)] = 1
	numptp = ptp.sum()

	tp = np.zeros(im_gt.shape)
	tp[(im==0) & (im_gt==0)] = 1
	numtp = tp.sum()

	tn = np.zeros(im_gt.shape)
	tn[(im==1) & (im_gt==1)] = 1
	numtn = tn.sum()

	fp = np.zeros(im_gt.shape)
	fp[(im==0) & (im_gt==1)] = 1
	numfp = fp.sum()

	fn = np.zeros(im_gt.shape)
	fn[(im==1) & (im_gt==0)] = 1
	numfn = fn.sum()

	precision = numtp / (numtp + numfp)
	recall = numtp / (numtp + numfn)
	precall = numptp / np.sum(1-im_sk)
	fmeasure = (2*recall*precision)/(recall+precision)
	pfmeasure = (2*precall*precision)/(precall+precision)

	mse = (numfp+numfn)/npixel
	psnr = 10.*np.log10(1./mse)

	nrfn = numfn / (numfn + numtp)
	nrfp = numfp / (numfp + numtn)
	nrm = (nrfn + nrfp)/2

	im_dn = im_gtbd.copy()
	im_dn[fn==0] = 0
	dn = np.sum(im_dn)
	mpfn = dn / nd

	im_dp = im_gtbd.copy()
	im_dp[fp==0] = 0;
	dp = np.sum(im_dp)
	mpfp = dp / nd

	mpm = (mpfp + mpfn) / 2
	drd = drd_fn(im, im_gt)

	print( "F-measure\t: {0}\npF-measure\t: {1}\nPSNR\t\t: {2}\nNRM\t\t: {3}\nMPM\t\t: {4}\nDRD\t\t: {5}".format(fmeasure, pfmeasure, psnr, nrm, mpm, drd))
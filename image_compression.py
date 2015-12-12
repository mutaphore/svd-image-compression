# Image compression using SVD
# Author: Dewei Chen
# 12/12/15
# This program has dependency on: 
# skimage, matplotlib, scipy, numpy and PIL

import argparse

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

from scipy import misc
from scipy import linalg

from PIL import Image

from skimage import img_as_float
from skimage.measure import structural_similarity as ssim


def svd_compress_gs(img, k):
	"""Given a matrix representing a grayscale image, compress 
	it by taking the largest k elements from its singular values"""
	U, singular_vals, V = linalg.svd(img)
	rank = len(singular_vals)
	if k > rank:
		print "k is larger than rank of image %r" % rank
		return img
	# take columns less than k from U
	U_p = U[:,:k]
	# take rows less than k from V
	V_p = V[:k,:]
	# build the new S matrix with top k diagnal elements
	S_p = np.zeros((k, k))
	for i in range(k):
		S_p[i][i] = singular_vals[i]
	print "U_p shape {0}, S_p shape {1}, V_p shape {2}".format(
		U_p.shape, S_p.shape, V_p.shape)
	compressed = np.dot(np.dot(U_p, S_p), V_p)
	ss = ssim(img, compressed,
		dynamic_range=compressed.max()-compressed.min())
	print "Strucural similarity: %r" % ss
	return compressed

def svd_compress_rgb(img, k_r, k_g, k_b):
	"""Given a matrix representing a RGB image, compress 
	it by taking the largest k elements from its singular values"""
	# split into separate channels
	comp_r = svd_compress_gs(img[:,:,0], k_r)
	comp_g = svd_compress_gs(img[:,:,1], k_g)
	comp_b = svd_compress_gs(img[:,:,2], k_b)
	new_img = np.zeros(img.shape, 'float64')
	nrows = img.shape[0]
	ncols = img.shape[1]
	nchans = img.shape[2]
	for i in range(nrows):
		for j in range(ncols):
			for c in range(nchans):
				val = 0
				if c == 0:
					val = comp_r[i][j]
				elif c == 1:
					val = comp_g[i][j]
				else:
					val = comp_b[i][j]
				new_img[i][j][c] = val
	return new_img

def compress_ratio(orig_img, k):
	"""Calculate the compression ratio of an compressed image
	to the original image."""
	m = orig_img.shape[0]
	n = orig_img.shape[1]
	comp_bytes = 0
	if len(orig_img.shape) > 2:
		comp_bytes += k[0] * (m + n + 1)
		comp_bytes += k[1] * (m + n + 1)
		comp_bytes += k[2] * (m + n + 1)
		return comp_bytes / (3 * float(m**2))
	else:
		comp_bytes = k[0] * (m + n + 1)
		return comp_bytes / float(m**2)

def main():
	parser = argparse.ArgumentParser(description='Image compression with SVD')
	parser.add_argument('-c', dest='compress', nargs='?', help='compress image using SVD')
	parser.add_argument('-k', dest='k', nargs='*', default=['5', '5', '5'], help='compression factor k (default 5)')
	parser.add_argument('-s', dest='ssim', nargs=2, help='calculate ssim between 2 images')
	parser.add_argument('-r', dest='size', type=int, default=100, help='image resize percentage (default 100)')
	args = parser.parse_args()
	args.k = [int(x) for x in args.k]

	if args.ssim:
		img1 = img_as_float(mpimg.imread(args.ssim[0]))
		img2 = img_as_float(mpimg.imread(args.ssim[1]))
		ss = ssim(img1, img2)
		print "Strucural similarity: %r" % ss
	elif args.compress:
		img = mpimg.imread(args.compress)
		print "Original image dimensions {0}".format(img.shape)
		if args.size < 100:
			img = misc.imresize(img, args.size)
		img = img_as_float(img)
		# check if this is an RGB or grayscale image
		if len(img.shape) > 2:
			if 
			compressed = svd_compress_rgb(img, args.k[0], args.k[1], args.k[2])
		else:
			compressed = svd_compress_gs(img, args.k[0])
		print "Compression ratio: %r" % compress_ratio(img, args.k)
		plt.figure(figsize=(10, 3.6))
		plt.imshow(compressed, cmap=plt.cm.gray)
		plt.show()
	elif args.size < 100:
		print "Resizing image to {0}%".format(args.r)
		img = misc.imresize(img, args.size)
		plt.figure(figsize=(10, 3.6))
		plt.imshow(img)
		plt.show()
	else:
		parser.print_help()

if __name__ == '__main__':
	main()
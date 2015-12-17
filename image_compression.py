# Image compression using SVD
#
# Author: Dewei Chen
# Date: 12/12/15
#
# This program has following dependencies:
# skimage, matplotlib, scipy, numpy and PIL

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import linalg
from PIL import Image
from skimage import io, img_as_float, img_as_uint
from skimage.measure import structural_similarity as ssim

def combine(U, S, V):
	return np.dot(np.dot(U, S), V)

def svd_compress_ssim(img, target_ss):
	"""Compress image by finding k that is closest to target ssim.
	Since rank and ssim relationship is linear, we do a 
	binary search, followed by finer grained linear search"""
	rank = min(img.shape[0], img.shape[1])
	left = 1
	right = rank
	last_ss = 100
	k = 1
	compressed = None
	U, singular_vals, V = linalg.svd(img)
	# binary search
	while left < right:	
		k = (left + right) / 2
		S_p = np.zeros((k, k), img.dtype)
		for i in range(k):
			S_p[i][i] = singular_vals[i]
		compressed = combine(U[:,:k], S_p, V[:k,:])
		ss = ssim(img, compressed,
			dynamic_range=compressed.max()-compressed.min())
		if abs(ss - target_ss) < abs(last_ss - target_ss):
			last_ss = ss
			if ss > target_ss:
				right = k
			else:
				left = k
		else:
			break
	# more fine grained linear search
	if last_ss < target_ss:
		while 1:
			S_p = np.zeros((k + 1, k + 1), img.dtype)
			for i in range(k + 1):
				S_p[i][i] = singular_vals[i]
			compressed = combine(U[:,:k+1], S_p, V[:k+1,:])
			ss = ssim(img, compressed,
				dynamic_range=compressed.max()-compressed.min())
			if abs(ss - target_ss) < abs(last_ss - target_ss):
				last_ss = ss
				k += 1	
			else:
				break
	else:
		while 1:
			S_p = np.zeros((k - 1, k - 1), img.dtype)
			for i in range(k - 1):
				S_p[i][i] = singular_vals[i]
			compressed = combine(U[:,:k-1], S_p, V[:k-1,:])
			ss = ssim(img, compressed,
				dynamic_range=compressed.max()-compressed.min())
			if abs(ss - target_ss) < abs(last_ss - target_ss):
				last_ss = ss
				k -= 1
			else:
				break	
	print "Best k found %r with ssim %r" % (k, last_ss)
	return compressed

def svd_compress_gs(img, k):
	"""Given a matrix representing a grayscale image, compress 
	it by taking the largest k elements from its singular values"""
	U, singular_vals, V = linalg.svd(img)
	rank = len(singular_vals)
	print "Image rank %r" % rank
	if k > rank:
		print "k is larger than rank of image %r" % rank
		return img
	# take columns less than k from U
	U_p = U[:,:k]
	# take rows less than k from V
	V_p = V[:k,:]
	# build the new S matrix with top k diagnal elements
	S_p = np.zeros((k, k), img.dtype)
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
	new_img = np.zeros(img.shape, img.dtype)
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
				# float64 values must be between -1.0 and 1.0
				if val < -1.0:
					val = -1.0
				elif val > 1.0:
					val = 1.0
				new_img[i][j][c] = val
	return new_img

def compress_ratio(orig_img, k):
	"""Calculate the compression ratio of an compressed image
	to the original image."""
	m = float(orig_img.shape[0])
	n = float(orig_img.shape[1])
	comp_bytes = 0
	if len(orig_img.shape) > 2:
		comp_bytes += k[0] * (m + n + 1)
		comp_bytes += k[1] * (m + n + 1)
		comp_bytes += k[2] * (m + n + 1)
		return comp_bytes / (3 * m * n)
	else:
		comp_bytes = k[0] * (m + n + 1)
		return comp_bytes / (m * n)

def compress_images_k(in_dir, out_dir, num_imgs, img_format, k_values):
	"""Compress images with different k values. Compressed format is png."""
	for i in range(1, num_imgs+1):
		img = io.imread("{0}/{1}.{2}".format(in_dir, i, img_format))
		img = img_as_float(img)
		for k in k_values:
			filename = "{0}/{1}/{2}.png".format(out_dir, i, k) 
			print filename
			# check if this is an RGB or grayscale image
			compressed = None
			if len(img.shape) > 2:
				compressed = svd_compress_rgb(img, k, k, k)
			else:
				compressed = svd_compress_gs(img, k)
			io.imsave(filename, compressed)

def compress_images_ss(in_dir, out_dir, num_imgs, img_format, ss_values):
	"""Compress images with different k values. Compressed format is png."""
	for i in range(1, num_imgs+1):
		img = io.imread("{0}/{1}.{2}".format(in_dir, i, img_format))
		img = img_as_float(img)
		for ss in ss_values:
			filename = "{0}/{1}/{2}.png".format(out_dir, i, str(ss).replace('.', ''))
			print filename
			# check if this is an RGB or grayscale image
			compressed = svd_compress_ssim(img, ss)
			io.imsave(filename, compressed)

def main():
	parser = argparse.ArgumentParser(description='Image compression with SVD')
	parser.add_argument('-c', dest='compress', nargs='?', help='compress image using SVD')
	parser.add_argument('-k', dest='k', nargs='*', default=['5', '5', '5'], help='compression factor k (default 5)')
	parser.add_argument('-s', dest='ssim', nargs=2, help='calculate ssim between 2 images')
	parser.add_argument('-r', dest='size', type=int, default=100, help='image resize percentage (default 100)')
	parser.add_argument('-f', dest='fname', nargs='?', help='saved compressed image to file')
	args = parser.parse_args()
	args.k = [int(x) for x in args.k]

	if args.ssim:
		img1 = img_as_float(io.imread(args.ssim[0]))
		img2 = img_as_float(io.imread(args.ssim[1]))
		ss = ssim(img1, img2)
		print "Strucural similarity: %r" % ss
	elif args.compress:
		img = io.imread(args.compress)
		print "Original image dimensions {0}".format(img.shape)
		if args.size < 100:
			img = misc.imresize(img, args.size)
		img = img_as_float(img)
		# check if this is an RGB or grayscale image
		if len(img.shape) > 2:
			if len(args.k) != img.shape[2]:
				print "Provide correct number of k values (%r)" % img.shape[2]
				return 
			compressed = svd_compress_rgb(img, args.k[0], args.k[1], args.k[2])
		else:
			compressed = svd_compress_gs(img, args.k[0])
		print "Compression ratio: %r" % compress_ratio(img, args.k)
		if args.fname:
			io.imsave(args.fname, compressed)
		io.imshow(compressed)
		io.show()
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

	# compress images with different k values
	# compress_images_k("../gs", "../gsdiffk", 10, "pgm", [1, 3, 5, 10, 15, 20])
	# compress_images_k("../rgb", "../rgbdiffk", 10, "jpg", [5, 15, 25, 35, 45])

	# compress images with different target ss values using svd_compress_ssim
	# compress_images_ss("../gs", "../gsdiffss", 10, "pgm", [0.5, 0.6, 0.7, 0.8, 0.9])
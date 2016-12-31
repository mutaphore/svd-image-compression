Optimal Facial Image Compression by SVD
======

### Introduction
Images are prevalent in the Internet. While an excellent way to communicate information ("an image is worth a thousand words"), it is also an expensive form of data for transport over the network. To solve this problem, compression software is often used to encode an image to a smaller physical size so it can be transported more efficiently while minimizing the perceptible quality change from its original copy. How much could an image be compressed before it is no longer valuable in providing information to the user? In this project we explore how to optimize an image compression algorithm based on finding the singular value decomposition of a matrix. For an objective measurement, we introduce an image structural similarity index to the algorithm so it could adjust the compression ratio via a feedback loop.

### Compression examples

Images at various levels of k: 5, 15, 25, 35 going from left to right, top to bottom.

![example1](images/ex1.png)

![example2](images/ex2.png)

![example3](images/ex3.png)

### Plots

![plot1](images/plot1.png)

![plot2](images/plot2.png)

![plot3](images/plot3.png)

### How to run
This program depends on the following Python libraries:
* scipy
* numpy
* PIL
* skimage
* matplotlib

After dependencies are installed with pip, you can run like the following:
```
$ python image_compression.py
usage: image_compression.py [-h] [-c [COMPRESS]] [-k [K [K ...]]]
                            [-s SSIM SSIM] [-r SIZE] [-f [FNAME]]

Image compression with SVD

optional arguments:
  -h, --help      show this help message and exit
  -c [COMPRESS]   compress image using SVD
  -k [K [K ...]]  compression factor k (default 5)
  -s SSIM SSIM    calculate ssim between 2 images
  -r SIZE         image resize percentage (default 100)
  -f [FNAME]      saved compressed image to file
```

For example, to compress a gray scale image saving its top 10 singular values:
```
python image_compression.py -c myimage.png -k 10
```

For colored (RGB) images, if you don't want to use the default, make sure you enter a k value for each channel like this:
```
python image_compression.py -c myimage.png -k 10 10 10
```



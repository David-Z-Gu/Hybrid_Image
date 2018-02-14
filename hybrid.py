import sys
sys.path.append('/Users/kb/bin/opencv-3.1.0/build/lib/')

import cv2
import numpy as np
import math

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN

    kernel_height, kernel_width = kernel.shape
    k_height = int(kernel_height /2)
    k_width = int(kernel_width /2)

    if len(img.shape) == 3:
        img_height, img_width, _ = img.shape
        mod_img = np.zeros((img_height, img_width, 3))

        img = np.pad(img,((k_height, k_height), (k_width, k_width), (0,0)), "constant", constant_values = 0)


        for channel in range(3):
            #GET IMG SHAPE (x,y) value
            #mod_channel = np.zeros((img_height, img_width))
            #temp = np.zeros((img_height+k_height*2, img_width+k_width*2))
            #temp[k_height: img_height+k_height, k_width: img_width+k_width] = img[:,:,channel]
            for i in range(img_height):
                for j in range(img_width):
                    t = img[i:i+kernel_height, j:j+kernel_width, channel]

                    mod_img[i][j][channel] = (kernel*t).sum()

            #mod_img.append(mod_channel)

    else:
        img_height, img_width = img.shape
        mod_img = np.zeros((img_height, img_width))

        img = np.pad(img,((k_height, k_height), (k_width, k_width)), "constant", constant_values = 0)

        for i in range(img_height):
            for j in range(img_width):
                t = img[i:i+kernel_height, j:j+kernel_width]
                mod_img[i][j] = (kernel*t).sum()

    return mod_img
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    #print "yes", img.size
    #flipped_kernel = np.fliplr(np.flipud(kernel))

    result = cross_correlation_2d(img, np.fliplr(np.flipud(kernel)))
    return result
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, width, height):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions width x height such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    gaussian = np.zeros((height, width))
    regulated = 0
    for u in range(height):
        for v in range(width):
            x=v-width/2
            y=u-height/2
            temp = (1.0/(2*math.pi*(sigma**2)))*np.exp(-(x**2+y**2)/(2.0*(sigma**2)))
            gaussian[u, v] = temp
            regulated += temp
    return gaussian.Tc/regulated
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    gaussian = gaussian_blur_kernel_2d(sigma, size, size)
    blurred_img = convolve_2d (img, gaussian)
    return blurred_img
    #raise Exception("TODO in hybrid.py not implemented")
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    #raise Exception("TODO in hybrid.py not implemented")
    # gaussian = gaussian_blur_kernel_2d(sigma, size, size)
    # blurred_img = convolve_2d (img, gaussian)
    return img - low_pass(img, sigma, size)
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

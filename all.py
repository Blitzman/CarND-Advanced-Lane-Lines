import cv2
import glob
import numpy as np
import seaborn as sbs

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

###################################################################################################
## Camera Calibration
###################################################################################################

print('Camera calibration...')

nx = 9 # Number of columns
ny = 6 # Number of rows

obj_points = [] # 3D points in real-world space
img_points = [] # 2D points in image plane

objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

calibration_filenames = glob.glob("camera_cal/calibration*.jpg")

image_shape = []

for calibration_filename in calibration_filenames:

    print('Getting object and image points for ' + calibration_filename)

    calibration_img = mpimg.imread(calibration_filename)
    grayscale = cv2.cvtColor(calibration_img, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(grayscale, (nx, ny), None)

    if ret == True:
        image_shape = grayscale.shape[::-1]
        img_points.append(corners)
        obj_points.append(objp)


ret, camera_mtx, dist_coeffs, r_vecs, t_vecs = cv2.calibrateCamera(obj_points, img_points, image_shape, None, None)

print('Calibration done...')
print(camera_mtx)
print(dist_coeffs)
print(r_vecs)
print(t_vecs)

###################################################################################################
## Test Undistort
###################################################################################################

test_undistort = True

if test_undistort == True:

    for calibration_filename in calibration_filenames:

        calibration_img = mpimg.imread(calibration_filename)
        undistorted = cv2.undistort(calibration_img, camera_mtx, dist_coeffs, None, camera_mtx)

        # Plot original and undistorted image
        sbs.set_style("dark")
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        #f.tight_layout()
        ax1.imshow(calibration_img)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        f.savefig("camera_undistorted/" + calibration_filename)

###################################################################################################
## Load and Undistort Test Images
###################################################################################################

test_images_filenames = glob.glob("test_images/test*.jpg")

test_images = []

for test_image_filename in test_images_filenames:
    test_image = mpimg.imread(test_image_filename)
    undistorted_test_image = cv2.undistort(test_image, camera_mtx, dist_coeffs, None, camera_mtx)
    test_images.append(undistorted_test_image)

    # Plot original and undistorted image
    sbs.set_style("dark")
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    #f.tight_layout()
    ax1.imshow(test_image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted_test_image)
    ax2.set_title('Undistorted Image', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig("test_undistorted/" + test_image_filename)

###################################################################################################
## Thresholding
###################################################################################################

def threshold_x_gradient (img, sobel_size = 3, threshold = [0, 255]):
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel_x = np.uint(255 * abs_sobel_x / np.max(abs_sobel_x))

    binary_output = np.zeros_like(scaled_sobel_x)
    binary_output[(scaled_sobel_x >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1

    return binary_output

## TODO

###################################################################################################
## Perspective Transform
###################################################################################################

## TODO

###################################################################################################
## Finding Lines
###################################################################################################

## TODO

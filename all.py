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

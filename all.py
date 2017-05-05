import cv2
import glob
import numpy as np
import seaborn as sbs

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from moviepy.editor import VideoFileClip

class Line():
    def __init__ (self):
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0, 0, 0], dtype='float')
        self.allx = None
        self.ally = None

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
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2))
        ax1.imshow(calibration_img)
        ax1.set_title('Original Image')
        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image')
        f.savefig("camera_undistorted/" + calibration_filename)


###################################################################################################
## Pipeline
###################################################################################################

left_line = Line()
right_line = Line()

def pipeline(img, filename = None):

    ###############################################################################################
    ## Undistort Image
    ###############################################################################################

    undistorted_image = None

    if filename != None:
        print()
        print("Undistorting...")

    undistorted_image = cv2.undistort(img, camera_mtx, dist_coeffs, None, camera_mtx)

    # Plot original and undistorted image
    if filename != None:

        sbs.set_style("dark")
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax2.imshow(undistorted_image)
        ax2.set_title('Undistorted Image')
        f.savefig("test_undistorted/" + filename)

    ###################################################################################################
    ## Thresholding
    ###################################################################################################

    def threshold_x_gradient (img, sobel_size = 3, threshold = [0, 255]):
        grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
        sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0)
        abs_sobel_x = np.absolute(sobel_x)
        scaled_sobel_x = np.uint(255 * abs_sobel_x / np.max(abs_sobel_x))

        binary_output = np.zeros_like(scaled_sobel_x)
        binary_output[(scaled_sobel_x >= threshold[0]) & (scaled_sobel_x <= threshold[1])] = 1

        return binary_output

    def threshold_hls_s_gradient (img, threshold = [0, 255]):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        binary_output = np.zeros_like(s_channel)
        binary_output[(s_channel >= threshold[0]) & (s_channel <= threshold[1])] = 1

        return binary_output

    def threshold_hsv (img, threshold_low = np.array([0, 0, 0]), threshold_high = np.array([255, 255, 255])):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, threshold_low, threshold_high)

        binary_output = np.zeros_like(hsv[:, :, 2])
        binary_output[(mask > 0)] = 1
        return binary_output

    def sobel (img, sobel_size = 3, sobel_x = 0, sobel_y = 0, threshold = [0, 255]):
        sobel = cv2.Sobel(img, cv2.CV_64F, sobel_x, sobel_y)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint(255 * abs_sobel / np.max(abs_sobel))
        binary_output = np.zeros_like(img)
        binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
        return binary_output

    def threshold_sobel_ls (img, sobel_size = 3, threshold = [0, 255]):
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        
        sobel_s_x = sobel(s_channel, sobel_size, 1, 0, threshold)
        sobel_s_y = sobel(s_channel, sobel_size, 0, 1, threshold)
        sobel_l_x = sobel(l_channel, sobel_size, 1, 0, threshold)
        sobel_l_y = sobel(l_channel, sobel_size, 0, 1, threshold)

        binary_output = np.zeros_like(s_channel)
        binary_output[sobel_s_x | sobel_s_y | sobel_l_x | sobel_l_y] = 1
        return binary_output

    thresholded_image = None

    if filename != None:
        print()
        print("Thresholding...")

    x_binary = threshold_x_gradient(undistorted_image, 3, [20, 100])
    s_binary = threshold_hls_s_gradient(undistorted_image, [170, 255])

    yellow_binary = threshold_hsv(undistorted_image, np.array([0, 80, 200]), np.array([40, 255, 255]))
    white_binary = threshold_hsv(undistorted_image, np.array([20, 0, 200]), np.array([255, 80, 255]))
    sobel_binary = threshold_sobel_ls(undistorted_image, 3, [30, 100])

    colored_binary = np.dstack((np.zeros_like(x_binary), white_binary, yellow_binary, sobel_binary))

    thresholded_image = np.zeros_like(x_binary)
    thresholded_image[(white_binary == 1) | (yellow_binary == 1) | (sobel_binary == 1)] = 1

    # Plot original and thresholded image
    if filename != None:

        sbs.set_style("dark")
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.set_title('Original Image')
        ax1.imshow(undistorted_image)
        ax2.set_title('Stacked Thresholds')
        ax2.imshow(np.uint8(colored_binary * 255.999))
        f.savefig("test_thresholded/" + filename)

    ###################################################################################################
    ## Perspective Transform
    ###################################################################################################

    if filename != None:
        print()
        print("Perspective transformations...")

    def perspective_transform(img, src_points, dst_points):

        img_size = (img.shape[1], img.shape[0])
        src = np.float32(src_points)
        dst = np.float32(dst_points)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size)

        return warped, M

    transformed_image = None
    transformation_matrix = None

    src_tl = [570, 470]
    src_tr = [720, 470]
    src_br = [1130, 720]
    src_bl = [200, 720]

    dst_tl = [320, 0]
    dst_tr = [980, 0]
    dst_br = [980, 720]
    dst_bl = [320, 720]

    src_points = [src_tl, src_tr, src_br, src_bl]
    dst_points = [dst_tl, dst_tr, dst_br, dst_bl]

    thresholded_rgb = np.dstack((np.zeros_like(thresholded_image), thresholded_image, thresholded_image, thresholded_image))
    thresholded_rgb = (np.uint8(thresholded_rgb * 255.999))

    transformed_image, transformation_matrix = perspective_transform(thresholded_rgb, src_points, dst_points)

    transformed_binary = np.zeros_like(thresholded_image)
    transformed_binary[(transformed_image[:, :, 0] > 0) | (transformed_image[:, :, 1] > 0) | (transformed_image[:, :, 2] > 0)] = 1 


    # Plot original and transformed image
    if filename != None:

        sbs.set_style("dark")
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Original Image')
        ax1.imshow(thresholded_image)
        ax2.set_title('Transformed Image')
        ax2.imshow(np.uint8(transformed_binary * 255.999))
        f.savefig("test_transformed/" + filename)

    ###################################################################################################
    ## Finding Lines
    ###################################################################################################

    if filename != None:
        print()
        print("Line finding...")

    def find_peaks (img, filename = None):
    
        histogram = np.sum(img[img.shape[0]//2:, :], axis = 0)

        m = np.int(histogram.shape[0]/2)
        l = np.argmax(histogram[:m])
        r = np.argmax(histogram[m:]) + m

        if (filename != None):
            sbs.set_style("dark")
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.set_title('Original Image')
            ax1.imshow(np.uint8(img * 255.999))
            plt.plot(histogram)
            ax2.set_title('Histogram')
            ax2.imshow(np.uint8(img * 255.999))
            f.savefig("test_histogram/" + filename)

        return m, l, r

    def sliding_window_lines (img, left, right, n_windows = 9, margin = 100, minpix = 100):

        window_height = np.int(img.shape[0] / n_windows)

        nonzero = img.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        left_current = left
        right_current = right

        left_lane_indices = []
        right_lane_indices = []

        left_rects = []
        right_rects = []

        for window in range(n_windows):

            window_y_low = img.shape[0] - (window + 1) * window_height
            window_y_high = img.shape[0] - window * window_height
            window_x_left_low = left_current - margin
            window_x_left_high = left_current + margin
            window_x_right_low = right_current - margin
            window_x_right_high = right_current + margin

            left_rects.append([(window_x_left_low, window_y_low), (window_x_left_high, window_y_high)])
            right_rects.append([(window_x_right_low, window_y_low), (window_x_right_high, window_y_high)])

            good_left_indices = ((nonzero_y >= window_y_low) & (nonzero_y < window_y_high) & (nonzero_x >= window_x_left_low) & (nonzero_x < window_x_left_high)).nonzero()[0]
            good_right_indices = ((nonzero_y >= window_y_low) & (nonzero_y < window_y_high) & (nonzero_x >= window_x_right_low) & (nonzero_x < window_x_right_high)).nonzero()[0]

            left_lane_indices.append(good_left_indices)
            right_lane_indices.append(good_right_indices)

            if len(good_left_indices) > minpix:
                left_current = np.int(np.mean(nonzero_x[good_left_indices]))
            if len(good_right_indices) > minpix:
                right_current = np.int(np.mean(nonzero_x[good_right_indices]))

        left_lane_indices = np.concatenate(left_lane_indices)
        right_lane_indices = np.concatenate(right_lane_indices)

        left_x = nonzero_x[left_lane_indices]
        left_y = nonzero_y[left_lane_indices]
        right_x = nonzero_x[right_lane_indices]
        right_y = nonzero_y[right_lane_indices]

        return left_rects, left_x, left_y, right_rects, right_x, right_y

    lines_image = None

    plot_y = np.linspace(0, transformed_image[0].shape[0]-1, transformed_image[0].shape[0])

    mid, left, right = find_peaks(transformed_binary, filename)
    left_r, left_x, left_y, right_r, right_x, right_y = sliding_window_lines(transformed_binary, left, right)

    lines_image = (np.dstack((transformed_binary, transformed_binary, transformed_binary)) * 255).astype(np.uint8).copy()

    for left_rect, right_rect in zip(left_r, right_r):
        cv2.rectangle(lines_image, left_rect[0], left_rect[1], (0, 255, 0), 2)
        cv2.rectangle(lines_image, right_rect[0], right_rect[1], (0, 255, 0), 2)

    if left_x.size:
        left_fit = np.polyfit(left_y, left_x, 2)
        left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
        left_line.bestx = left_fit_x
        left_line.current_fit = left_fit
        left_line.allx = left_x
        left_line.ally = left_y

    if right_x.size:
        right_fit = np.polyfit(right_y, right_x, 2)
        right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
        right_line.bestx = right_fit_x
        right_line.current_fit = right_fit
        right_line.allx = right_x
        right_line.ally = right_y

    lines_image[left_line.ally, left_line.allx] = [255, 0 ,0]
    lines_image[right_line.ally, right_line.allx] = [0, 0, 255]

    # Plot original and lines image
    if filename != None:

        f = plt.figure()
        plt.imshow(lines_image)
        plt.plot(left_fit_x, plot_y, color='yellow')
        plt.plot(right_fit_x, plot_y, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        f.savefig("test_lines/" + filename)

    ###################################################################################################
    ## Compute Curvature
    ###################################################################################################

    if filename != None:
        print()
        print("Curvature computation...")

    def correct_curve(plot_y, line_x, curve_fit, ympp = 30/720, xmpp = 3.7/700):
        curve_fit_cr = np.polyfit(plot_y * ympp, line_x * xmpp, 2)
        return curve_fit_cr

    def compute_curvature(curve_fit, y_eval = 0, ympp = 30/720):
        curvature = ((1 + (2 * curve_fit[0] * y_eval * ympp + curve_fit[1]) ** 2) ** 1.5) / np.absolute(2 * curve_fit[0])
        return curvature

    left_curve_fit_corrected = correct_curve(plot_y, left_line.bestx, left_line.current_fit)
    left_line.curvature = compute_curvature(left_curve_fit_corrected, np.max(plot_y))

    right_curve_fit_corrected = correct_curve(plot_y, right_line.bestx, right_line.current_fit)
    right_line.curvature = compute_curvature(right_curve_fit_corrected, np.max(plot_y))

    if filename != None:
        print("Curvature left: " + str(left_line.curvature) + " meters")
        print("Curvature right: " + str(right_line.curvature) + " meters")

    ###################################################################################################
    ## Reprojection
    ###################################################################################################

    if filename != None:
        print()
        print("Reprojection...")

    warp_zero = np.zeros_like(transformed_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    points_left = np.array([np.transpose(np.vstack([left_line.bestx, plot_y]))])
    points_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, plot_y])))])
    points = np.hstack((points_left, points_right))

    cv2.fillPoly(color_warp, np.int_([points]), (0, 255, 0))

    new_warp = cv2.warpPerspective(color_warp, np.linalg.inv(transformation_matrix), (transformed_image.shape[1], transformed_image.shape[0]))

    result = cv2.addWeighted(undistorted_image, 1, new_warp, 0.3, 0)

    # Plot original and reprojected image
    if filename != None:

        f = plt.figure()
        plt.imshow(result)
        f.savefig("test_poly/" + filename)

    return result

###################################################################################################
## Load Test Images
###################################################################################################

test_images_filenames = glob.glob("test_images/*.jpg")
test_images = []

for test_image_filename in test_images_filenames:

    print("Loading "  + test_image_filename)
    test_image = mpimg.imread(test_image_filename)
    test_images.append(test_image)

for test_image, test_image_filename in zip(test_images, test_images_filenames):
    print("Processing " + test_image_filename)
    pipeline(test_image, test_image_filename)

###################################################################################################
## Video Processing
###################################################################################################

clip_output_filename = 'project_video_lines.mp4'
clip_input = VideoFileClip('project_video.mp4')
clip_output = clip_input.fl_image(pipeline)
clip_output.write_videofile(clip_output_filename, audio=False)

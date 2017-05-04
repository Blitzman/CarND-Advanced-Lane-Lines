## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undist_cal1]: ./img/calibration_undistort/calibration1.jpg
[undist_cal2]: ./img/calibration_undistort/calibration2.jpg
[undist_cal3]: ./img/calibration_undistort/calibration3.jpg
[undist_cal4]: ./img/calibration_undistort/calibration4.jpg
[undist_cal5]: ./img/calibration_undistort/calibration5.jpg
[undist_cal6]: ./img/calibration_undistort/calibration6.jpg
[undist_cal7]: ./img/calibration_undistort/calibration7.jpg
[undist_cal8]: ./img/calibration_undistort/calibration8.jpg
[undist_cal9]: ./img/calibration_undistort/calibration9.jpg
[undist_cal10]: ./img/calibration_undistort/calibration10.jpg

[test_undist1]: ./img/test_undistorted/straight_lines1.jpg
[test_undist2]: ./img/test_undistorted/straight_lines2.jpg
[test_undist3]: ./img/test_undistorted/test1.jpg
[test_undist4]: ./img/test_undistorted/test2.jpg
[test_undist5]: ./img/test_undistorted/test3.jpg
[test_undist6]: ./img/test_undistorted/test4.jpg
[test_undist7]: ./img/test_undistorted/test5.jpg
[test_undist8]: ./img/test_undistorted/test6.jpg

[test_thresh1]: ./img/test_thresholded/straight_lines1.jpg
[test_thresh2]: ./img/test_thresholded/straight_lines2.jpg
[test_thresh3]: ./img/test_thresholded/test1.jpg
[test_thresh4]: ./img/test_thresholded/test2.jpg
[test_thresh5]: ./img/test_thresholded/test3.jpg
[test_thresh6]: ./img/test_thresholded/test4.jpg
[test_thresh7]: ./img/test_thresholded/test5.jpg
[test_thresh8]: ./img/test_thresholded/test6.jpg

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first section of the `all.py` file (lines 30 to 57).

First, the dimensions of the chessboard are defined by the variables `nx` and `ny` as a 9x6 board. Then we prepare `obj_points`, which will be the (x, y, z) coordinates of the chessboard corners in the world (assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image). Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time we successfully detect all chessboard corners in a test image. `img_points` will be appended with the (x, y) pixel position of each one of the corners in the image plane with each successful chessboard detection.  

We then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  We then applied this distortion correction to the test images using the `cv2.undistort()` function and obtained these results:

![undist_cal1][undist_cal1]
![undist_cal2][undist_cal2]
![undist_cal3][undist_cal3]
![undist_cal4][undist_cal4]
![undist_cal5][undist_cal5]
![undist_cal6][undist_cal6]
![undist_cal7][undist_cal7]
![undist_cal8][undist_cal8]
![undist_cal9][undist_cal9]
![undist_cal10][undist_cal10]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The first step of the pipeline consists of applying the calibration information to correct distortion on the test images using `cv2.undistort()` as we did before. To demonstrate this step, we provide the results of applying distortion correction to the provided test images:

![test_undist1][test_undist1]
![test_undist2][test_undist2]
![test_undist3][test_undist3]
![test_undist4][test_undist4]
![test_undist5][test_undist5]
![test_undist6][test_undist6]
![test_undist7][test_undist7]
![test_undist8][test_undist8]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![test_thresh1][test_thresh1]
![test_thresh2][test_thresh2]
![test_thresh3][test_thresh3]
![test_thresh4][test_thresh4]
![test_thresh5][test_thresh5]
![test_thresh6][test_thresh6]
![test_thresh7][test_thresh7]
![test_thresh8][test_thresh8]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

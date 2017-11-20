**Advanced Lane Finding Project**

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

[image0]: ./camera_cal/corners_calibration10.jpg "Camera Corners Calibration"
[image1]: ./test_images/test1.jpg "Road Transformed"
[image2]: ./output_images/undisoted_test1.jpg "UnDistored"
[image3]: ./output_images/thresh_test1.jpg "Threshold"
[image4]: ./output_images/warped_test1.jpg "Warped"
[image5]: ./output_images/hist_test1.jpg "Hist"
[image6]: ./output_images/sliding_window_test1.jpg "Sliding"
[image7]: ./output_images/selection_window_test1.jpg "Selection"
[image8]: ./output_images/final_test1.jpg "Final"

[video1]: ./output_project_video.mp4 "Project Video"
[video2]: ./output_challenge_video.mp4 "Challenge Video"
[video3]: ./output_harder_challenge_video.mp4 "Harder Challenge Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

All the code is in `lane.py` file.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. Here is one of the image on which corners were calculated. All the processed images are in `camera_cal` directory.

![alt text][image0]

### Pipeline (single images)

For the pipeline, I started to use the `test_images`. All the processed images are in `output_images` directory.
Here is one the image on which all the pipeline processing was done.

![alt text][image1]

#### 1. Provide an example of a distortion-corrected image.

Then using the output of camera cal (calibration coffiecinets), I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: (The undistortion can be noticed in car bumper)

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image defined in the `threshold` function which starts at line 50.  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which starts at line 37.  I selected the following src and dst points using the image size.

```
src = np.float32([[580, 460], [700, 460], [1040, 680], [260, 680]])
dst = np.float32([[200, 0], [1040, 0], [1040,720], [200,720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the second order polynomial fit functions to determine the lane line pixels, the code for it is in the sliding_window and selection_window method at line 78 and 168.

I also plotted the histogram to see the thresholded images are correct such that correct pixels belong to the left line and the right line.


Histogram image:

![alt text][image5]


Sliding window image:

![alt text][image6]


Selection window image:

![alt text][image7]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the curvature method specified in the lesson which is defined in `curvature` function at line 231.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

After determining the curvature, I filled in the lane area as taught in the lesson. Here is an example image which includes the curvatures as well.

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
Using the VideoClip method I processed the video by passing each image to the lane detection pipeline.
Here's a [link to my video result](./output_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The code is not using the optimization for the selection window approach to process first frame using sliding window and next using selection window. The code works well on the project video but not on the challenge videos. Ideally looking to L channel or using the RGB color channels in combination could have helped such as they work well with the white lines.




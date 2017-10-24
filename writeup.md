## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/distortion_corrected_image.png "Road Transformed"
[image3]: ./examples/binary_combo_example.png "Binary Example"
[image4]: ./examples/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

The code with image analysis located in [link](solution.ipynb).

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in [link](experiments.py#L150).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction and crop the image to remove the car front parts to reduce noise in images (see example below):
![alt text][image2]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `apply_warp()`, which appears in [link](experiments.py#L333).  The `apply_warp()` function takes as inputs an image (`image`). I chose the source and destination points coefficient in the following manner:

```python
top_left_x = 0.475
top_left_y = 0.68
top_right_x = 0.54
top_right_y = 0.68
bottom_right_x = 0.785
bottom_right_y = 1.0
bottom_left_x = 0.245
bottom_left_y = 1.0
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| top_left: (608.0, 448.8)      | top_left: (313.6, 0)        | 
| top_right: (691.2, 448.8)      | top_right: (1004.8, 0)      |
| bottom_right: (1004.8, 660.0)     | bottom_right: (1004.8, 660.0)      |
| bottom_left: (313.6, 660.0)      | bottom_left: (313.6, 660.0)        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines [link](experiments.py#L320)).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines [link](experiments.py#L569) and [link](experiments.py#L586)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines [link](experiments.py#L605) in the function `draw_lane_space()`. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./output_images/out_project_video.mp4) and [to my challenge video result](./output_images/out_challenge_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Most of the problems were caused by shadows (challenge video), different lighting conditions. I've spent most of the time playing with color ranges / color spaces. As result I have developed some UI tools that helped me to proceed faster such as [ConfigColorUi.py](ConfigColorUi.py#L1), [ConfigCropUi.py](ConfigCropUi.py#L1), [ConfigRoiUi.py](ConfigRoiUi.py#L1), [ConfigThreshUi.py](ConfigThreshUi.py#L1).

I think to make the algorithm more robust we can apply knowledge from the deep learning course with parameters we have utilized in this project.
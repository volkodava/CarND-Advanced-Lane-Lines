import glob
import io
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

# Chessboard size
chessboard_nx = 9
chessboard_ny = 6

# ROI polygon coefficients
top_left_x = 0.475
top_left_y = 0.68
top_right_x = 0.54
top_right_y = 0.68
bottom_right_x = 0.785
bottom_right_y = 1.0
bottom_left_x = 0.245
bottom_left_y = 1.0

# Color threshold parameters
lower_yellow_1 = 0
lower_yellow_2 = 60
lower_yellow_3 = 150
upper_yellow_1 = 100
upper_yellow_2 = 220
upper_yellow_3 = 250

lower_white_1 = 0
lower_white_2 = 0
lower_white_3 = 220
upper_white_1 = 80
upper_white_2 = 20
upper_white_3 = 300

# Threshold parameters
grad_ksize = 3
grad_thresh_low = 20
grad_thresh_high = 100
mag_binary_ksize = 3
mag_binary_thresh_low = 30
mag_binary_thresh_high = 100
dir_binary_ksize = 15
dir_binary_thresh_low = 0.7
dir_binary_thresh_high = 1.3

# Image crop size
crop_bottom_px = 60

# Sliding Window parameters
n_sliding_windows = 9
window_width_margin = 100
windows_recenter_minpix = 50

# Video processing params
QUEUE_LENGTH = 10


def show_images(images, labels, cols, figsize=(16, 8), title=None):
    assert len(images) == len(labels)

    rows = (len(images) / cols) + 1

    plt.figure(figsize=figsize)

    for idx, image in enumerate(images):
        plt.subplot(rows, cols, idx + 1)
        image = image.squeeze()
        if len(image.shape) == 2:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
        plt.title(labels[idx])
        plt.axis('off')

    if title is not None:
        plt.suptitle(title, fontsize=16)

    plt.tight_layout(pad=3.0)
    plt.show()


def read_images(fnames, gray_conv=lambda rgb_image: cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)):
    images = []
    gray_images = []

    for fname in fnames:
        bgr_image = cv2.imread(fname)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        gray = gray_conv(rgb_image)

        images.append(rgb_image)
        gray_images.append(gray)

    return images, gray_images


def get_chessboard_corners(images, gray_images, nx=chessboard_nx, ny=chessboard_ny):
    pattern_size = (nx, ny)
    upd_images = []

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    # 3d points in real world space
    objpoints = []
    # 2d points in image plane
    imgpoints = []

    for image, gray in zip(images, gray_images):
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # If found, add object points, image points (after refining them)
        upd_img = None
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            upd_img = cv2.drawChessboardCorners(image, pattern_size, corners2, ret)

        upd_images.append(upd_img)

    assert len(images) == len(gray_images) == len(upd_images)

    return objpoints, imgpoints, upd_images


def correct_distortion(image, objpoints, imgpoints):
    height, width = image.shape[:2]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    cam_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 0, (width, height))
    undistorted_image = cv2.undistort(image, mtx, dist, None, cam_matrix)

    return undistorted_image


def warp_image(image, src, dst):
    height, width = image.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped, M, Minv


def mask_image(image, poly_vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(image)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image.shape) > 2:
        # i.e. 3 or 4 depending on your image
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, np.int32([poly_vertices]), ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def filter_color(image, lower_color_mask, upper_color_mask=None, trg_color_space=cv2.COLOR_RGB2HSV):
    if upper_color_mask is None:
        upper_color_mask = np.array([255, 255, 255])

    image = cv2.cvtColor(image, trg_color_space)

    mask = cv2.inRange(image, lower_color_mask, upper_color_mask)
    return cv2.bitwise_and(image, image, mask=mask)


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
def abs_sobel_thresh(image, sobel_kernel=grad_ksize, orient='x', thresh=(grad_thresh_low, grad_thresh_high),
                     rgb2gray=True):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = image
    if rgb2gray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sbl = cv2.Sobel(gray, cv2.CV_64F, orient == 'x', orient == 'y', ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sbl = np.absolute(sbl)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sbl = np.uint8(255 * abs_sbl / np.max(abs_sbl))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sbl)
    binary_output[(scaled_sbl >= thresh[0]) & (scaled_sbl <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return np.float32(binary_output)


# Define a function that applies Sobel x and y,
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thresh(image, sobel_kernel=mag_binary_ksize, thresh=(mag_binary_thresh_low, mag_binary_thresh_high),
               rgb2gray=True):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = image
    if rgb2gray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude
    abs_sobel = np.power(sobelx, 2) + np.power(sobely, 2)
    magnitude = np.sqrt(abs_sobel)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return np.float32(binary_output)


# Define a function that applies Sobel x and y,
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(image, sobel_kernel=dir_binary_ksize, thresh=(dir_binary_thresh_low, dir_binary_thresh_high),
                  rgb2gray=True):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = image
    if rgb2gray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradient_direct = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gradient_direct)
    binary_output[(gradient_direct >= thresh[0]) & (gradient_direct <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return np.float32(binary_output)


def gaussian_blur(image, kernel_size):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def moving_average(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period


def apply_grayscale(image, lower_yellow=np.array([lower_yellow_1, lower_yellow_2, lower_yellow_3]),
                    upper_yellow=np.array([upper_yellow_1, upper_yellow_2, upper_yellow_3]),
                    lower_white=np.array([lower_white_1, lower_white_2, lower_white_3]),
                    upper_white=np.array([upper_white_1, upper_white_2, upper_white_3])):
    yellow_image = filter_color(image, lower_yellow, upper_yellow)
    white_image = filter_color(image, lower_white, upper_white)
    combined = cv2.bitwise_or(yellow_image, white_image)

    return np.mean(combined, axis=2)


def apply_threshold(image):
    gradx = abs_sobel_thresh(image, orient='x', rgb2gray=False)
    grady = abs_sobel_thresh(image, orient='y', rgb2gray=False)
    mag_binary = mag_thresh(image, rgb2gray=False)
    dir_binary = dir_threshold(image, rgb2gray=False)

    combined_sobel = np.zeros_like(gradx)
    combined_sobel[((gradx == 1) & (grady == 1))] = 1

    combined_magn_grad = np.zeros_like(mag_binary)
    combined_magn_grad[((mag_binary == 1) & (dir_binary == 1))] = 1

    result_image = np.zeros_like(combined_sobel)
    result_image[((combined_sobel == 1) | (combined_magn_grad == 1))] = 1
    return result_image


def apply_crop_bottom(image, bottom_px=crop_bottom_px):
    height, width = image.shape[:2]
    return image.copy()[0:height - bottom_px, 0:width]


def apply_warp(image):
    height, width = image.shape[:2]

    src_top_left = (width * top_left_x, height * top_left_y)
    src_top_right = (width * top_right_x, height * top_right_y)
    src_bottom_right = (width * bottom_right_x, height * bottom_right_y)
    src_bottom_left = (width * bottom_left_x, height * bottom_left_y)

    trg_top_left = (width * bottom_left_x, 0)
    trg_top_right = (width * bottom_right_x, 0)
    trg_bottom_right = (width * bottom_right_x, height * bottom_right_y)
    trg_bottom_left = (width * bottom_left_x, height * bottom_left_y)

    src = np.float32([src_top_left, src_top_right, src_bottom_right, src_bottom_left])
    trg = np.float32([trg_top_left, trg_top_right, trg_bottom_right, trg_bottom_left])

    warped_image, M, Minv = warp_image(image, src, trg)
    return warped_image, M, Minv


def grayscale_ro_rgb(grayscale):
    return np.asarray(np.dstack((grayscale, grayscale, grayscale)), dtype=np.float32)


def combine_images_horiz(a, b):
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa + wb
    new_img = np.zeros(shape=(max_height, total_width, 1), dtype=np.float32)

    new_img[:ha, :wa] = a
    new_img[:hb, wa:wa + wb] = b

    return new_img


def calc_moving_average_y(image, num_of_bins=50):
    height, width = image.shape[:2]
    half_image = image[height // 2:, :]

    bin_size = width // num_of_bins
    avg_vertical = np.mean(half_image, axis=0)
    avg_vertical = moving_average(avg_vertical, bin_size)

    return avg_vertical, half_image


def debug_image(thresh_gray_image, num_of_bins=50):
    if thresh_gray_image is None:
        return None

    nrows = 2
    ncols = 1
    plot_number = 1

    avg_y, half_image = calc_moving_average_y(thresh_gray_image, num_of_bins)

    plt.subplot(nrows, ncols, plot_number)
    plt.imshow(half_image, cmap='gray')
    plt.axis('off')
    plot_number += 1

    plt.subplot(nrows, ncols, plot_number)
    plt.plot(avg_y, 'b')
    plt.xlabel('Counts')
    plt.ylabel('Pixel Position')
    plot_number += 1

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    buf.close()

    # close plot before return to stop from adding more information from outer scope
    plt.close()

    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)


def combine_3_images(main, first, second):
    if main is None \
            or first is None \
            or second is None:
        return main

    height, width, depth = main.shape

    result_image = np.zeros((height, width, depth), dtype=np.uint8)

    right_width = width // 4
    right_height = height // 2

    # height, width
    main_size_height = height
    main_size_width = width - right_width
    first_size_height = height - right_height
    first_size_width = right_width
    second_size_height = right_height
    second_size_width = right_width

    main_height_range = (0, main_size_height)
    main_width_range = (0, main_size_width)
    first_height_range = (0, first_size_height)
    first_width_range = (main_size_width, main_size_width + first_size_width)
    second_height_range = (first_size_height, first_size_height + second_size_height)
    second_width_range = (main_size_width, main_size_width + second_size_width)

    # main
    result_image[main_height_range[0]:main_height_range[1], main_width_range[0]:main_width_range[1], :] = \
        cv2.resize(main, (main_size_width, main_size_height))
    # first
    result_image[first_height_range[0]:first_height_range[1], first_width_range[0]:first_width_range[1], :] = \
        cv2.resize(first, (first_size_width, first_size_height))
    # second
    result_image[second_height_range[0]:second_height_range[1], second_width_range[0]:second_width_range[1], :] = \
        cv2.resize(second, (second_size_width, second_size_height))

    return result_image


def find_fitpolynomial(binary_warped, histogram,
                       n_sliding_windows=n_sliding_windows,
                       window_width_margin=window_width_margin,
                       windows_recenter_minpix=windows_recenter_minpix):
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = histogram.shape[0] // 2
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / n_sliding_windows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(n_sliding_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - window_width_margin
        win_xleft_high = leftx_current + window_width_margin
        win_xright_low = rightx_current - window_width_margin
        win_xright_high = rightx_current + window_width_margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > windows_recenter_minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > windows_recenter_minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = right_fit = None
    if len(leftx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    if len(rightx) > 0:
        right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return (leftx, lefty, rightx, righty), (left_fit, right_fit), out_img


def find_fitpolynomial_next(binary_warped, left_fit, right_fit,
                            window_width_margin=window_width_margin):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy
                                   + left_fit[2] - window_width_margin))
                      & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy
                                     + left_fit[2] + window_width_margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy
                                    + right_fit[2] - window_width_margin))
                       & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy
                                      + right_fit[2] + window_width_margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return (leftx, lefty, rightx, righty), (left_fit, right_fit), out_img


def calculate_radius2(x, y, ploty):
    # maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ym_per_pix = 30 / 720
    # meters per pixel in x dimension
    xm_per_pix = 3.7 / 700

    curverad = 0

    if x is not None and len(x) > 0 \
            and y is not None and len(y) > 0:
        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix
                          + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

    return curverad


# TODO: remove me and adjust code
def calculate_radius(leftx, lefty, rightx, righty, ploty):
    # maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    # meters per pixel in y dimension
    ym_per_pix = 30 / 720
    # meters per pixel in x dimension
    xm_per_pix = 3.7 / 700

    left_curverad, right_curverad = (0, 0)

    if leftx is not None and rightx is not None:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix
                               + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix
                                + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    return left_curverad, right_curverad


def draw_lane_space(image, warped, Minv, left_fitx, right_fitx):
    if left_fitx is None or right_fitx is None:
        return image

    height, width = image.shape[:2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    return result


class LaneProcessor:
    def __init__(self, objpoints, imgpoints):
        self.left_lines = deque(maxlen=QUEUE_LENGTH)
        self.right_lines = deque(maxlen=QUEUE_LENGTH)
        self.left_rads = deque(maxlen=QUEUE_LENGTH)
        self.right_rads = deque(maxlen=QUEUE_LENGTH)

        self.objpoints = objpoints
        self.imgpoints = imgpoints

    def mean_value(self, value, values):
        if value is not None:
            values.append(value)

        if len(values) > 0:
            value = np.mean(values, axis=0, dtype=np.int32)
        return value

    def process_simple(self, image):
        image = correct_distortion(image, self.objpoints, self.imgpoints)
        image = apply_crop_bottom(image)

        main_gray_image = apply_grayscale(image)
        main_warped_image, M, Minv = apply_warp(main_gray_image)
        main_thresh_image = np.uint8(apply_threshold(main_warped_image) * 255)

        thresh_debug_image = debug_image(main_thresh_image)

        combined_image = combine_3_images(image, grayscale_ro_rgb(main_warped_image), thresh_debug_image)

        return combined_image

    def process(self, image):
        image = correct_distortion(image, self.objpoints, self.imgpoints)
        image = apply_crop_bottom(image)

        main_gray_image = apply_grayscale(image)
        main_warped_image, M, Minv = apply_warp(main_gray_image)
        main_thresh_image = np.uint8(apply_threshold(main_warped_image) * 255)

        height, width = main_thresh_image.shape[:2]
        ploty = np.linspace(0, height - 1, height)

        # Take a histogram of the bottom half of the image
        # histogram = np.sum(main_thresh_image[main_thresh_image.shape[0] // 2:, :], axis=0)
        histogram, half_image = calc_moving_average_y(main_thresh_image)

        (leftx, lefty, rightx, righty), (left_fit, right_fit), out_img = \
            find_fitpolynomial(main_thresh_image, histogram)

        # Generate x and y values for plotting
        left_line = right_line = None
        if left_fit is None:
            left_line = self.mean_value(None, self.left_lines)
        else:
            left_line = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]

        if right_fit is None:
            right_line = self.mean_value(None, self.right_lines)
        else:
            right_line = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # left_radius, right_radius = \
        #     calculate_radius(leftx, lefty, rightx, righty, ploty)
        left_radius = right_radius = None
        if leftx is None or lefty is None:
            left_radius = self.mean_value(None, self.left_rads)
        else:
            left_radius = calculate_radius2(leftx, lefty, ploty)

        if rightx is None or righty is None:
            right_radius = self.mean_value(None, self.right_rads)
        else:
            right_radius = calculate_radius2(rightx, righty, ploty)

        left_line = self.mean_value(left_line, self.left_lines)
        right_line = self.mean_value(right_line, self.right_lines)

        left_radius = self.mean_value(left_radius, self.left_rads)
        right_radius = self.mean_value(right_radius, self.right_rads)

        if left_line is None or right_line is None:
            return image

        main_lane_space_image = draw_lane_space(image, main_thresh_image, Minv, left_line, right_line)

        search_area_image = get_search_area_image(out_img, left_line, right_line, ploty)
        thresh_debug_image = debug_image(main_thresh_image)

        combined_image = combine_3_images(main_lane_space_image, search_area_image, thresh_debug_image)

        cv2.putText(combined_image, "Left radius: %s m." % left_radius, (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), lineType=cv2.LINE_AA, thickness=2)
        cv2.putText(combined_image, "Right radius: %s m." % right_radius, (80, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), lineType=cv2.LINE_AA, thickness=2)

        return combined_image


def tag_video(finput, foutput, objpoints, imgpoints, subclip_secs=None, simple=False):
    detector = LaneProcessor(objpoints, imgpoints)

    video_clip = VideoFileClip(finput)
    if subclip_secs is not None:
        video_clip = video_clip.subclip(*subclip_secs)

    out_clip = None
    if simple:
        out_clip = video_clip.fl_image(detector.process_simple)
    else:
        out_clip = video_clip.fl_image(detector.process)
    out_clip.write_videofile(foutput, audio=False)


def get_search_area_image(out_img, left_fitx, right_fitx, ploty, window_width_margin=window_width_margin):
    if left_fitx is None or right_fitx is None:
        return None

    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - window_width_margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + window_width_margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - window_width_margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + window_width_margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result


if __name__ == "__main__":
    cal_fnames = [path for path in glob.iglob('camera_cal/*.jpg', recursive=True)]
    cal_images, cal_gray_images = read_images(cal_fnames)
    objpoints, imgpoints, cal_upd_images = get_chessboard_corners(cal_images, cal_gray_images)

    cal_selected_idx = 0
    cal_test_image = cal_images[cal_selected_idx]
    cal_test_image_gray = cal_gray_images[cal_selected_idx]

    cal_undistorted_image = correct_distortion(cal_test_image, objpoints, imgpoints)

    # images_to_show = [cal_test_image, cal_test_image_gray, cal_undistorted_image]
    # labels_to_show = ["Image", "Gray Image", "Undistorted Image"]
    # show_images(images_to_show, labels_to_show, cols=len(images_to_show), title="Calibrate Camera")

    print("num of camera calibration images: ", len(cal_images))
    print("camera calibration image shape: ", cal_images[0].shape)
    print("camera calibration gray image shape: ", cal_gray_images[0].shape)

    example_fnames = ["test_images/test4.jpg", "test_images/test5.jpg"]
    example_images, example_gray_images = read_images(example_fnames)

    # show_images(example_images, labels=example_fnames, cols=len(example_images), title="Input Images")
    # show_images(example_gray_images, labels=example_fnames, cols=len(example_gray_images),
    #             title="Input Gray Images")

    example_selected_idx = 0
    example_test_image = example_images[example_selected_idx]
    example_test_image_gray = example_gray_images[example_selected_idx]

    example_undistorted_image = correct_distortion(example_test_image, objpoints, imgpoints)

    example_cropped_undistorted_image = apply_crop_bottom(example_undistorted_image)

    images_to_show = [example_test_image, example_test_image_gray, example_undistorted_image,
                      example_cropped_undistorted_image]
    labels_to_show = ["Image", "Gray Image", "Undistorted Image", "Undistorted Cropped Image"]
    # show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
    #             title="Input Image Transformation")

    example_test_image = example_cropped_undistorted_image
    example_height, example_width = example_cropped_undistorted_image.shape[:2]

    print("example image width: ", example_width)
    print("example image height: ", example_height)

    src_top_left = (example_width * top_left_x, example_height * top_left_y)
    src_top_right = (example_width * top_right_x, example_height * top_right_y)
    src_bottom_right = (example_width * bottom_right_x, example_height * bottom_right_y)
    src_bottom_left = (example_width * bottom_left_x, example_height * bottom_left_y)

    trg_top_left = (example_width * bottom_left_x, 0)
    trg_top_right = (example_width * bottom_right_x, 0)
    trg_bottom_right = (example_width * bottom_right_x, example_height * bottom_right_y)
    trg_bottom_left = (example_width * bottom_left_x, example_height * bottom_left_y)

    src_vertices = [src_top_left, src_top_right, src_bottom_right, src_bottom_left]
    trg_vertices = [trg_top_left, trg_top_right, trg_bottom_right, trg_bottom_left]

    src = np.float32(src_vertices)
    trg = np.float32(trg_vertices)

    example_warped_image, M, Minv = warp_image(example_test_image, src, trg)
    example_src_masked_image = mask_image(example_test_image, poly_vertices=src_vertices)

    lower_yellow = np.array([lower_yellow_1, lower_yellow_2, lower_yellow_3])
    upper_yellow = np.array([upper_yellow_1, upper_yellow_2, upper_yellow_3])

    example_filtered_yellow = filter_color(example_warped_image, lower_yellow, upper_yellow)

    lower_white = np.array([lower_white_1, lower_white_2, lower_white_3])
    upper_white = np.array([upper_white_1, upper_white_2, upper_white_3])

    example_filtered_white = filter_color(example_warped_image, lower_white, upper_white)

    example_combined_filtered = cv2.bitwise_or(example_filtered_yellow, example_filtered_white)

    images_to_show = [example_test_image, example_src_masked_image, example_warped_image,
                      example_filtered_yellow, example_filtered_white, example_combined_filtered]
    labels_to_show = ["Image", "Masked Image", "Warped Image", "Warped Yellow",
                      "Warped White", "Warped Combined Colors"]
    # show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
    #             title="Warped Input Image Color Transformation")

    gradx = abs_sobel_thresh(example_warped_image, orient='x')
    grady = abs_sobel_thresh(example_warped_image, orient='y')
    mag_binary = mag_thresh(example_warped_image)
    dir_binary = dir_threshold(example_warped_image)

    images_to_show = [example_test_image, example_warped_image, gradx, grady, mag_binary, dir_binary]
    labels_to_show = ["Image", "Warped Image", "Sobel Thresh X", "Sobel Thresh Y", "Magnitude Thresh",
                      "Gradient Direction"]
    # show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
    #             title="Warped Image Threshold Transformation")

    example_warped_gray_image = apply_grayscale(example_warped_image)
    combined_threshold = apply_threshold(example_warped_gray_image)

    images_to_show = [example_test_image, example_warped_image, example_warped_gray_image, combined_threshold]
    labels_to_show = ["Image", "Warped Image", "Warped Gray Image", "Combined Threshold"]
    # show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
    #             title="Warped Gray Image Threshold Transformation")

    main_image = example_test_image.copy()
    main_gray_image = apply_grayscale(example_test_image)
    main_warped_image, M, Minv = apply_warp(main_gray_image)
    main_thresh_image = np.uint8(apply_threshold(main_warped_image) * 255)

    # debug_image(main_thresh_image)
    # thresh_debug_image = debug_image(main_thresh_image)
    #
    # combined_image = combine_3_images(main_image, grayscale_ro_rgb(main_warped_image),
    #                                   thresh_debug_image)
    #
    # plt.imshow(main_thresh_image, cmap="gray")
    # plt.imshow(combined_image)
    # plt.show()

    height, width = main_thresh_image.shape[:2]
    ploty = np.linspace(0, height - 1, height)

    # Take a histogram of the bottom half of the image
    # histogram = np.sum(main_thresh_image[main_thresh_image.shape[0] // 2:, :], axis=0)
    histogram, half_image = calc_moving_average_y(main_thresh_image)

    (leftx, lefty, rightx, righty), (left_fit, right_fit), out_img = \
        find_fitpolynomial(main_thresh_image, histogram)

    # Generate x and y values for plotting
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    plt.close()
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, width)
    plt.ylim(height, 0)
    # plt.show()

    left_radius, right_radius = \
        calculate_radius(leftx, lefty, rightx, righty, ploty)

    print("left_radius: ", left_radius, "m")
    print("right_radius: ", right_radius, "m")

    (leftx, lefty, rightx, righty), (left_fit, right_fit), out_img = \
        find_fitpolynomial_next(main_thresh_image, left_fit, right_fit)

    # Generate x and y values for plotting
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    result = get_search_area_image(out_img, left_fitx, right_fitx, ploty)

    plt.close()
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, width)
    plt.ylim(height, 0)
    # plt.show()

    left_radius, right_radius = \
        calculate_radius(leftx, lefty, rightx, righty, ploty)

    print("left_radius: ", left_radius, "m")
    print("right_radius: ", right_radius, "m")

    main_lane_space_image = draw_lane_space(main_image, main_thresh_image, Minv, left_fitx, right_fitx)

    plt.close()
    plt.imshow(main_lane_space_image)
    plt.axis('off')
    # plt.show()

    # tag_video("project_video.mp4", "%s_qsize_out_project_video.mp4" % QUEUE_LENGTH, objpoints, imgpoints)
    # tag_video("challenge_video.mp4", "%s_qsize_out_challenge_video.mp4" % QUEUE_LENGTH, objpoints, imgpoints)
    # tag_video("project_video.mp4", "%s_qsize_nomean_sub_out_project_video.mp4" % QUEUE_LENGTH, objpoints, imgpoints,
    #           subclip_secs=(10, 15))
    tag_video("challenge_video.mp4", "%s_qsize_out_challenge_video.mp4" % QUEUE_LENGTH, objpoints, imgpoints)

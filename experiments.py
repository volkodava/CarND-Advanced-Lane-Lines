import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ROI polygon coefficients
top_left_x = 0.4
top_left_y = 0.6
top_right_x = 0.6
top_right_y = 0.6
bottom_right_x = 1.0
bottom_right_y = 1.0
bottom_left_x = 0.1
bottom_left_y = 1.0


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


def get_chessboard_corners(images, gray_images, nx=9, ny=6):
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
    warped = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warped, M


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
def abs_sobel_thresh(image, sobel_kernel=3, orient='x', thresh=(0, 255), rgb2gray=True):
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
def mag_thresh(image, sobel_kernel=3, thresh=(0, 255), rgb2gray=True):
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
def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi / 2), rgb2gray=True):
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


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def moving_average(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period


def grayscale(image, lower_yellow=np.array([0, 70, 100]), upper_yellow=np.array([30, 255, 255]),
              lower_white=np.array([0, 0, 220]), upper_white=np.array([255, 40, 255])):
    yellow_image = filter_color(image, lower_yellow, upper_yellow)
    white_image = filter_color(image, lower_white, upper_white)
    combined = cv2.bitwise_or(yellow_image, white_image)

    return np.mean(combined, axis=2)


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

    example_height, example_width = example_test_image.shape[:2]

    print("example image width: ", example_width)
    print("example image height: ", example_height)

    src_top_left = (example_width * top_left_x, example_height * top_left_y)
    src_top_right = (example_width * top_right_x, example_height * top_right_y)
    src_bottom_right = (example_width * bottom_right_x, example_height * bottom_right_y)
    src_bottom_left = (example_width * bottom_left_x, example_height * bottom_left_y)

    trg_top_left = (example_width * bottom_left_x, 0)
    trg_top_right = (example_width * bottom_right_x, 0)
    trg_bottom_right = (example_width * bottom_right_x, example_height)
    trg_bottom_left = (example_width * bottom_left_x, example_height)

    src_vertices = [src_top_left, src_top_right, src_bottom_right, src_bottom_left]
    trg_vertices = [trg_top_left, trg_top_right, trg_bottom_right, trg_bottom_left]

    src = np.float32(src_vertices)
    trg = np.float32(trg_vertices)

    example_warped_image, M = warp_image(example_test_image, src, trg)
    example_src_masked_image = mask_image(example_test_image, poly_vertices=src_vertices)

    lower_yellow = np.array([0, 70, 100])
    upper_yellow = np.array([30, 255, 255])

    example_filtered_yellow = filter_color(example_warped_image, lower_yellow, upper_yellow)

    lower_white = np.array([0, 0, 220])
    upper_white = np.array([255, 40, 255])

    example_filtered_white = filter_color(example_warped_image, lower_white, upper_white)

    example_combined_filtered = cv2.bitwise_or(example_filtered_yellow, example_filtered_white)

    images_to_show = [example_test_image, example_test_image_gray, example_undistorted_image, example_src_masked_image,
                      example_warped_image, example_filtered_yellow, example_filtered_white, example_combined_filtered]
    labels_to_show = ["Image", "Gray Image", "Undistorted Image", "Masked Image", "Warped Image", "Warped Yellow",
                      "Warped White", "Warped Combined Colors"]
    # show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
    #             title="Input Image Transformation")

    ksize = 3
    gradx = abs_sobel_thresh(example_warped_image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(example_warped_image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(example_warped_image, sobel_kernel=ksize, thresh=(30, 100))

    ksize = 15
    dir_binary = dir_threshold(example_warped_image, sobel_kernel=ksize, thresh=(0.7, 1.3))

    images_to_show = [example_test_image, example_warped_image, gradx, grady, mag_binary, dir_binary]
    labels_to_show = ["Image", "Warped Image", "Sobel Thresh X", "Sobel Thresh Y", "Magnitude Thresh",
                      "Gradient Direction"]
    # show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
    #             title="Warped Image Transformation")

    ksize = 3
    example_warped_gray_image = grayscale(example_warped_image)

    gradx = abs_sobel_thresh(example_warped_gray_image, orient='x', sobel_kernel=ksize, thresh=(20, 100),
                             rgb2gray=False)
    grady = abs_sobel_thresh(example_warped_gray_image, orient='y', sobel_kernel=ksize, thresh=(20, 100),
                             rgb2gray=False)
    mag_binary = mag_thresh(example_warped_gray_image, sobel_kernel=ksize, thresh=(30, 100), rgb2gray=False)

    ksize = 15
    dir_binary = dir_threshold(example_warped_gray_image, sobel_kernel=ksize, thresh=(0.7, 1.3), rgb2gray=False)

    images_to_show = [example_test_image, example_warped_image, example_warped_gray_image, gradx, grady, mag_binary,
                      dir_binary]
    labels_to_show = ["Image", "Warped Image", "Warped Gray Image", "Sobel Thresh X", "Sobel Thresh Y",
                      "Magnitude Thresh", "Gradient Direction"]
    show_images(images_to_show, labels=labels_to_show, cols=len(images_to_show) // 2,
                title="Warped Image Transformation")
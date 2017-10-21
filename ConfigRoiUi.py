from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import Slider, CheckBox, ComboBox, Button

from experiments import *


class RoiViewer:
    def __init__(self, search_pattern):
        blur_kernel_size = 1

        cal_fnames = [path for path in glob.iglob('camera_cal/*.jpg', recursive=True)]
        cal_images, cal_gray_images = read_images(cal_fnames)
        self.objpoints, self.imgpoints, _ = get_chessboard_corners(cal_images, cal_gray_images)

        self.blur_kernel_size = blur_kernel_size
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.top_right_x = top_right_x
        self.top_right_y = top_right_y
        self.bottom_right_x = bottom_right_x
        self.bottom_right_y = bottom_right_y
        self.bottom_left_x = bottom_left_x
        self.bottom_left_y = bottom_left_y

        plugin = Plugin(image_filter=self.image_filter, dock="right")

        self.setup_names = ['ROI poly', "ROI Transformation", 'ROI Transformed',
                            "Final Transformation poly", 'Final Transformation', 'Final Transformed']

        self.show_orig = CheckBox('show_orig', value=False, alignment='left')

        plugin += self.show_orig
        plugin += ComboBox('setup', self.setup_names)
        plugin += Slider('blur_kernel_size', 0, 31, value=self.blur_kernel_size, value_type='int')
        plugin += Slider('top_left_x', 0, 1, value=self.top_left_x)
        plugin += Slider('top_left_y', 0, 1, value=self.top_left_y)
        plugin += Slider('top_right_x', 0, 1, value=self.top_right_x)
        plugin += Slider('top_right_y', 0, 1, value=self.top_right_y)
        plugin += Slider('bottom_right_x', 0, 1, value=self.bottom_right_x)
        plugin += Slider('bottom_right_y', 0, 1, value=self.bottom_right_y)
        plugin += Slider('bottom_left_x', 0, 1, value=self.bottom_left_x)
        plugin += Slider('bottom_left_y', 0, 1, value=self.bottom_left_y)
        plugin += Button("Print", callback=self.on_print_click)

        fnames = [path for path in glob.iglob(search_pattern, recursive=True)]
        images, gray_images = read_images(fnames)

        self.viewer = CollectionViewer(images)
        self.viewer += plugin

    def image_filter(self, image, *args, **kwargs):
        print("image: ", image.shape)

        image = correct_distortion(image, self.objpoints, self.imgpoints)
        print("undistorted image: ", image.shape)

        image = apply_crop_bottom(image)
        print("cropped image: ", image.shape)

        show_orig = kwargs["show_orig"]
        setup = kwargs["setup"]
        self.blur_kernel_size = kwargs["blur_kernel_size"]
        self.top_left_x = kwargs["top_left_x"]
        self.top_left_y = kwargs["top_left_y"]
        self.top_right_x = kwargs["top_right_x"]
        self.top_right_y = kwargs["top_right_y"]
        self.bottom_right_x = kwargs["bottom_right_x"]
        self.bottom_right_y = kwargs["bottom_right_y"]
        self.bottom_left_x = kwargs["bottom_left_x"]
        self.bottom_left_y = kwargs["bottom_left_y"]

        if show_orig:
            return image

        height, width = image.shape[:2]
        src_top_left = (width * self.top_left_x, height * self.top_left_y)
        src_top_right = (width * self.top_right_x, height * self.top_right_y)
        src_bottom_right = (width * self.bottom_right_x, height * self.bottom_right_y)
        src_bottom_left = (width * self.bottom_left_x, height * self.bottom_left_y)

        trg_top_left = (width * self.bottom_left_x, 0)
        trg_top_right = (width * self.bottom_right_x, 0)
        trg_bottom_right = (width * self.bottom_right_x, height * self.bottom_right_y)
        trg_bottom_left = (width * self.bottom_left_x, height * self.bottom_left_y)

        grayscale_image = apply_grayscale(image)

        result_image = image
        if setup == "ROI poly":
            vertices = np.array([[src_top_left, src_top_right, src_bottom_right, src_bottom_left]], dtype=np.int32)
            cv2.polylines(result_image, [vertices], isClosed=True, color=(0, 255, 255), thickness=2)
        elif setup == "ROI Transformation":
            vertices = np.array([[trg_top_left, trg_top_right, trg_bottom_right, trg_bottom_left]], dtype=np.int32)
            cv2.polylines(result_image, [vertices], isClosed=True, color=(0, 255, 255), thickness=2)
        elif setup == "ROI Transformed":
            src = np.float32([src_top_left, src_top_right, src_bottom_right, src_bottom_left])
            trg = np.float32([trg_top_left, trg_top_right, trg_bottom_right, trg_bottom_left])
            result_image, M, Minv = warp_image(result_image, src, trg)
        elif setup == "Final Transformation poly":
            blur_image = gaussian_blur(grayscale_image, self.blur_kernel_size)
            result_image = apply_threshold(blur_image)
            vertices = np.array([[src_top_left, src_top_right, src_bottom_right, src_bottom_left]], dtype=np.int32)
            cv2.polylines(result_image, [vertices], isClosed=True, color=(255, 255, 255), thickness=2)
        elif setup == "Final Transformation":
            blur_image = gaussian_blur(grayscale_image, self.blur_kernel_size)
            result_image = apply_threshold(blur_image)
            vertices = np.array([[trg_top_left, trg_top_right, trg_bottom_right, trg_bottom_left]], dtype=np.int32)
            cv2.polylines(result_image, [vertices], isClosed=True, color=(255, 255, 255), thickness=2)
        elif setup == "Final Transformed":
            src = np.float32([src_top_left, src_top_right, src_bottom_right, src_bottom_left])
            trg = np.float32([trg_top_left, trg_top_right, trg_bottom_right, trg_bottom_left])
            result_image, M, Minv = warp_image(grayscale_image, src, trg)
            result_image = gaussian_blur(result_image, self.blur_kernel_size)
            result_image = apply_threshold(result_image)

        return result_image

    def on_print_click(self, args):
        print("""
        blur_kernel_size = {}
        top_left_x = {}
        top_left_y = {}
        top_right_x = {}
        top_right_y = {}
        bottom_right_x = {}
        bottom_right_y = {}
        bottom_left_x = {}
        bottom_left_y = {}
        """.format(self.blur_kernel_size, self.top_left_x, self.top_left_y, self.top_right_x, self.top_right_y,
                   self.bottom_right_x, self.bottom_right_y, self.bottom_left_x, self.bottom_left_y))

    def show(self):
        self.viewer.show()


if __name__ == "__main__":
    RoiViewer('test_images/*.jpg').show()

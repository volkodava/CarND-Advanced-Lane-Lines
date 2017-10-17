from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import Slider, ComboBox, CheckBox

from experiments import *


class ThreshViewer:
    def __init__(self, search_pattern):
        gradx_ksize = 3
        gradx_thresh_low = 20
        gradx_thresh_high = 100
        grady_ksize = 3
        grady_thresh_low = 20
        grady_thresh_high = 100
        mag_binary_ksize = 3
        mag_binary_thresh_low = 30
        mag_binary_thresh_high = 100
        dir_binary_ksize = 15
        dir_binary_thresh_low = 0.7
        dir_binary_thresh_high = 1.3

        plugin = Plugin(image_filter=self.image_filter, dock="right")

        self.setup_names = ['Sobel Thresh X', 'Sobel Thresh Y', 'Sobel Thresh X / Y', 'Magnitude Thresh',
                            'Gradient Direction', 'Magnitude / Gradient Direction Thresh', "2 & 2", "2 | 2"]

        self.show_orig = CheckBox('show_orig', value=False, alignment='left')

        plugin += self.show_orig
        plugin += ComboBox('setup', self.setup_names)
        plugin += Slider('gradx_ksize', 0, 31, value=gradx_ksize, value_type='int')
        plugin += Slider('gradx_thresh_low', 0, 255, value=gradx_thresh_low, value_type='int')
        plugin += Slider('gradx_thresh_high', 0, 255, value=gradx_thresh_high, value_type='int')
        plugin += Slider('grady_ksize', 0, 31, value=grady_ksize, value_type='int')
        plugin += Slider('grady_thresh_low', 0, 255, value=grady_thresh_low, value_type='int')
        plugin += Slider('grady_thresh_high', 0, 255, value=grady_thresh_high, value_type='int')
        plugin += Slider('mag_binary_ksize', 0, 31, value=mag_binary_ksize, value_type='int')
        plugin += Slider('mag_binary_thresh_low', 0, 255, value=mag_binary_thresh_low, value_type='int')
        plugin += Slider('mag_binary_thresh_high', 0, 255, value=mag_binary_thresh_high, value_type='int')
        plugin += Slider('dir_binary_ksize', 0, 31, value=dir_binary_ksize, value_type='int')
        plugin += Slider('dir_binary_thresh_low', 0, np.pi, value=dir_binary_thresh_low)
        plugin += Slider('dir_binary_thresh_high', 0, np.pi, value=dir_binary_thresh_high)

        fnames = [path for path in glob.iglob(search_pattern, recursive=True)]
        images, gray_images = read_images(fnames)

        self.viewer = CollectionViewer(images)
        self.viewer += plugin

    def image_filter(self, image, *args, **kwargs):
        print("image: ", image.shape)

        # use grayscale based on calculated color values
        image_gray = grayscale(image)
        print("gray image: ", image_gray.shape)

        show_orig = kwargs["show_orig"]
        setup = kwargs["setup"]
        gradx_ksize = kwargs["gradx_ksize"]
        gradx_thresh_low = kwargs["gradx_thresh_low"]
        gradx_thresh_high = kwargs["gradx_thresh_high"]
        grady_ksize = kwargs["grady_ksize"]
        grady_thresh_low = kwargs["grady_thresh_low"]
        grady_thresh_high = kwargs["grady_thresh_high"]
        mag_binary_ksize = kwargs["mag_binary_ksize"]
        mag_binary_thresh_low = kwargs["mag_binary_thresh_low"]
        mag_binary_thresh_high = kwargs["mag_binary_thresh_high"]
        dir_binary_ksize = kwargs["dir_binary_ksize"]
        dir_binary_thresh_low = kwargs["dir_binary_thresh_low"]
        dir_binary_thresh_high = kwargs["dir_binary_thresh_high"]

        if show_orig:
            return image

        gradx = abs_sobel_thresh(image_gray, orient='x', sobel_kernel=gradx_ksize,
                                 thresh=(gradx_thresh_low, gradx_thresh_high), rgb2gray=False)
        grady = abs_sobel_thresh(image_gray, orient='y', sobel_kernel=grady_ksize,
                                 thresh=(grady_thresh_low, grady_thresh_high), rgb2gray=False)

        mag_binary = mag_thresh(image_gray, sobel_kernel=mag_binary_ksize,
                                thresh=(mag_binary_thresh_low, mag_binary_thresh_high), rgb2gray=False)
        dir_binary = dir_threshold(image_gray, sobel_kernel=dir_binary_ksize,
                                   thresh=(dir_binary_thresh_low, dir_binary_thresh_high), rgb2gray=False)

        combined_sobel = np.zeros_like(gradx)
        combined_sobel[((gradx == 1) & (grady == 1))] = 1

        combined_magn_grad = np.zeros_like(mag_binary)
        combined_magn_grad[((mag_binary == 1) & (dir_binary == 1))] = 1

        result_image = None
        if setup == "Sobel Thresh X":
            result_image = gradx
        elif setup == "Sobel Thresh Y":
            result_image = grady
        elif setup == "Sobel Thresh X / Y":
            result_image = combined_sobel
        elif setup == "Magnitude Thresh":
            result_image = mag_binary
        elif setup == "Gradient Direction":
            result_image = dir_binary
        elif setup == "Magnitude / Gradient Direction Thresh":
            result_image = combined_magn_grad
        elif setup == "2 & 2":
            result_image = np.zeros_like(combined_sobel)
            result_image[((combined_sobel == 1) & (combined_magn_grad == 1))] = 1
        elif setup == "2 | 2":
            result_image = np.zeros_like(combined_sobel)
            result_image[((combined_sobel == 1) | (combined_magn_grad == 1))] = 1

        return result_image

    def show(self):
        self.viewer.show()


if __name__ == "__main__":
    ThreshViewer('test_images/*.jpg').show()
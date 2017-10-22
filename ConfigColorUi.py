from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import Slider, CheckBox, ComboBox, Button

from experiments import *

lower_yellow_1 = 0
lower_yellow_2 = 0
lower_yellow_3 = 0
upper_yellow_1 = 359
upper_yellow_2 = 359
upper_yellow_3 = 359

lower_white_1 = 0
lower_white_2 = 0
lower_white_3 = 0
upper_white_1 = 359
upper_white_2 = 359
upper_white_3 = 359


class ColorViewer:
    def __init__(self, search_pattern):
        plugin = Plugin(image_filter=self.image_filter, dock="right")

        self.setup_names = ['Yellow', 'White', 'Yellow / White']
        self.color_spaces = ['HSV', 'LAB', 'HLS']

        self.show_orig = CheckBox('show_orig', value=False, alignment='left')

        plugin += self.show_orig
        plugin += ComboBox('setup', self.setup_names)
        plugin += ComboBox('color_space', self.color_spaces)
        plugin += Slider('lower_yellow_1', 0, 359, value=lower_yellow_1, value_type='int')
        plugin += Slider('lower_yellow_2', 0, 359, value=lower_yellow_2, value_type='int')
        plugin += Slider('lower_yellow_3', 0, 359, value=lower_yellow_3, value_type='int')

        plugin += Slider('upper_yellow_1', 0, 359, value=upper_yellow_1, value_type='int')
        plugin += Slider('upper_yellow_2', 0, 359, value=upper_yellow_2, value_type='int')
        plugin += Slider('upper_yellow_3', 0, 359, value=upper_yellow_3, value_type='int')

        plugin += Slider('lower_white_1', 0, 359, value=lower_white_1, value_type='int')
        plugin += Slider('lower_white_2', 0, 359, value=lower_white_2, value_type='int')
        plugin += Slider('lower_white_3', 0, 359, value=lower_white_3, value_type='int')

        plugin += Slider('upper_white_1', 0, 359, value=upper_white_1, value_type='int')
        plugin += Slider('upper_white_2', 0, 359, value=upper_white_2, value_type='int')
        plugin += Slider('upper_white_3', 0, 359, value=upper_white_3, value_type='int')

        plugin += Button("Reset", callback=self.on_reset_click)

        fnames = [path for path in glob.iglob(search_pattern, recursive=True)]
        images, gray_images = read_images(fnames)

        self.viewer = CollectionViewer(images)
        self.viewer.connect_event('button_press_event', self.print_color_range)
        self.viewer += plugin

    def print_color_range(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            self.viewer.status_message(self.format_coord(event.xdata, event.ydata))
        else:
            self.viewer.status_message('')

    def format_coord(self, x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        pixel = self.viewer.image[y, x]

        # TODO: move this value to constant
        range = 20
        lower = np.array([pixel[0] - range, pixel[1] - range, pixel[2] - range])
        upper = np.array([pixel[0] + range, pixel[1] + range, pixel[2] + range])
        self.viewer.image = cv2.inRange(self.viewer.image, lower, upper)

        # TODO: update all sliders here

        print("""
lower_yellow_1 = {}
lower_yellow_2 = {}
lower_yellow_3 = {}
upper_yellow_1 = {}
upper_yellow_2 = {}
upper_yellow_3 = {}
        """.format(lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]))

    def image_filter(self, image, *args, **kwargs):
        print("image: ", image.shape)

        image = apply_crop_bottom(image)
        print("cropped image: ", image.shape)

        image, M, Minv = apply_warp(image)
        print("warped image: ", image.shape)

        show_orig = kwargs["show_orig"]
        setup = kwargs["setup"]
        color_space = kwargs["color_space"]
        lower_yellow_1 = kwargs["lower_yellow_1"]
        lower_yellow_2 = kwargs["lower_yellow_2"]
        lower_yellow_3 = kwargs["lower_yellow_3"]
        upper_yellow_1 = kwargs["upper_yellow_1"]
        upper_yellow_2 = kwargs["upper_yellow_2"]
        upper_yellow_3 = kwargs["upper_yellow_3"]

        lower_white_1 = kwargs["lower_white_1"]
        lower_white_2 = kwargs["lower_white_2"]
        lower_white_3 = kwargs["lower_white_3"]
        upper_white_1 = kwargs["upper_white_1"]
        upper_white_2 = kwargs["upper_white_2"]
        upper_white_3 = kwargs["upper_white_3"]

        if show_orig:
            return image

        lower_yellow = np.array([lower_yellow_1, lower_yellow_2, lower_yellow_3])
        upper_yellow = np.array([upper_yellow_1, upper_yellow_2, upper_yellow_3])

        lower_white = np.array([lower_white_1, lower_white_2, lower_white_3])
        upper_white = np.array([upper_white_1, upper_white_2, upper_white_3])

        target_color_space = cv2.COLOR_RGB2HSV
        if color_space == "LAB":
            target_color_space = cv2.COLOR_RGB2Lab
        elif color_space == "HLS":
            target_color_space = cv2.COLOR_RGB2HLS

        result_image = None
        if setup == "Yellow":
            result_image = filter_color(image, lower_yellow, upper_yellow, trg_color_space=target_color_space)
        elif setup == "White":
            result_image = filter_color(image, lower_white, upper_white, trg_color_space=target_color_space)
        elif setup == "Yellow / White":
            result_yellow = filter_color(image, lower_yellow, upper_yellow, trg_color_space=target_color_space)
            result_white = filter_color(image, lower_white, upper_white, trg_color_space=target_color_space)
            result_image = cv2.bitwise_or(result_yellow, result_white)

        return result_image

    def on_reset_click(self, args):
        # TODO: update all sliders to its default state
        pass

    def show(self):
        self.viewer.show()


if __name__ == "__main__":
    ColorViewer('test_images/*.jpg').show()

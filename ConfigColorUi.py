from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import Slider, CheckBox, ComboBox

from experiments import *

sensity_range = 20

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
        self.plugin = Plugin(image_filter=self.image_filter, dock="right")

        self.setup_names = ['Yellow', 'White', 'Yellow / White']
        self.color_spaces = ['HSV', 'LAB', 'HLS', 'LUV']

        self.show_orig = CheckBox('show_orig', value=False, alignment='left')

        self.plugin += self.show_orig
        self.setup = ComboBox('setup', self.setup_names)
        self.color_space = ComboBox('color_space', self.color_spaces)
        self.sensity_range = Slider('sensity_range', 10, 100, value=sensity_range, value_type='int')
        self.lower_yellow_1 = Slider('lower_yellow_1', 0, 359, value=lower_yellow_1, value_type='int')
        self.lower_yellow_2 = Slider('lower_yellow_2', 0, 359, value=lower_yellow_2, value_type='int')
        self.lower_yellow_3 = Slider('lower_yellow_3', 0, 359, value=lower_yellow_3, value_type='int')
        self.upper_yellow_1 = Slider('upper_yellow_1', 0, 359, value=upper_yellow_1, value_type='int')
        self.upper_yellow_2 = Slider('upper_yellow_2', 0, 359, value=upper_yellow_2, value_type='int')
        self.upper_yellow_3 = Slider('upper_yellow_3', 0, 359, value=upper_yellow_3, value_type='int')

        self.lower_white_1 = Slider('lower_white_1', 0, 359, value=lower_white_1, value_type='int')
        self.lower_white_2 = Slider('lower_white_2', 0, 359, value=lower_white_2, value_type='int')
        self.lower_white_3 = Slider('lower_white_3', 0, 359, value=lower_white_3, value_type='int')
        self.upper_white_1 = Slider('upper_white_1', 0, 359, value=upper_white_1, value_type='int')
        self.upper_white_2 = Slider('upper_white_2', 0, 359, value=upper_white_2, value_type='int')
        self.upper_white_3 = Slider('upper_white_3', 0, 359, value=upper_white_3, value_type='int')

        self.plugin += self.setup
        self.plugin += self.color_space
        self.plugin += self.sensity_range
        self.plugin += self.lower_yellow_1
        self.plugin += self.lower_yellow_2
        self.plugin += self.lower_yellow_3
        self.plugin += self.upper_yellow_1
        self.plugin += self.upper_yellow_2
        self.plugin += self.upper_yellow_3

        self.plugin += self.lower_white_1
        self.plugin += self.lower_white_2
        self.plugin += self.lower_white_3
        self.plugin += self.upper_white_1
        self.plugin += self.upper_white_2
        self.plugin += self.upper_white_3

        fnames = [path for path in glob.iglob(search_pattern, recursive=True)]
        images, gray_images = read_images(fnames)

        self.viewer = CollectionViewer(images)
        self.viewer.connect_event('button_press_event', self.on_filter_color)
        self.viewer.connect_event('key_press_event', self.on_press)
        self.viewer += self.plugin

    def on_filter_color(self, event):
        if event.inaxes and event.inaxes.get_navigate():
            self.viewer.status_message(self.format_coord(event.xdata, event.ydata))
        else:
            self.viewer.status_message('')

    def format_coord(self, x, y):
        x = int(x + 0.5)
        y = int(y + 0.5)
        pixel = self.viewer.image[y, x]

        lower_lst = [pixel[0] - self.sensity_range.val, pixel[1] - self.sensity_range.val,
                     pixel[2] - self.sensity_range.val]
        upper_lst = [pixel[0] + self.sensity_range.val, pixel[1] + self.sensity_range.val,
                     pixel[2] + self.sensity_range.val]

        setup = self.setup.val
        if setup == "Yellow":
            self.update_yellow_params(*lower_lst, *upper_lst)
        elif setup == "White":
            self.update_white_params(*lower_lst, *upper_lst)
        else:
            print("Select only one color!")

    def update_yellow_params(self, lower1, lower2, lower3, upper1, upper2, upper3):
        lower1 = self.update_val(self.lower_yellow_1, lower1)
        lower2 = self.update_val(self.lower_yellow_2, lower2)
        lower3 = self.update_val(self.lower_yellow_3, lower3)
        upper1 = self.update_val(self.upper_yellow_1, upper1)
        upper2 = self.update_val(self.upper_yellow_2, upper2)
        upper3 = self.update_val(self.upper_yellow_3, upper3)
        self.plugin.filter_image()

    def update_white_params(self, lower1, lower2, lower3, upper1, upper2, upper3):
        lower1 = self.update_val(self.lower_white_1, lower1)
        lower2 = self.update_val(self.lower_white_2, lower2)
        lower3 = self.update_val(self.lower_white_3, lower3)
        upper1 = self.update_val(self.upper_white_1, upper1)
        upper2 = self.update_val(self.upper_white_2, upper2)
        upper3 = self.update_val(self.upper_white_3, upper3)
        self.plugin.filter_image()

    def image_filter(self, image, *args, **kwargs):
        print("image: ", image.shape)

        image = apply_crop_bottom(image)
        # print("cropped image: ", image.shape)

        image, M, Minv = apply_warp(image)
        # print("warped image: ", image.shape)

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
        elif color_space == "LUV":
            target_color_space = cv2.COLOR_RGB2LUV

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

    def on_press(self, event):
        if event.key == 'ctrl+r':
            self.on_reset()
        elif event.key == 'ctrl+p':
            self.on_print()

    def on_print(self, args=None):
        print("""
        lower_yellow_1 = {}
        lower_yellow_2 = {}
        lower_yellow_3 = {}
        upper_yellow_1 = {}
        upper_yellow_2 = {}
        upper_yellow_3 = {}

        lower_white_1 = {}
        lower_white_2 = {}
        lower_white_3 = {}
        upper_white_1 = {}
        upper_white_2 = {}
        upper_white_3 = {}
                    """.format(self.lower_yellow_1.val, self.lower_yellow_2.val, self.lower_yellow_3.val,
                               self.upper_yellow_1.val, self.upper_yellow_2.val, self.upper_yellow_3.val,
                               self.lower_white_1.val, self.lower_white_2.val, self.lower_white_3.val,
                               self.upper_white_1.val, self.upper_white_2.val, self.upper_white_3.val
                               ))

    def on_reset(self, args=None):
        self.update_val(self.lower_yellow_1, lower_yellow_1)
        self.update_val(self.lower_yellow_2, lower_yellow_2)
        self.update_val(self.lower_yellow_3, lower_yellow_3)
        self.update_val(self.upper_yellow_1, upper_yellow_1)
        self.update_val(self.upper_yellow_2, upper_yellow_2)
        self.update_val(self.upper_yellow_3, upper_yellow_3)

        self.update_val(self.lower_white_1, lower_white_1)
        self.update_val(self.lower_white_2, lower_white_2)
        self.update_val(self.lower_white_3, lower_white_3)
        self.update_val(self.upper_white_1, upper_white_1)
        self.update_val(self.upper_white_2, upper_white_2)
        self.update_val(self.upper_white_3, upper_white_3)

        self.plugin.filter_image()

    def update_val(self, comp, newval, min_val=0, max_val=359):
        newval = max(0, newval)
        newval = min(359, newval)

        comp.val = newval
        comp.editbox.setText("%s" % newval)

        return newval

    def show(self):
        self.viewer.show()


if __name__ == "__main__":
    ColorViewer('test_images/*.jpg').show()

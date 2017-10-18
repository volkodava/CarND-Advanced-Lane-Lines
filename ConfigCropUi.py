from skimage.viewer import CollectionViewer
from skimage.viewer.plugins import Plugin
from skimage.viewer.widgets import Slider, CheckBox, ComboBox, Button

from experiments import *


class CropViewer:
    def __init__(self, search_pattern):
        bottom_px = 60

        self.bottom_px = bottom_px

        plugin = Plugin(image_filter=self.image_filter, dock="right")

        self.setup_names = ['Bottom']

        self.show_orig = CheckBox('show_orig', value=False, alignment='left')

        plugin += self.show_orig
        plugin += ComboBox('setup', self.setup_names)
        plugin += Slider('bottom_px', 0, 500, value=self.bottom_px, value_type='int')
        plugin += Button("Print", callback=self.on_print_click)

        fnames = [path for path in glob.iglob(search_pattern, recursive=True)]
        images, gray_images = read_images(fnames)

        self.viewer = CollectionViewer(images)
        self.viewer += plugin

    def image_filter(self, image, *args, **kwargs):
        print("image: ", image.shape)

        show_orig = kwargs["show_orig"]
        setup = kwargs["setup"]
        self.bottom_px = kwargs["bottom_px"]

        if show_orig:
            return image

        # crop_img = img[200:400, 100:300]  # Crop from x, y, w, h -> 100, 200, 300, 400
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        result_image = None
        if setup == "Bottom":
            result_image = apply_crop_bottom(image, bottom_px=self.bottom_px)

        return result_image

    def on_print_click(self, args):
        print("""
        bottom_px = {}
        """.format(self.bottom_px))

    def show(self):
        self.viewer.show()


if __name__ == "__main__":
    CropViewer('test_images/*.jpg').show()

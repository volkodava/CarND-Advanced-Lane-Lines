import glob
import ntpath
import os
import tkinter as tk
from collections import OrderedDict
from tkinter import Canvas, Frame, Tk, Scale, HORIZONTAL, Image, IntVar, DoubleVar

import cv2
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageTk, ImageDraw


class ImageToolUi(Tk):
    fileTypes = [('JPG files', '*.jpg'), ('All files', '*')]
    images_path = glob.glob('test_images/*.jpg', recursive=True)
    kernelOptions = [i for i in range(1, 50, 2)]
    color_spaces = OrderedDict({"RGB": None, "HSL": cv2.COLOR_RGB2HLS, "HSV": cv2.COLOR_RGB2HSV})

    def __init__(self):
        Tk.__init__(self)

        self.img_path = None
        self.orig_img = None

        self.copy_img = None
        self.filtered_img = None
        self.gray_img = None
        self.blur_gray_img = None
        self.edges_img = None
        self.masked_edges_img = None
        self.line_img = None
        self.color_edges_img = None
        self.lines_edges_img = None
        self.canvas_img = None

        self.ysize = None
        self.xsize = None

        self.top_left = None
        self.top_right = None
        self.bottom_right = None
        self.bottom_left = None

        # INPUT PARAMETERS
        default_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images", "test4.jpg")

        self.kernel_size = 15
        self.low_threshold = 50
        self.high_threshold = 150
        # TO DEFINE PARAMETERS
        # distance resolution in pixels of the Hough grid
        self.rho = 2
        # angular resolution in radians of the Hough grid
        self.selected_theta_degree = 1
        self.theta = self.selected_theta_degree * np.pi / 180
        # minimum number of votes (intersections in Hough grid cell)
        self.threshold = 20
        # minimum number of pixels making up a line
        self.min_line_length = 7
        # maximum gap in pixels between connectable line segments
        self.max_line_gap = 200

        # create ui
        colorFrame = Frame(self, bd=2)
        self.lower_color_first = 0
        self.lower_color_second = 70
        self.lower_color_third = 100

        self.upper_color_first = 30
        self.upper_color_second = 255
        self.upper_color_third = 255

        img_path_var = tk.StringVar(self)
        img_path_var.set(self.images_path[0])
        tk.OptionMenu(colorFrame, img_path_var,
                      *self.images_path, command=self.on_img_path_select).pack(side='left')

        options = ['Final', 'Original', 'Color Filtered', 'Gray', 'Blur', 'Edges',
                   'Masked Edges', 'Line', 'Color Edges']
        self.default_img_config_var = tk.StringVar(self)
        self.default_img_config_var.set(options[0])
        tk.OptionMenu(colorFrame, self.default_img_config_var,
                      *options, command=self.on_img_config_select).pack(side='left')

        clr_options = list(self.color_spaces.keys())
        self.default_clr_config_var = tk.StringVar(self)
        self.default_clr_config_var.set(clr_options[0])
        tk.OptionMenu(colorFrame, self.default_clr_config_var,
                      *clr_options, command=self.on_clr_config_select).pack(side='left')

        self.show_orig_var = tk.BooleanVar()
        self.show_orig_var.set(False)
        tk.Checkbutton(
            colorFrame, text="Show Orig",
            variable=self.show_orig_var,
            onvalue=True, offvalue=False,
            command=self.on_show_orig).pack(side='left')

        self.build_scale_menu(colorFrame, self.lower_color_first, 'Lower First', self.on_lower_color_first_changed,
                              frm=0, to=255)
        self.build_scale_menu(colorFrame, self.lower_color_second, 'Lower Second', self.on_lower_color_second_changed,
                              frm=0, to=255)
        self.build_scale_menu(colorFrame, self.lower_color_third, 'Lower Third', self.on_lower_color_third_changed,
                              frm=0, to=255)

        self.build_scale_menu(colorFrame, self.upper_color_first, 'Upper First', self.on_upper_color_first_changed,
                              frm=0, to=255)
        self.build_scale_menu(colorFrame, self.upper_color_second, 'Upper Second', self.on_upper_color_second_changed,
                              frm=0, to=255)
        self.build_scale_menu(colorFrame, self.upper_color_third, 'Upper Third', self.on_upper_color_third_changed,
                              frm=0, to=255)

        colorFrame.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)

        edgeDetectionFrame = Frame(self, bd=2)
        self.build_scale_menu(edgeDetectionFrame, self.rho, 'Rho', self.on_rho_changed)
        self.build_scale_menu(edgeDetectionFrame, self.selected_theta_degree, 'Theta (degree)',
                              self.on_theta_degree_changed,
                              frm=1, to=90)
        self.build_scale_menu(edgeDetectionFrame, self.threshold, 'Threshold (votes)', self.on_threshold_changed)
        self.build_scale_menu(edgeDetectionFrame, self.min_line_length, 'Min Length', self.on_min_length_changed)
        self.build_scale_menu(edgeDetectionFrame, self.max_line_gap, 'Max Gap', self.on_max_gap_changed)

        self.file_size_var = tk.StringVar()
        self.file_size_var.set("W x H")
        tk.Label(edgeDetectionFrame, textvariable=self.file_size_var).pack(side='right')

        self.file_name_var = tk.StringVar()
        self.file_name_var.set(self.get_filename(default_file_path))
        tk.Label(edgeDetectionFrame, textvariable=self.file_name_var).pack(side='right')
        edgeDetectionFrame.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)

        roiFrame = Frame(self, bd=2)
        self.top_left_x = 0.4
        self.top_left_y = 0.6
        self.bottom_left_x = 0.1
        self.bottom_left_y = 1.0
        self.top_right_x = 0.6
        self.top_right_y = 0.6
        self.bottom_right_x = 0.95
        self.bottom_right_y = 1.0
        self.build_scale_menu_double(roiFrame, self.top_left_x, 'X Top Left', self.on_top_left_x_changed)
        self.build_scale_menu_double(roiFrame, self.top_left_y, 'Y Top Left', self.on_top_left_y_changed)
        self.build_scale_menu_double(roiFrame, self.bottom_left_x, 'X Bottom Left', self.on_bottom_left_x_changed)
        self.build_scale_menu_double(roiFrame, self.bottom_left_y, 'Y Bottom Left', self.on_bottom_left_y_changed)
        self.build_scale_menu_double(roiFrame, self.top_right_x, 'X Top Right', self.on_top_right_x_changed)
        self.build_scale_menu_double(roiFrame, self.top_right_y, 'Y Top Right', self.on_top_right_y_changed)
        self.build_scale_menu_double(roiFrame, self.bottom_right_x, 'X Bottom Right', self.on_bottom_right_x_changed)
        self.build_scale_menu_double(roiFrame, self.bottom_right_y, 'Y Bottom Right', self.on_bottom_right_y_changed)
        roiFrame.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)

        self.canvas = Canvas(self, bd=0, highlightthickness=0,
                             width=800, height=800, scrollregion=(0, 0, 800, 800))
        x_scroll = tk.Scrollbar(self, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")
        x_scroll.config(command=self.canvas.xview)
        y_scroll = tk.Scrollbar(self, orient="vertical")
        y_scroll.pack(side="right", fill="y")
        y_scroll.config(command=self.canvas.yview)
        self.canvas.config(
            xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)

        self.load_image(default_file_path)

    def build_scale_menu(self, frame, default_val, label, clbk, frm=1, to=500):
        max_gap_var = IntVar()
        max_gap_var.set(default_val)
        Scale(frame, label=label, orient=HORIZONTAL, from_=frm, to=to,
              variable=max_gap_var, command=clbk).pack(side='left')

    def build_scale_menu_double(self, frame, default_val, label, clbk, frm=0.01, to=2.99, resolution=0.01):
        max_gap_var = DoubleVar()
        max_gap_var.set(default_val)
        Scale(frame, label=label, orient=HORIZONTAL, from_=frm, to=to,
              variable=max_gap_var, command=clbk, resolution=resolution).pack(side='left')

    def on_img_path_select(self, value):
        print("on_img_path_select")
        self.img_path = None
        self.orig_img = None

        self.copy_img = None
        self.filtered_img = None
        self.gray_img = None
        self.blur_gray_img = None
        self.edges_img = None
        self.masked_edges_img = None
        self.line_img = None
        self.color_edges_img = None
        self.lines_edges_img = None
        self.canvas_img = None

        self.ysize = None
        self.xsize = None

        self.file_size_var.set("W x H")
        self.file_name_var.set("Not selected.")

        self.load_image(value)
        self.apply_changes()

    def on_clr_config_select(self, value):
        print("on_clr_config_select")
        self.apply_changes()

    def on_img_config_select(self, value):
        print("on_img_config_select")
        self.handle_img_rendering()

    def handle_img_rendering(self):
        print("handle_img_rendering")
        if self.lines_edges_img is None:
            print("No Results. Use Original image.")
            self.default_img_config_var.set('Original')
            self.update_img(np.copy(self.orig_img))
            return

        value = self.default_img_config_var.get()

        if value == 'Final':
            self.update_img(np.copy(self.lines_edges_img))
            self.show_orig_var.set(False)
        elif value == 'Original':
            self.update_img(np.copy(self.orig_img))
            self.show_orig_var.set(True)
        elif value == 'Color Filtered':
            self.update_img(np.copy(self.filtered_img))
            self.show_orig_var.set(False)
        elif value == 'Gray':
            self.update_img(np.copy(self.gray_img))
            self.show_orig_var.set(False)
        elif value == 'Blur':
            self.update_img(np.copy(self.blur_gray_img))
            self.show_orig_var.set(False)
        elif value == 'Edges':
            self.update_img(np.copy(self.edges_img))
            self.show_orig_var.set(False)
        elif value == 'Masked Edges':
            self.update_img(np.copy(self.masked_edges_img))
            self.show_orig_var.set(False)
        elif value == 'Line':
            self.update_img(np.copy(self.line_img))
            self.show_orig_var.set(False)
        elif value == 'Color Edges':
            self.update_img(np.copy(self.color_edges_img))
            self.show_orig_var.set(False)

    def load_image(self, path):
        print("load_image")
        self.img_path = path
        self.orig_img = mpimg.imread(path)
        self.ysize = self.orig_img.shape[0]
        self.xsize = self.orig_img.shape[1]

        self.file_size_var.set("%s x %s" % (self.xsize, self.ysize))
        self.file_name_var.set(self.get_filename(self.img_path))

        self.top_left = [self.xsize * self.top_left_x, self.ysize * self.top_left_y]
        self.top_right = [self.xsize * self.top_right_x, self.ysize * self.top_right_y]
        self.bottom_right = [self.xsize * self.bottom_right_x, self.ysize * self.bottom_right_y]
        self.bottom_left = [self.xsize * self.bottom_left_x, self.ysize * self.bottom_left_y]
        self.handle_img_rendering()

    def on_show_orig(self):
        print("on_show_orig")
        if self.show_orig_var.get():
            self.default_img_config_var.set('Original')
            self.handle_img_rendering()
        else:
            self.default_img_config_var.set('Final')
            self.handle_img_rendering()

    def on_lower_color_first_changed(self, value):
        print("on_lower_color_first_changed")
        self.lower_color_first = int(value)

        if self.copy_img is not None:
            print("Set W color_first to: ", self.lower_color_first)
            self.apply_changes()

    def on_lower_color_second_changed(self, value):
        print("on_lower_color_second_changed")
        self.lower_color_second = int(value)

        if self.copy_img is not None:
            print("Set W color_second to: ", self.lower_color_second)
            self.apply_changes()

    def on_lower_color_third_changed(self, value):
        print("on_lower_color_third_changed")
        self.lower_color_third = int(value)

        if self.copy_img is not None:
            print("Set W color_third to: ", self.lower_color_third)
            self.apply_changes()

    def on_upper_color_first_changed(self, value):
        print("on_upper_color_first_changed")
        self.upper_color_first = int(value)

        if self.copy_img is not None:
            print("Set Y color_first to: ", self.upper_color_first)
            self.apply_changes()

    def on_upper_color_second_changed(self, value):
        print("on_upper_color_second_changed")
        self.upper_color_second = int(value)

        if self.copy_img is not None:
            print("Set Y color_second to: ", self.upper_color_second)
            self.apply_changes()

    def on_upper_color_third_changed(self, value):
        print("on_upper_color_third_changed")
        self.upper_color_third = int(value)

        if self.copy_img is not None:
            print("Set Y color_third to: ", self.upper_color_third)
            self.apply_changes()

    def on_rho_changed(self, value):
        print("on_rho_changed")
        self.rho = int(value)

        if self.copy_img is not None:
            print("Set rho to: ", self.rho)
            self.apply_changes()

    def on_theta_degree_changed(self, value):
        print("on_theta_degree_changed")
        self.selected_theta_degree = int(value)
        self.theta = self.selected_theta_degree * np.pi / 180

        if self.copy_img is not None:
            print("Set selected_degree to: ", self.selected_theta_degree)
            print("Set theta to: ", self.theta)
            self.apply_changes()

    def on_threshold_changed(self, value):
        print("on_threshold_changed")
        self.threshold = int(value)

        if self.copy_img is not None:
            print("Set threshold (votes) to: ", self.threshold)
            self.apply_changes()

    def on_min_length_changed(self, value):
        print("on_min_length_changed")
        self.min_line_length = int(value)

        if self.copy_img is not None:
            print("Set min_line_length to: ", self.min_line_length)
            self.apply_changes()

    def on_max_gap_changed(self, value):
        print("on_max_gap_changed")
        self.max_line_gap = int(value)

        if self.copy_img is not None:
            print("Set max_line_gap to: ", self.max_line_gap)
            self.apply_changes()

    def on_top_left_x_changed(self, value):
        print("on_top_left_x_changed")
        self.top_left_x = float(value)
        self.change_x_roi(self.top_left, self.top_left_x)

    def on_top_left_y_changed(self, value):
        print("on_top_left_y_changed")
        self.top_left_y = float(value)
        self.change_y_roi(self.top_left, self.top_left_y)

    def on_top_right_x_changed(self, value):
        print("on_top_right_x_changed")
        self.top_right_x = float(value)
        self.change_x_roi(self.top_right, self.top_right_x)

    def on_top_right_y_changed(self, value):
        print("on_top_right_y_changed")
        self.top_right_y = float(value)
        self.change_y_roi(self.top_right, self.top_right_y)

    def on_bottom_right_x_changed(self, value):
        print("on_bottom_right_x_changed")
        self.bottom_right_x = float(value)
        self.change_x_roi(self.bottom_right, self.bottom_right_x)

    def on_bottom_right_y_changed(self, value):
        print("on_bottom_right_y_changed")
        self.bottom_right_y = float(value)
        self.change_y_roi(self.bottom_right, self.bottom_right_y)

    def on_bottom_left_x_changed(self, value):
        print("on_bottom_left_x_changed")
        self.bottom_left_x = float(value)
        self.change_x_roi(self.bottom_left, self.bottom_left_x)

    def on_bottom_left_y_changed(self, value):
        print("on_bottom_left_y_changed")
        self.bottom_left_y = float(value)
        self.change_y_roi(self.bottom_left, self.bottom_left_y)

    def change_x_roi(self, param, value):
        print("change_x_roi")
        if self.copy_img is not None:
            param[0] = self.xsize * float(value)
            self.apply_changes()

    def change_y_roi(self, param, value):
        print("change_y_roi")
        if self.copy_img is not None:
            param[1] = self.ysize * float(value)
            self.apply_changes()

    def on_mouse_moved(self, event):
        self.canvas.itemconfigure(self.canvas_tag, text="(%r, %r)" % (event.x, event.y))

    def update_canvas(self, im):
        print("update_canvas")
        draw = ImageDraw.Draw(im)
        draw.polygon([*self.top_left, *self.top_right, *self.bottom_right, *self.bottom_left], outline='white')
        self.tkphoto = ImageTk.PhotoImage(im)
        self.canvasItem = self.canvas.create_image(0, 0, anchor='nw', image=self.tkphoto)
        self.canvas.config(width=im.size[0], height=im.size[1])
        self.canvas.bind("<Motion>", func=self.on_mouse_moved)
        self.canvas_tag = self.canvas.create_text(10, 10, text="", anchor="nw", fill="yellow")
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def filter_colors(self, img):
        print("filter_colors")

        color_space_key = self.default_clr_config_var.get()
        print("Selected color: ", color_space_key)

        color_space = self.color_spaces.get(color_space_key)

        in_img = img
        if color_space is not None:
            # Convert rgb
            in_img = cv2.cvtColor(in_img, color_space)

        # define range of white color in hls
        lower = np.array([self.lower_color_first, self.lower_color_second, self.lower_color_third])
        upper = np.array([self.upper_color_first, self.upper_color_second, self.upper_color_third])
        mask = cv2.inRange(in_img, lower, upper)

        return cv2.bitwise_and(img, img, mask=mask)

    def apply_changes(self):
        print("apply_changes")
        cpy_im = np.copy(self.orig_img)

        self.filtered_img = self.filter_colors(cpy_im)
        self.gray_img = cv2.cvtColor(self.filtered_img, cv2.COLOR_RGB2GRAY)
        self.blur_gray_img = cv2.GaussianBlur(self.gray_img, (self.kernel_size, self.kernel_size), 0)
        self.edges_img = cv2.Canny(self.blur_gray_img, self.low_threshold, self.high_threshold)

        # Next we'll create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(self.edges_img)
        ignore_mask_color = 255

        # This time we are defining a four sided polygon to mask
        vertices = np.array([[self.top_left, self.top_right, self.bottom_right, self.bottom_left]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        self.masked_edges_img = cv2.bitwise_and(self.edges_img, mask)

        # creating a blank to draw lines on
        self.line_img = np.zeros_like(self.copy_img)

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines_img = cv2.HoughLinesP(self.masked_edges_img, self.rho, self.theta, self.threshold, np.array([]),
                                    self.min_line_length, self.max_line_gap)

        if lines_img is None:
            self.handle_img_rendering()
            return

            # Iterate over the output "lines" and draw lines on a blank image
        for line in lines_img:
            for x1, y1, x2, y2 in line:
                cv2.line(self.line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

        # Create a "color" binary image to combine with line image
        self.color_edges_img = np.dstack((self.edges_img, self.edges_img, self.edges_img))

        # Draw the lines on the edge image
        if len(self.color_edges_img.shape) != len(self.line_img.shape):
            print(
                "Error: Color image channels != Line image channels. Skipping further transformation and refresh image.")
            self.refresh_img()
            return

        # draw the lines on the edge image
        self.lines_edges_img = cv2.addWeighted(self.color_edges_img, 0.8, self.line_img, 1, 0)
        # path, file = os.path.split(self.img_path)
        # new_file_path = os.path.join(path, "tmp_" + file)
        # mpimg.imsave(new_file_path, self.lines_edges_img)
        # print("Processed image stored: ", new_file_path)
        self.handle_img_rendering()

    def refresh_img(self):
        print("refresh_img")
        self.load_image(self.img_path)

    def update_img(self, copy_img):
        print("update_img")
        self.copy_img = copy_img
        self.canvas_img = Image.fromarray(self.copy_img)
        self.update_canvas(self.canvas_img)

    def get_filename(self, path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)


if __name__ == "__main__":
    app = ImageToolUi()
    app.mainloop()

# coding=utf-8
import Tkinter as tk
from PIL import Image, ImageTk
import glob
import os
# import mouse
import multiprocessing as mp
import time


class App(tk.Tk):
    def __init__(self, folder, expected_image_count, w, h):
        tk.Tk.__init__(self)
        self.geometry('+{}+{}'.format(w, h))
        # self.pictures = cycle((ImageTk.PhotoImage(file=image), image) for image in image_files)
        self.folder = folder
        self.expected_image_count = expected_image_count
        self.image_display_w = w
        self.image_display_h = h
        self.load_dir()
        self.loaded_image = {}
        self.load_image()
        self.ind = ''
        self.image_display = tk.Label(self, bg='black')
        self.image_display.pack()
        self.next()

    def load_dir(self):
        self.images_path = glob.glob(os.path.expanduser(self.folder + '*.JPG'))
        if len(self.images_path) < self.expected_image_count:
            self.after(100, self.load_more(self.folder, self.expected_image_count))

    def load_image(self, specific_image_path=None):
        if specific_image_path is None:
            for k, image_path in enumerate(self.images_path):
                if image_path not in self.loaded_image:
                    image_pil = Image.open(image_path).resize((self.image_display_w, self.image_display_h))
                    self.loaded_image[image_path] = ImageTk.PhotoImage(image_pil)
                    if len(self.loaded_image) < self.expected_image_count:
                        self.after(500 if k==1 else 100, self.load_image)
                    return
        else:
            image_pil = Image.open(specific_image_path).resize((self.image_display_w, self.image_display_h))
            self.loaded_image[specific_image_path] = image = ImageTk.PhotoImage(image_pil)
            return image

    def next(self):
        if self.ind == '':
            self.ind = 0
        else:
            self.ind = (self.ind + 1) % self.expected_image_count
        self.show_image()
        # self.after(1000, self.next)

    def prior(self):
        if self.ind == '':
            self.ind = self.expected_image_count - 1
        else:
            self.ind = (self.ind + self.expected_image_count - 1) % self.expected_image_count
        self.show_image()

    def show_image(self):
        try:
            image_path = self.images_path[self.ind]
        except:
            print('No path for index: %d' % self.ind)
            return

        if image_path in self.loaded_image:
            image = self.loaded_image[image_path]
        else:
            image = self.load_image(image_path)

        self.image_display.config(image=image)
        self.title(image_path)

    def run(self):
        self.after(1000, self.next)
        self.after(2000, self.next)
        self.after(3000, self.next)
        self.mainloop()

# def test():
#     x = 200
#     y = 150
#     app = App('~/Documents/Photos/沙巴行-sony-T2/', 3, x, y)
#     app.mainloop()
#
# test()
x = 200
y = 150
app = App('~/Documents/Photos/沙巴行-sony-T2/', 3, x, y)
app.run()
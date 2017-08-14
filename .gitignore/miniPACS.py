# coding=utf-8
import Tkinter as tk
from PIL import Image, ImageTk
import glob
import multiprocessing as mp
import os

import Tkinter as tk
from PIL import Image, ImageTk
from itertools import cycle

class App(tk.Tk):

    def __init__(self, image_files, x, y):
        tk.Tk.__init__(self)
        self.geometry('+{}+{}'.format(x,y))
        #self.pictures = cycle((ImageTk.PhotoImage(file=image), image) for image in image_files)
        self.pictures = image_files
        self.ind=''
        self.picture_display = tk.Label(self)
        self.picture_display.pack()
        self.images = [] # to keep references to images.

    def next(self):
        if self.ind=='':
            self.ind=1
        else:
            self.ind= self.ind+1 % len(self.pictures)
        return self.pictures(self.ind)

    def prior(self):
        if self.ind == '':
            self.ind = len(self.pictures)
        else:
            self.ind = self.ind+len(self.pictures) -1 % len(self.pictures)
        return self.pictures(self.ind)

    def next_image(self, prior=False):
        if not prior:
            img_name = self.next()
        else
            img_name = prior(self.pictures)
        image_pil = Image.open(img_name).resize((300, 300)) #<-- resize images here

        self.images.append(ImageTk.PhotoImage(image_pil))

        self.picture_display.config(image=self.images[-1])
        self.title(img_name)

    def run(self):
        self.mainloop()

delay = 500

x = 200
y = 150
app = App(glob.glob(os.path.expanduser('~/Documents/Photos/沙巴行-sony-T2/*.JPG')),x,y,delay)
app.show_slides()
app.run()



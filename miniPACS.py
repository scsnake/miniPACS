# -*- coding: utf-8 -*-

import ctypes.wintypes
import glob
import inspect
import json
import logging
import os
import statistics as stat
import sys
import threading
from collections import OrderedDict
from contextlib import suppress
from functools import partial
from time import sleep, clock

import cv2
import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
# from PyQt4.QtCore import Qt, SIGNAL
# from PyQt4.QtGui import QApplication, QMainWindow, QTextEdit, QMessageBox, QLabel, QImage
# from PyQt4.QtGui import QPixmap, QDesktopWidget, QFont
# from PyQt4.QtGui import QWidget, QSizePolicy
from screeninfo import get_monitors

from win32func import WM_COPYDATA_Listener, Send_WM_COPYDATA


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    with suppress(Exception):
        _async_raise(thread.ident, SystemExit)

def _get_monitors():
    global app
    d = app.desktop()
    return [d.screenGeometry(i) for i in range(d.screenCount())]


class ImageLabel(QLabel):
    def __init__(self, *args):
        super(ImageLabel, self).__init__(*args)
        self.mousePressed = False
        self.mouseMoving = False
        self.zooming_mode = False

        mag_label = QLabel(self)
        mag_label.setFixedSize(400, 400)
        mag_label.setStyleSheet('border: 1px solid white;')
        mag_label.hide()
        self.mag_label = mag_label
        self.cal_th = None
        self.cal_th_ev = threading.Event()
        self.calculating = False
        self.is_sharp_image = False
        # self.setMouseTracking(True)

        self.connect(self, SIGNAL('set_pixmap'), self.setPixmap)
        self.connect(self, SIGNAL('set_pixmap_qimage'), self.set_pixmap_qimage)
        self.connect(self, SIGNAL('sharp_image'), self.sharp_image)

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton or QMouseEvent.button() == Qt.RightButton:
            self.mousePressed = QMouseEvent.button()
            self.mousePressedPosX = QMouseEvent.x()
            self.mousePressedPosY = QMouseEvent.y()
        elif QMouseEvent.button() == Qt.MiddleButton:
            self.is_sharp_image = not self.is_sharp_image
            self.emit(SIGNAL('sharp_image'), self.is_sharp_image)

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton or QMouseEvent.button() == Qt.RightButton:
            try:
                self.cal_th_ev.set()
                self.cal_th.join()
                self.original_image_processing_tm.cancel()
            except:
                pass
            finally:
                # self.setPixmap(self.original_px)
                self.emit(SIGNAL('set_pixmap'), self.base_px)
                self.mousePressed = False

    def mouseMoveEvent(self, QMouseEvent):
        if not self.mousePressed:
            return

        if not self.cal_th or not self.cal_th.isAlive():
            if self.mousePressed == Qt.LeftButton:
                target = self.cal_brightness_contrast
            else:
                target = self.cal_zoomin
            self.cal_th_ev.clear()
            th = threading.Thread(target=target, args=(QMouseEvent.x(), QMouseEvent.y()))
            th.start()
            self.cal_th = th

    # def show_mag(self):

    def sharp_image(self, do_sharpen=True):

        if not do_sharpen:
            self.base_image = self.original_image
        else:
            self.base_image = self.sharpen_image
        # cl = clock()
        self.base_px = QPixmap.fromImage(
            QImage(self.base_image.data, self.base_image.shape[1], self.base_image.shape[0],
                   self.base_image.shape[1], QImage.Format_Indexed8) \
                .scaled(self.fixedWidth, self.fixedHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # print clock() - cl
        self.setPixmap(self.base_px)

    def cal_zoomin(self, mouseX, mouseY):
        ratio = np.float16(2.5)

        x = np.float16(mouseX) / self.fixedWidth
        y = np.float16(mouseY) / self.fixedHeight

        if x <= 0.5 / ratio:
            x = 0
        elif x >= 1 - 0.5 / ratio:
            x = 1 - 1 / ratio
        else:
            x = x - 0.5 / ratio

        if y <= 0.5 / ratio:
            y = 0
        elif y >= 1 - 0.5 / ratio:
            y = 1 - 1 / ratio
        else:
            y = y - 0.5 / ratio

        x = int(np.floor(self.base_image.shape[1] * x))
        y = int(np.floor(self.base_image.shape[0] * y))
        w = int(np.floor(self.base_image.shape[1] / ratio))
        h = int(np.floor(self.base_image.shape[0] / ratio))

        v = np.ascontiguousarray(self.base_image[y:(y + h), x:(x + w)])
        if self.cal_th_ev.is_set():
            return

        im2 = QImage(v.data, v.shape[1], v.shape[0],
                     v.shape[1], QImage.Format_Indexed8)
        if self.cal_th_ev.is_set():
            return
        self.emit(SIGNAL('set_pixmap_qimage'), im2)

    def cal_brightness_contrast(self, mouseX, mouseY, use_downsampled=True):
        # cl = clock()

        if not use_downsampled:
            using_image = self.base_image
        else:
            using_image = self.downsampled_sharpen_image if self.is_sharp_image else self.downsampled_image

        contrast = self.sigmoid(np.float16(mouseX - self.mousePressedPosX) / self.fixedWidth)
        brightness = self.sigmoid(np.float16(mouseY - self.mousePressedPosY) / self.fixedHeight)
        # print contrast, brightness
        v = (using_image * contrast + brightness)
        v[v < 0] = 0
        v[v > 255] = 255
        v = v.astype(np.uint8)
        if self.cal_th_ev.is_set():
            return
        im2 = QImage(v.data, using_image.shape[1], using_image.shape[0],
                     using_image.shape[1], QImage.Format_Indexed8)
        if self.cal_th_ev.is_set():
            return
        self.emit(SIGNAL('set_pixmap_qimage'), im2)

        if use_downsampled:
            try:
                self.original_image_processing_tm.cancel()
            except:
                pass
            finally:
                th = threading.Timer(0.05, self.cal_brightness_contrast, [mouseX, mouseY, False])
                th.start()
                self.original_image_processing_tm = th

                # print clock() - cl

    def set_pixmap_qimage(self, im):
        # self.setPixmap(QPixmap.fromImage(im))
        self.setPixmap(QPixmap.fromImage(
            im.scaled(self.fixedWidth, self.fixedHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation)))

    def sigmoid(self, x):
        return 1 + x / (1 + abs(x))

    def scale_image(self, v):
        w, h = np.float16(v.shape[0]), np.float16(v.shape[1])
        if h / w > np.float16(self.fixedHeight) / self.fixedWidth:
            return cv2.resize(v, (self.fixedHeight, int(np.round(w * self.fixedHeight / h))))
        else:
            return cv2.resize(v, (int(np.round(h * self.fixedWidth / w)), self.fixedWidth))


class ProgressWin(QWidget):
    def __init__(self, app=None):
        super(ProgressWin, self).__init__()
        self.app = app
        self.total_count = 0
        self.read_count = 0
        self.read_time = []
        self.read_time_sum = 0
        self.read_time_mean = 0
        self.estimated_time_remaining = 0
        self.pTick = 0

        progressLabel = QLabel(self)
        progressLabel.setStyleSheet('background-color: transparent; color: rgba(255,255,255,100); ')
        progressLabel.setFixedSize(250, 120)
        progressLabel.setGeometry(0, 0, 250, 120)
        progressLabel.setGeometry(0, 0, 0, 0)
        progressLabel.setFont(QFont("Verdana", 24, QFont.Normal))
        self.progress_label = progressLabel

        self.setGeometry(self.app.x + 50, self.app.h - 150, 250, 120)

        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.connect(self, SIGNAL('update_text'), self.update_text)
        self.connect(self, SIGNAL('show'), self.show_self)

    def next_study(self):
        cl = clock()
        t = (cl - self.pTick) * 1.0
        self.read_time.append(t)
        self.read_count += 1

        if self.read_count == 1:
            self.read_time_mean = t
            self.read_time_sd = 0
        else:
            self.read_time_sd = stat.stdev(self.read_time)
            mean = stat.mean(self.read_time)
            in_range = [t for t in self.read_time if abs(t - mean) < 2 * self.read_time_sd]

            self.read_time_mean = stat.mean(in_range)

        etr = int((self.total_count - self.read_count) * self.read_time_mean)
        if etr < 0:
            self.estimated_time_remaining = ''
        elif etr < 60:
            self.estimated_time_remaining = '%d s' % (etr,)
        else:
            s = etr % 60
            m = int((etr - s) / 60)
            self.estimated_time_remaining = '%dm %ds' % (m, s)
        self.emit(SIGNAL('update_text'))

    def update_text(self):
        s = ('%d / %d\n%.1f ± %.1f\nETR: %s' % (self.read_count + 1
                                                , self.total_count
                                                , self.read_time_mean
                                                , self.read_time_sd
                                                , self.estimated_time_remaining))

        # self.progress_label.setText(QString.fromUtf8(s))
        self.progress_label.setText(s)
        # print s.decode('utf-8')

    def show_self(self):
        self.show()


class ImageViewer(QMainWindow):
    def __init__(self, use_monitor=(1, -1), app=None):
        '''
        :param use_monitor: tuple, default with second to last monitor, use (None, None) for all monitors
        '''
        super(ImageViewer, self).__init__()

        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name)
        self.load_lock = threading.Lock()
        self.show_lock = threading.Lock()
        self.setStyleSheet('background-color: black;')
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.preloading_AccNo = ''
        self.app = app
        self.reset()

        monitors = get_monitors()
        if len(monitors) == 1:
            use_monitor = (0, 0)

        w_w = w_h = w_x = w_y = 0
        tmp_i = 0
        self.image_labels = []
        for i, m in enumerate(sorted(monitors, key=lambda m: m.x)):
            if i < use_monitor[0]:
                continue
            if i == use_monitor[0]:
                w_x, w_y = m.x, m.y
                self.app.x = m.x
                self.app.y = m.y
                self.app.h = m.height

            tmp_i += 1

            if m.height > w_h:
                w_h = m.height

            u = (m.height + m.width) / 4000.0  # length unit

            imageLabel = ImageLabel(self)
            imageLabel.setStyleSheet('background-color: black;')
            imageLabel.setFixedSize(m.width, m.height)
            imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            # imageLabel.setScaledContents(True)
            imageLabel.setGeometry(w_w, 0, m.width, m.height)
            imageLabel.setAlignment(Qt.AlignCenter)
            imageLabel.fixedWidth = m.width
            imageLabel.fixedHeight = m.height

            countLabel = QLabel(imageLabel)
            countLabel.setStyleSheet('background-color: transparent; color: rgba(255,255,255,100); ')
            countLabel.setFixedSize(200 * u, 100 * u)
            countLabel.setGeometry(50 * u, 50 * u, 250 * u, 100 * u)
            countLabel.setFont(QFont("Verdana", 50 * u, QFont.Normal))

            curtainLabel = QLabel(imageLabel)
            curtainLabel.setStyleSheet('background-color: rgba(0,0,0,100); ')
            curtainLabel.setFixedSize(m.width, m.height)
            curtainLabel.setGeometry(w_w, m.y, m.width, m.height)
            curtainLabel.hide()

            imageLabel.count_label = countLabel
            imageLabel.curtain_label = curtainLabel
            self.image_labels.append(imageLabel)

            if tmp_i == 2 or use_monitor[0] == use_monitor[1]:
                oldHxLabel = QTextEdit(self)
                oldHxLabel.setStyleSheet('background-color: rgb(0,0,0,50); color: rgb(255,255,255,200); ')
                oldHxLabel.viewport().setAutoFillBackground(False)
                oldHxLabel.setGeometry(w_w, m.y + m.height - 200 * u, m.width, 200 * u)
                oldHxLabel.setFont(QFont('Verdana', 24 * u, QFont.Normal))
                oldHxLabel.setReadOnly(True)
                # oldHxLabel.setText('test\ntest\ntest\ntest\ntest\ntest')
                # oldHxLabel.show()
                oldHxLabel.hide()
                self.old_hx_label = oldHxLabel

            w_w += m.width

            if i == use_monitor[1]:
                break

        infoLabel = QLabel(self)
        infoLabel.setStyleSheet('background-color: transparent; color: rgba(255,255,255,100); ')
        infoLabel.setFixedSize(250 * u, 100 * u)
        infoLabel.setGeometry(50 * u, 150 * u, 250 * u, 100 * u)
        infoLabel.setFont(QFont("Verdana", 24 * u, QFont.Normal))
        self.info_label = infoLabel

        self.setFixedSize(w_w, w_h)
        self.move(w_x, w_y)
        self.hide()
        self.setEnabled(False)

        self.connect(self, SIGNAL('load'), self.load)
        self.connect(self, SIGNAL('show'), self.show)
        self.connect(self, SIGNAL('load_old_hx'), self.load_old_hx)
        self.connect(self, SIGNAL('show_image'), self.show_image)
        self.connect(self, SIGNAL('show_enable'), self.show_enable)
        self.connect(self, SIGNAL('hide_disable'), self.hide_disable)
        self.connect(self, SIGNAL('next_image'), self.next_image)
        self.connect(self, SIGNAL('prior_image'), self.prior_image)
        self.connect(self, SIGNAL('change_image'), self.change_image)
        self.connect(self, SIGNAL('hide_count_label'), self.hide_count_label)
        self.connect(self, SIGNAL('show_count_label'), self.show_count_label)
        self.connect(self, SIGNAL('hide_old_hx'), self.hide_old_hx)
        self.connect(self, SIGNAL('clear_labels'), self.clear_labels)
        self.connect(self, SIGNAL('border_color_toggle'), self.border_color_toggle)

        # self._define_global_shortcuts()

        # def keyPressEvent(self, QKeyEvent):
        # if QKeyEvent.key() == Qt.Key_Z:
        #     image_label = self.image_labels[0]
        #     image_label.zooming_mode = not image_label.zooming_mode
        #     if not image_label.zooming_mode:
        #         image_label.emit(SIGNAL('set_pixmap'), image_label.original_px)

    # def keyReleaseEvent(self, QKeyEvent):
    #     if QKeyEvent.key()==Qt.Key_Z:
    #         image_label=self.image_labels[0]
    #         image_label.emit(SIGNAL('set_pixmap'), image_label.original_px)
    #         image_label.z_pressed=False
    def border_color_toggle(self, to_state=None):
        '''
        :param state: False to hide border
        :return:
        '''

        if to_state is not None:
            self.border_color_state = to_state
        else:
            self.border_color_state = ~self.border_color_state
        border_width = 5 if self.border_color_state else 0
        self.image_labels[0].setStyleSheet('ImageLabel {border: %dpx solid white;}' % (border_width,))

    def clear_labels(self):
        for l in self.image_labels:
            l.clear()

    def show_enable(self):
        self.setEnabled(True)
        # self.setWindowFlags(Qt.Tool | Qt.Widget | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowStaysOnTopHint)
        self.show()
        self.activateWindow()
        map(lambda l: l.setEnabled(True), self.image_labels)

    def hide_disable(self):
        self.hide()
        # self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.setEnabled(False)

    # def wheelEvent(self, QWheelEvent):
    #
    #     logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
    #     ind = -1
    #     for i, imageLabel in enumerate(self.image_labels):
    #         if imageLabel.contentsRect().contains(QWheelEvent.pos()):
    #             ind = i
    #             break
    #     if ind == -1:
    #         return
    #     if QWheelEvent.delta() < 0:
    #         self.next_image(index=ind)
    #     elif QWheelEvent.delta() > 0:
    #         self.prior_image(index=ind)

    # def _define_global_shortcuts(self):
    #
    #     shortcuts = []
    #
    #     sequence = {
    #         'Ctrl+Shift+Left': self.on_action_previous_comic_triggered,
    #         'Ctrl+Left': self.on_action_first_page_triggered,
    #         'Left': self.on_action_previous_page_triggered,
    #         'Right': self.on_action_next_page_triggered,
    #         'Ctrl+Right': self.on_action_last_page_triggered,
    #         'Ctrl+Shift+Right': self.on_action_next_comic_triggered,
    #         'Ctrl+R': self.on_action_rotate_left_triggered,
    #         'Ctrl+Shift+R': self.on_action_rotate_right_triggered,
    #     }
    #
    #     for key, value in list(sequence.items()):
    #         s = QWidget.QShortcut(QKeySequence(key),
    #                               self.ui.qscroll_area_viewer, value)
    #         s.setEnabled(False)
    #         shortcuts.append(s)
    #
    #     return shortcuts

    def reset(self):
        try:
            list(map(lambda x: x.clear(), self.image_labels))
            list(map(lambda x: x.count_label.setText(''), self.image_labels))
            self.load_threads_ev.set()
            list(map(lambda x: x.join(), self.load_threads))
        except:
            pass
        self.old_hx = ''
        self.load_threads = []
        self.folder = ''
        self.expected_image_count = []
        self.total_image_count = 0
        self.loaded_image = OrderedDict()
        self.ind = OrderedDict()
        self.AccNo = ''
        self.ChartNo = ''
        self.timers = []
        self.preprocessed = []
        self.load_image_th = []
        self.load_lock = threading.Lock()
        self.show_lock = threading.Lock()
        self.border_color_state = False

    def load(self, study):

        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        logging.debug(study['ChartNo'])
        self.reset()

        ChartNo = study['ChartNo']
        AccNo = study['AccNo']
        folder_path = study['folder_path']
        expected_image_count = study['expected_image_count']

        self.ChartNo = ChartNo
        self.folder = folder_path
        self.expected_image_count = expected_image_count

        image_count_sum = 0
        for image_count in self.expected_image_count:
            image_count_sum += sum(image_count.values())
        self.total_image_count = image_count_sum

        # with suppress(Exception):
        #     self.load_lock.release()
        map(lambda th: stop_thread(th), self.load_image_th)
        self.load_image_th = []

        # self.load_dir()
        for i, d in enumerate(self.expected_image_count):
            if len(list(d.keys())) < 1:
                continue
            acc, image_count = list(d.keys())[0], list(d.values())[0]
            if i >= len(self.image_labels):
                break
            self.loaded_image[acc] = {}
            self.ind[acc] = ''
            th = threading.Thread(target=partial(self.load_image, acc, image_count))
            th.start()
            self.load_image_th.append(th)
            self.next_image(acc, i)
        self.AccNo = AccNo
        self.info_label.setText(self.AccNo + '\n' + self.ChartNo)

    # def loading_image(self, AccNo, image_path):
    #     self.loaded_image[AccNo][image_path] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def load_image(self, AccNo, expected_count):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        AccNo, index = self.whichLabel(AccNo=AccNo)
        last_k = 0
        ind = 0
        while True:

            # self.load_lock.acquire()
            for k, image_path in enumerate(glob.glob(os.path.join(self.folder, AccNo + ' *.jpeg'))):
                if image_path not in self.loaded_image[AccNo]:
                    # self.loaded_image[AccNo][image_path] = QImage(image_path)
                    self.loaded_image[AccNo][image_path] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    # threading.Thread(target=self.loading_image, args=(AccNo, image_path)).start()
                    last_k = k
                    break
            is_done = len(self.loaded_image[AccNo]) < self.expected_image_count[index][AccNo]
            # self.load_lock.release()
            if ind > 300:
                logging.error('load image timeout: %s expected %d images, only %d images' %
                              (AccNo, self.expected_image_count[index][AccNo], len(self.loaded_image[AccNo])))
                break
            elif is_done:
                sleep(0.5 if last_k == 0 else 0.1)
                ind += 1
            else:
                break

        return

    def load_single_image(self, AccNo, image_path):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        # self.load_lock.acquire()
        self.loaded_image[AccNo][image_path] = image = QImage(image_path)
        # self.load_lock.release()
        return image

    def load_old_hx(self, old_hx):
        print(self.AccNo)
        print(old_hx)

        self.old_hx_label.clear()
        self.old_hx_label.hide()
        if old_hx.strip() == '':
            self.old_hx_label.hide()
        else:
            self.old_hx_label.setText(old_hx.strip())
            self.old_hx_label.show()

    def hide_old_hx(self):
        self.old_hx_label.hide()

    def whichLabel(self, AccNo='', index=0):
        if AccNo != '':
            for i, (acc, _) in enumerate(self.ind.items()):
                if acc == AccNo:
                    return (AccNo, i)
        else:
            return list(self.ind.items())[index][0], index

    def change_image(self, is_next, mouseX, mouseY):
        index = 0
        g = self.geometry()
        mouseX -= g.left()
        mouseY -= g.top()
        for i, image_label in enumerate(self.image_labels):
            if image_label.geometry().contains(mouseX, mouseY):
                index = i
                break
        if is_next:
            self.emit(SIGNAL('next_image'), '', index)
        else:
            self.emit(SIGNAL('prior_image'), '', index)

    def next_image(self, AccNo='', index=0):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        self.show_lock.acquire()
        AccNo, index = self.whichLabel(AccNo, index)
        expected_image_count = self.expected_image_count[index][AccNo]
        ind = self.ind[AccNo]

        if ind == '':
            ind = 0
        else:
            ind = (ind + 1) % expected_image_count
        self.ind[AccNo] = ind

        # self.show_image(ind, AccNo=AccNo)
        self.emit(SIGNAL('show_image'), ind, AccNo)
        # self.after(1000, self.next_image)

    def prior_image(self, AccNo='', index=0):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        self.show_lock.acquire()
        AccNo, index = self.whichLabel(AccNo, index)
        expected_image_count = self.expected_image_count[index][AccNo]
        ind = self.ind[AccNo]

        if ind == '':
            ind = expected_image_count - 1
        else:
            ind = (ind + expected_image_count - 1) % expected_image_count
        self.ind[AccNo] = ind
        self.show_image(ind, AccNo=AccNo)

    def show_curtain(self, index=0, curtain_label=None):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        if curtain_label is None:
            curtain_label = self.image_labels[index].curtain_label
        curtain_label.raise_()
        curtain_label.show()
        # curtain_label.activateWindow()

    def hide_count_label(self, index):
        self.image_labels[index].count_label.hide()

    def show_count_label(self, index):
        self.image_labels[index].count_label.show()

    def show_image(self, image_ind, AccNo='', index=0):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        AccNo, index = self.whichLabel(AccNo, index)

        image_label = self.image_labels[index]
        image_label.count_label.setText('%d / %d' % (image_ind + 1,
                                                     self.expected_image_count[index][AccNo]))
        try:
            image_label.count_label.hide_label_tm.cancel()
        except:
            pass
        finally:
            th = threading.Timer(1, lambda i: self.emit(SIGNAL('hide_count_label'), i), [index])
            th.start()
            image_label.count_label.hide_label_tm = th

        try:
            image_path = glob.glob(os.path.join(self.folder, AccNo + ' ??????? ' + str(image_ind + 1) + '.jpeg'))[0]
        except:
            image_label.clear()
            # self.show_curtain(index=index)
            print(('Image %d not found!' % image_ind))
            threading.Timer(1, lambda args: self.emit(SIGNAL('show_image'), *args), [[image_ind, AccNo, index]]).start()
            return

        try:
            if image_path in self.loaded_image[AccNo]:
                image = self.loaded_image[AccNo][image_path]
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # self.load_single_image(Acc, image_path, image)
        except:
            image_label.clear()
            # self.show_curtain(index=index)
            print(('Image %d not loaded!' % image_ind))
            threading.Timer(1, lambda args: self.emit(SIGNAL('show_image'), *args), [[image_ind, AccNo, index]]).start()
            return

        self.image_labels[index].curtain_label.hide()
        self.image_labels[index].count_label.show()
        self.info_label.show()

        # TODO: keep preprocessed data according to image_ind, not image_label
        w = image_label.width()
        h = image_label.height()
        px = QPixmap.fromImage(
            QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Indexed8)
                .scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        scaled = px
        image_label.emit(SIGNAL('set_pixmap'), scaled)
        # image_label.setPixmap(scaled)
        image_label.setEnabled(True)
        image_label.show()
        # image_label.activateWindow()

        threading.Thread(target=self.preprocessing, args=(image_label, image, scaled)).start()
        # self.setWindowTitle(image_path)
        self.show_lock.release()
        image_label.curtain_label.hide()
        Send_WM_COPYDATA(self.app.bridge_hwnd, json.dumps({'activateSimpleRIS_2': 1}), self.app.dwData)

    def preprocessing(self, image_label, image, scaled):
        # cl=clock()
        image_label.original_image = image
        image_label.base_image = image
        image_label.downsampled_image = cv2.resize(image, (0, 0), fx=0.3, fy=0.3)
        image_label.base_px = scaled

        im = cv2.GaussianBlur(image, (0, 0), 10)
        cv2.addWeighted(image, 1.5, im, -0.5, 0, im)
        image_label.sharpen_image = im
        image_label.downsampled_sharpen_image = cv2.resize(im, (0, 0), fx=0.3, fy=0.3)
        # print clock()-cl


class ImageViewerApp(QApplication):
    dwData = 17

    def __init__(self, args, folderPath, totalViewer=None):
        super(ImageViewerApp, self).__init__(args)

        self.original_args = args
        if totalViewer is None:
            # try:
            #     self.total_viewer_count = int(list[0])
            # except:
            #     self.total_viewer_count = 4
            self.total_viewer_count = 4
            # print list
            for v in args:
                try:
                    self.total_viewer_count = int(v)
                    break
                except:
                    pass

        else:
            self.total_viewer_count = totalViewer
        self.screen_count = QDesktopWidget().screenCount()

        # self.current_study = QMainWindow()
        self.folder_path = folderPath
        self.viewers = []
        self.viewer_index = -1
        self.study_index = -1
        self.show_index = -1
        self.study_list = {}
        self.preload_timers = []
        # self.study_list_lock = threading.Lock()
        self.show_study_lock = threading.Lock()
        self.load_thread_lock = threading.Lock()
        self.bridge_hwnd = 0
        self.old_hx_list = {}
        self.AccNo = ''
        self.old_hx_threads = []
        self.old_hx_threads_ev = threading.Event()
        self.fast_mode = False
        self.turbo_mode = False
        self.total_study_count = 0
        self.x = self.y = self.h = 0
        self.non_fast_mode_win_pos = []

        if self.total_viewer_count > 2:
            self.preload_count = self.total_viewer_count - 2
        elif self.total_viewer_count > 1:
            self.preload_count = 1
        else:
            self.preload_count = 0

        for _ in range(self.total_viewer_count):
            self.viewers.append(ImageViewer(app=self))

        self.progressWin = ProgressWin(app=self)

        oldHxLabel = QLabel()
        oldHxLabel.setGeometry(0, 0, 0, 0)
        oldHxLabel.hide()
        self.oldHx_label = oldHxLabel

        self.connect(self, SIGNAL('show_study'), self.show_study)
        self.connect(self, SIGNAL('next_study'), self.next_study)
        self.connect(self, SIGNAL('activate_main'), self.activate_main)
        self.connect(self, SIGNAL('hide_all'), self.hide_all)
        self.connect(self, SIGNAL('show_dialog'), self.show_dialog)
        self.WM_COPYDATA_Listener = WM_COPYDATA_Listener(receiver=self.listener)

        # self.next_study()

    def reset(self):
        list(map(lambda t: t.close(), self.viewers))
        self.viewers = []
        for _ in range(self.total_viewer_count):
            self.viewers.append(ImageViewer(app=self))
        self.viewer_index = -1
        self.study_index = -1
        self.show_index = -1
        self.study_list = {}
        self.fast_mode = False
        self.turbo_mode = False
        self.total_study_count = 0
        self.preload_timers = []
        self.show_study_lock = threading.Lock()
        self.load_thread_lock = threading.Lock()
        self.bridge_hwnd = 0
        self.old_hx_list = {}
        self.AccNo = ''
        self.old_hx_threads = []
        self.non_fast_mode_win_pos = []

    def load(self, jsonStr):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        return self.listener(dwData=ImageViewerApp.dwData, lpData=jsonStr)

    def listener(self, *args, **kwargs):
        # logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        try:
            if kwargs['dwData'] != ImageViewerApp.dwData:
                return

            json_data = json.loads(kwargs['lpData'])

            if 'study_data' in json_data:
                for k, l in json_data.items():
                    if k != 'study_data':
                        l['folder_path'] = os.path.join(self.folder_path, l['AccNo'] + ' ' + l['ChartNo'])
                        self.study_list[int(k)] = l
            elif 'oldHx' in json_data:
                self.old_hx_list.update(json_data['oldHx'])
                # logging.warning(self.old_hx_list)
            elif 'next_study' in json_data:
                case = int(json_data['next_study'])
                if case == 1:
                    self.next_study()
                else:
                    self.prior_study()
                    # self.emit(SIGNAL('next_study'))
                    # QTimer.singleShot(0, self.next_study)
            elif 'next_image' in json_data:
                case = int(json_data['next_image'])

                self.viewers[self.viewer_index].emit(SIGNAL('change_image'),
                                                     case,
                                                     json_data['x'],
                                                     json_data['y'])
            elif 'activate_main' in json_data:
                self.emit(SIGNAL('activate_main'))
            elif 'setBridgeHwnd' in json_data:
                logging.debug('set bridge: ' + str(json_data['setBridgeHwnd']))
                self.bridge_hwnd = int(json_data['setBridgeHwnd'])

            elif 'total_study_count' in json_data:
                self.total_study_count = int(json_data['total_study_count'])
                self.progressWin.total_count = self.total_study_count
            elif 'request_info' in json_data:
                v = self.viewers[self.viewer_index]
                d = {}
                d['request_info'] = 1
                d['from'] = json_data['from']
                d['AccNo'] = v.AccNo
                d['ChartNo'] = v.ChartNo
                Send_WM_COPYDATA(self.bridge_hwnd, json.dumps(d), ImageViewerApp.dwData)
                if json_data['from'] == 'open_impax':
                    self.emit(SIGNAL('hide_all'))
            elif 'sendReportEnd' in json_data:
                self.progressWin.next_study()
            elif 'fast_mode' in json_data:
                self.fast_mode = bool(json_data['fast_mode'])
                self.fast_mode_change()
            elif 'turbo_mode' in json_data:
                self.turbo_mode = bool(json_data['turbo_mode'])
                self.fast_mode_change()
            elif 'start_from' in json_data:
                self.emit(SIGNAL('next_study'), json_data['start_from'])
            elif 'reset_all' in json_data:
                self.reset()


        except Exception as e:
            print(e)
            return

    def fast_mode_change(self):
        monitors = get_monitors()
        self.use_monitor = (0,) if len(monitors) == 1 else (1,)
        self.monitors = sorted(monitors, key=lambda m: m.x)[self.use_monitor[0]:]
        self.monitors_total = len(self.monitors)

        if self.turbo_mode or (self.fast_mode and self.monitors_total == 1):
            expected_viewers_total = 4
            if expected_viewers_total > self.total_viewer_count:
                for _ in range(expected_viewers_total - self.total_viewer_count):
                    self.viewers.append(ImageViewer(app=self))
                self.total_viewer_count = expected_viewers_total

            self.non_fast_mode_win_pos = []
            m = self.monitors[self.use_monitor[0]]
            w, h = int(m.width / 2.0), int(m.height / 2.0)
            x, y = np.full((1, 4), m.x), np.full((1, 4), m.y)
            x += np.array([0, w, w, 0])
            y += np.array([0, 0, h, h])
            for i, v in enumerate(self.viewers):
                self.non_fast_mode_win_pos.append(v.geometry())
                v.setFixedSize(w, h)
                v.move(x[0, i], y[0, i])
                v.image_labels[0].setFixedSize(w, h)

            # if self.first_launch:
            #     for i in range(3, 0, -1):
            #         self.viewers[i].emit(SIGNAL('show_enable'))
            #         self.preload(i+1)

        elif self.fast_mode:
            expected_viewers_total = self.monitors_total * 2
            if expected_viewers_total > self.total_viewer_count:
                for _ in range(expected_viewers_total - self.total_viewer_count):
                    self.viewers.append(ImageViewer(app=self))
                self.total_viewer_count = expected_viewers_total

            self.non_fast_mode_win_pos = []

            for i, v in enumerate(self.viewers):
                self.non_fast_mode_win_pos.append(v.geometry())

                m = self.monitors[i % self.monitors_total]
                v.setFixedSize(m.width, m.height)
                v.move(m.x, m.y)

            self.monitors_ind = 0
        else:
            stored_win_pos_count = len(self.non_fast_mode_win_pos)
            for i, v in enumerate(self.viewers):
                if i < stored_win_pos_count:
                    v.setGeometry(self.non_fast_mode_win_pos[i])
                else:
                    v.setGeometry(self.non_fast_mode_win_pos[stored_win_pos_count - 1])

    def next_study(self, from_ind=None):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')

        # if not from_ind in self.study_list:
        #     threading.Timer(0.5, self.next_study, [from_ind]).start()
        #     return

        with suppress(Exception):
            self.next_study_th.cancel()

        self.show_study_lock.acquire()
        if from_ind is not None:
            from_ind = int(from_ind)
            # self.study_index = from_ind - 1
            thisStudyInd = from_ind
        else:
            thisStudyInd = self.study_index + 1

        if not thisStudyInd < self.total_study_count:
            self.show_study_lock.release()
            self.viewers[self.viewer_index].emit(SIGNAL('hide_disable'))
            self.emit(SIGNAL('show_dialog'))
            return
        if not thisStudyInd in self.study_list:
            self.show_study_lock.release()
            self.next_study_th = threading.Timer(0.5, self.next_study, [from_ind])
            self.next_study_th.start()
            return

        thisViewerInd = self.next_index(self.viewer_index, self.total_viewer_count)

        self.show_study(viewer=thisViewerInd, study=thisStudyInd, from_next=True)
        # self.emit(SIGNAL('show_study'), thisViewerInd, thisStudyInd)

        if self.turbo_mode:
            if self.show_index <= 0:
                for i in range(3):
                    self.preload(i + 1)
                    self.viewers[i + 1].emit(SIGNAL('show_enable'))
            else:
                self.preload(3)
                self.viewers[(self.viewer_index + 3) % self.total_viewer_count].emit(SIGNAL('show_enable'))
            # pass
        else:

            try:
                list(map(lambda t: t.cancel(), self.preload_timers))
            except:
                pass
            finally:
                self.preload_timers = []
                for i in range(self.preload_count):
                    if i == 0:
                        self.preload()
                    elif self.fast_mode:
                        break
                    else:
                        th = threading.Timer(i / 2.0, partial(self.preload, i + 1))
                        th.start()
                        self.preload_timers.append(th)

    def prior_study(self):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')

        with suppress(Exception):
            self.next_study_th.cancel()

        self.show_study_lock.acquire()
        thisStudyInd = self.study_index - 1
        if thisStudyInd < 0:
            print('Beyond first study!')
            self.show_study_lock.release()
            return
        thisViewerInd = self.prior_index(self.viewer_index, self.total_viewer_count)

        self.show_study(viewer=thisViewerInd, study=thisStudyInd)
        # self.emit(SIGNAL('show_study'), thisViewerInd, thisStudyInd)

    def show_dialog(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Already last study!\nExit?")
        msg.setWindowTitle("miniPACS")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowStaysOnTopHint)

        if msg.exec_() == QMessageBox.Ok:
            # Send_WM_COPYDATA(self.bridge_hwnd, json.dumps({'exit': 1}), ImageViewerApp.dwData)
            # stop_thread(self.WM_COPYDATA_Listener.thread)
            self.WM_COPYDATA_Listener.quit()
            self.exit(0)
            # sys.exit(0)
            # self.quit()

    def show_study(self, viewer, study, from_next=False):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')

        viewer = int(viewer)
        study = int(study)

        w = self.viewers[viewer]
        s = self.study_list[study]
        AccNo = s['AccNo']
        ChartNo = s['ChartNo']

        if self.fast_mode:
            g = w.geometry()
            Send_WM_COPYDATA(self.bridge_hwnd, json.dumps({'showStudy': 1, 'index': study, 'AccNo': AccNo,
                                                           'ChartNo': ChartNo,
                                                           'x': g.x(),
                                                           'y': g.y(),
                                                           'w': g.width(),
                                                           'h': g.height()}), ImageViewerApp.dwData)
            # self.current_study.setWindowTitle('CurrentStudy: ' + json.dumps({'showStudy': 1, 'index': study, 'AccNo': AccNo,
            #                                                'x': g.x(),
            #                                                'y': g.y(),
            #                                                'w': g.width(),
            #                                                'h': g.height()}))
        else:
            Send_WM_COPYDATA(self.bridge_hwnd, json.dumps({'showStudy': 1, 'index': study, 'AccNo': AccNo,
                                                           'ChartNo': ChartNo, }), ImageViewerApp.dwData)
            # self.current_study.setWindowTitle(
            #     'CurrentStudy: ' + json.dumps({'showStudy': 1, 'index': study, 'AccNo': AccNo}))
        # print self.current_study.windowTitle()
        # Send_WM_COPYDATA(self.bridge_hwnd, json.dumps({'showStudy':1}), ImageViewerApp.dwData)

        c = self.viewers[self.viewer_index]

        if w.AccNo != AccNo and w.preloading_AccNo != AccNo:
            # self.load_thread_lock.acquire()
            logging.info('load now')
            # w.load(**s)
            w.preloading_AccNo = AccNo
            w.emit(SIGNAL('load'), s)
            # SetWindowPos.insertAfter(w.winId(), c.winId())
            # self.load_thread_lock.release()

        w.emit(SIGNAL('hide_old_hx'))
        # WinToTop(w.winId())

        w.emit(SIGNAL('show_enable'))
        # w.setEnabled(True)
        # w.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowStaysOnTopHint)
        # w.show()
        # w.activateWindow()

        # c.emit(SIGNAL('hide_disable'))

        # c.hide()
        # c.setEnabled(False)

        w.emit(SIGNAL('show_count_label'), 0)
        th = threading.Timer(1, lambda i: w.emit(SIGNAL('hide_count_label'), i), [0])
        th.start()
        w.image_labels[0].count_label.hide_label_tm = th

        self.viewer_index = viewer
        self.study_index = study
        # print str(viewer) +','+str(study)
        if from_next:
            self.progressWin.pTick = clock()
            self.progressWin.emit(SIGNAL('show'))
        self.show_study_lock.release()
        self.AccNo = AccNo

        # self.load_old_hx()
        if self.turbo_mode:
            w.emit(SIGNAL('border_color_toggle'), True)
            c.emit(SIGNAL('border_color_toggle'), False)
            #
            # if self.first_launch:
            #     for i in range(3, 0, -1):
            #         self.viewers[i].emit(SIGNAL('show_enable'))
            # else:
            #     c.emit(SIGNAL('border_color_toggle'))
            #     self.viewers[(self.viewer_index + 3) % 4].emit(SIGNAL('show_enable'))
        elif not self.fast_mode:
            try:
                self.old_hx_threads_ev.set()
                list(map(lambda t: t.join(), self.old_hx_threads))
            except:
                pass
            finally:
                self.old_hx_threads_ev.clear()
                th = threading.Thread(target=partial(self.load_old_hx, AccNo, w))
                th.start()
                self.old_hx_threads.append(th)

        self.show_index += 1

    def load_old_hx(self, AccNo=None, win=None):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        if AccNo is None:
            AccNo = self.AccNo
        if win is None:
            win = self.viewers[self.viewer_index]
        while True:
            if self.old_hx_threads_ev.is_set():
                return
            elif AccNo in self.old_hx_list:
                old_hx = self.old_hx_list[AccNo]
                win.old_hx = old_hx.strip()

                win.emit(SIGNAL('load_old_hx'), old_hx.strip())
                # win.old_hx_label.setText(old_hx)
                # if old_hx == '':
                #     win.old_hx_label.hide()
                # else:
                #     win.old_hx_label.show()
                return
            else:
                sleep(0.5)

    def hide(self):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        self.show_study_lock.acquire()
        c = self.viewers[self.viewer_index]
        c.emit(SIGNAL('hide_disable'))
        self.show_study_lock.release()

    def next_index(self, ind, total):
        return (ind + 1) % total

    def prior_index(self, ind, total):
        return (ind - 1 + total) % total

    def preload(self, inc=1):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')

        # self.load_thread_lock.acquire()

        preload_ind = (self.viewer_index + inc) % self.total_viewer_count
        preload_prior_ind = (self.viewer_index + inc - 1) % self.total_viewer_count
        study_ind = self.study_index + inc

        # hwndInsertAfter = self.viewers[preload_prior_ind].winId()
        # self.study_list_lock needed?
        if not study_ind < self.total_study_count:
            return

        while not study_ind in self.study_list:
            sleep(0.5)

        study = self.study_list[study_ind]
        viewer = self.viewers[preload_ind]
        if viewer.AccNo != study['AccNo'] and viewer.preloading_AccNo != study['AccNo']:
            # viewer.load(**study)
            viewer.emit(SIGNAL('clear_labels'))
            viewer.emit(SIGNAL('load'), study)
            viewer.preloading_AccNo = study['AccNo']
            viewer.old_hx = ''

            if self.fast_mode and inc == 1:
                viewer.emit(SIGNAL('show_enable'))

                # SetWindowPos.insertAfter(viewer.winId(), hwndInsertAfter)

                # viewer.show()
                # if inc == 1:
                #     viewer.emit(SIGNAL('show'))
                # self.load_thread_lock.release()

    def save_report(self, next=True):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        study = self.study_list[self.study_index]
        try:
            Send_WM_COPYDATA(self.bridge_hwnd, json.dumps(study), ImageViewerApp.dwData)
            if next:
                self.emit(SIGNAL('next_study'))
        except:
            return

    def activate_main(self):
        self.viewers[self.viewer_index].emit(SIGNAL('show_enable'))

    def hide_all(self):
        self.viewers[self.viewer_index].emit(SIGNAL('hide_disable'))
        for w in self.viewers:
            w.emit(SIGNAL('hide_disable'))


def getMyDocPath():
    CSIDL_PERSONAL = 5  # My Documents

    SHGFP_TYPE_CURRENT = 0  # Want current, not default value
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_PERSONAL, 0, SHGFP_TYPE_CURRENT, buf)
    return buf.value


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app = ImageViewerApp(sys.argv, os.path.join(getMyDocPath(), 'feedRIS'))
    # app.load(r'[{"AccNo":"T0173515899", "ChartNo":"6380534", "expected_image_count":[{"T0173515899":1}]}]')
    # app.load(
    #     r'[{"AccNo":"T0173580748", "ChartNo":"5180465", "expected_image_count":[{"T0173580748":1}, {"T0173528014":1}]}]')
    sys.exit(app.exec_())


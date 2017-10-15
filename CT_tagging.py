# -*- coding: utf-8 -*-

import Queue
import codecs
import ctypes.wintypes
import glob
import inspect
import json
import logging
import os
import sys
import threading
from collections import OrderedDict
from functools import partial
from time import sleep, clock

import cv2
import dicom
import numpy as np
from PyQt4.QtCore import Qt, SIGNAL, QString, QPoint
from PyQt4.QtGui import QApplication, QMainWindow, QMessageBox, QLabel, QImage
from PyQt4.QtGui import QPixmap, QDesktopWidget, QFont
from PyQt4.QtGui import QWidget
from screeninfo import get_monitors

from win32func import WM_COPYDATA_Listener, Send_WM_COPYDATA


class ViewPort(QLabel):
    def __init__(self, *args, **kwargs):
        super(ViewPort, self).__init__(*args, **kwargs)
        self.mousePressed = False
        self.mouseMoving = False
        self.zooming_mode = False
        self.window_setting = (-600, 1500)
        self.image_ind = -1
        self.number = 0
        self.parent = kwargs.get('parent', None)

        mag_label = QLabel(self)
        mag_label.setFixedSize(400, 400)
        mag_label.setStyleSheet('border: 1px solid white;')
        mag_label.hide()
        self.mag_label = mag_label
        self.cal_th = None
        self.calculating = False
        self.is_sharp_image = False
        self.setMouseTracking(True)
        self.setStyleSheet('background-color: black;')

        self.connect(self, SIGNAL('set_pixmap'), self.setPixmap)
        self.connect(self, SIGNAL('set_pixmap_qimage'), self.set_pixmap_qimage)
        self.connect(self, SIGNAL('sharp_image'), self.sharp_image)
        self.setFocus()

    def apply_window(self, data):
        wl, ww = self.window_setting
        data[data < wl - ww] = wl - ww
        data[data >= wl + ww] = wl + ww - 1
        return ((data - (wl - ww)) / (2 * ww) * 256).astype(np.uint8)

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_Down:
            self.parent.emit(SIGNAL('next_image'), self.number)
        elif k == Qt.Key_Up:
            self.parent.emit(SIGNAL('prior_image'), self.number)
        elif k == Qt.Key_PageDown:
            self.parent.app.emit(SIGNAL('next_study'))
        elif k == Qt.Key_PageUp:
            self.parent.app.emit(SIGNAL('prior_study'))

    def mousePressEvent(self, QMouseEvent):
        self.setFocus()

        if QMouseEvent.button() == Qt.LeftButton:
            point = self.mapToGlobal(QMouseEvent.pos())

            if QApplication.keyboardModifiers() == Qt.ShiftModifier:
                self.parent.emit(SIGNAL('getCoord'), point.x(), point.y(), self.number)
            else:
                self.parent.emit(SIGNAL('getCoord'), point.x(), point.y(), self.number, True)

    def mouseReleaseEvent(self, QMouseEvent):
        return
        if QMouseEvent.button() == Qt.LeftButton or QMouseEvent.button() == Qt.RightButton:
            try:
                self.cal_th.terminate()
                self.original_image_processing_th.cancel()
                self.original_image_processing_th.terminate()
            except:
                pass
            finally:
                # self.setPixmap(self.original_px)
                self.emit(SIGNAL('set_pixmap'), self.base_px)
                self.mousePressed = False

    def mouseMoveEvent(self, e):
        if QApplication.mouseButtons() == Qt.NoButton:
            # if QApplication.mouseButtons() == Qt.NoButton and QApplication.keyboardModifiers() == Qt.ControlModifier:
            l = len(self.parent.cache)
            if l == 0:
                return

            geo = self.geometry()
            slice_z = (e.y() - geo.y()) / (geo.height() * 1.0 / l)
            slice_z = int(round(slice_z))
            self.parent.emit(SIGNAL('next_image'), self.number, slice_z)
        return
        if not self.mousePressed:
            return

        if not self.cal_th or not self.cal_th.isAlive():
            if self.mousePressed == Qt.LeftButton:
                target = self.cal_brightness_contrast
            else:
                target = self.cal_zoomin
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
                   self.base_image.shape[1], QImage.Format_Indexed8)) \
            .scaled(self.fixedWidth, self.fixedHeight, Qt.KeepAspectRatio)
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
        im2 = QImage(v.data, v.shape[1], v.shape[0],
                     v.shape[1], QImage.Format_Indexed8)

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
        im2 = QImage(v.data, using_image.shape[1], using_image.shape[0],
                     using_image.shape[1], QImage.Format_Indexed8)

        self.emit(SIGNAL('set_pixmap_qimage'), im2)

        if use_downsampled:
            try:
                self.original_image_processing_th.cancel()
                self.original_image_processing_th.terminate()
            except:
                pass
            finally:
                th = threading.Timer(0.05, self.cal_brightness_contrast, [mouseX, mouseY, False])
                th.start()
                self.original_image_processing_th = th

                # print clock() - cl

    def set_pixmap_qimage(self, im):
        # self.setPixmap(QPixmap.fromImage(im))
        self.setPixmap(QPixmap.fromImage(im).scaled(self.fixedWidth, self.fixedHeight, Qt.KeepAspectRatio))

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
        self.connect(self, SIGNAL('show'), self.show)

    def next_study(self):
        cl = clock()
        t = (cl - self.pTick) * 1.0
        self.read_time.append(t)
        self.read_time_sum += t
        self.read_count += 1
        m = self.read_time_sum * 1.0 / self.read_count
        self.read_time_mean = m
        sd = (sum([(i - m) ** 2 for i in self.read_time]) / self.read_count) ** 0.5
        self.read_time_sd = sd
        etr = int((self.total_count - self.read_count) * self.read_time_mean)
        if etr < 60:
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

        self.progress_label.setText(QString.fromUtf8(s))
        # print s.decode('utf-8')

    def show_self(self):
        self.show()


class Frame():
    def __init__(self, frame=None, ratio=None, usableHW=(0, 0), show=False, mainWin=None):
        self.w, self.h = usableHW
        self.mainWin = mainWin

        if ratio is not None:
            self.setRatio(ratio)
        elif frame is not None:
            self.setFrame(frame)
        else:
            self.ratio = []

        self.viewports = []

        self.update_pos(show=show)

    def setFrame(self, ratio):
        if ratio is None:
            return []
        ret = []
        cols = len(ratio)
        w = 1.0 / cols
        for ind, sc in enumerate(ratio):
            x, y = ind * 1.0 / cols, 0

            c, r = sc[0], sc[1]
            ww, h = w / c, 1.0 / r
            for i in range(c):
                for j in range(r):
                    ret.append([x + i * ww, y + j * h, ww, h])
        self.ratio = ret

    def setRatio(self, ratio):
        self.ratio = ratio

    def update_pos(self, show=False):
        for i, ratio in enumerate(self.ratio):
            if not i < len(self.viewports):
                vp = ViewPort(parent=self.mainWin)

                vp.setStyleSheet('background-color: black;')
                vp.setAlignment(Qt.AlignCenter)

                self.viewports.append(vp)
            self.viewports[i].setGeometry(ratio[0] * 1.0 * self.w,
                                          ratio[1] * 1.0 * self.h,
                                          ratio[2] * 1.0 * self.w,
                                          ratio[3] * 1.0 * self.h)
            self.viewports[i].number = i
            # self.viewports[i].setStyleSheet('color: white;')
            # self.viewports[i].setText(str(i))

            if show:
                self.viewports[i].show()
                # self.viewports[i].move(ratio[0], ratio[1])

    def get_viewport(self, which):
        return self.viewports[which]


class SetQueue(Queue.Queue):
    def _init(self, maxsize):
        self.queue = set()

    def _put(self, item):
        self.queue.add(item)

    def _get(self):
        return self.queue.pop()

class MainViewer(QMainWindow):
    def __init__(self, app=None):
        super(MainViewer, self).__init__()

        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name)
        self.load_lock = threading.Lock()
        self.show_lock = threading.Lock()
        self.setStyleSheet('background-color: black;')
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.preloading_AccNo = ''
        self.app = app
        self.cache = []
        self.px_cache = {}
        self.volume = []
        self.reset()

        self.monitors = sorted(get_monitors(), key=lambda m: m.x)

        if len(self.monitors) == 1:
            self.frames = Frame(ratio=[[0, 0, 0.5, 1.0], [0.5, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
                                usableHW=(self.monitors[0].width, self.monitors[0].height),
                                mainWin=self, show=True)
            self.setGeometry(self.monitors[0].x,
                             self.monitors[0].y,
                             self.monitors[0].width,
                             self.monitors[0].height)
        else:
            self.frames = Frame(frame=[[1, 1], [1, 2]], usableHW=(self.monitors[1].width + self.monitors[2].width,
                                                                  self.monitors[1].height),
                                mainWin=self, show=True)

            self.setGeometry(self.monitors[1].x,
                             self.monitors[1].y,
                             self.monitors[1].width + self.monitors[2].width,
                             self.monitors[1].height)

        # self.hide()
        # self.setEnabled(False)
        self.setHotkey()

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
        self.connect(self, SIGNAL('getCoord'), self.getCoord)
        self.connect(self, SIGNAL('preloading'), self.preloading)
        self.preload_seq = (1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10)
        self.preload_queue = SetQueue()
        threading.Thread(target=self.preload_th).start()
        threading.Thread(target=self.preload_image_clean).start()
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
    def preload_image(self):
        vp = self.frames.get_viewport(0)
        image_ind = vp.image_ind
        rn = 10
        l = len(self.cache)
        for i in self.preload_seq:
            ind = image_ind + i
            if not 0 <= ind < l:
                continue
            if not ind in self.px_cache:
                # threading.Thread(target=self.preloading, args=(ind,)).start()
                # threading.Thread(target=lambda:self.emit(SIGNAL('preloading'), ind)).start()
                self.preload_queue.put(ind)

                # threading.Thread(target=self.preload_image_clean).start()

    def preload_th(self):
        while (True):
            ind = self.preload_queue.get(True)
            self.emit(SIGNAL('preloading'), ind)

    def preload_image_clean(self):
        while (True):
            image_ind = self.frames.get_viewport(0).image_ind
            for k, v in self.px_cache.items():
                if abs(k - image_ind) > 50:
                    self.px_cache.pop(k, None)
            sleep(0.5)

    def preloading(self, image_ind):
        vp = self.frames.get_viewport(0)
        qi = self.cache[image_ind]['qimage']
        scaled = QPixmap.fromImage(qi).scaled(
            vp.width(), vp.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.px_cache[image_ind] = scaled
        return scaled

    def setHotkey(self):
        # QShortcut(QKeySequence('Ctrl+C'), self.frames.get_viewport(0), partial(self.next_image, 0))
        pass

    def process_cache(self):
        vp = self.frames.get_viewport(0)
        w, h = vp.width(), vp.height()

        d = self.cache[0]['data']
        depth = len(self.cache)

        volume = np.zeros((d.shape[1], d.shape[0], depth), d.dtype)
        i = 0
        for dataDic in self.cache:
            volume[:, :, i] = dataDic['data']
            scaled = dataDic['qpixmap'].scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.cache[i]['scaled'] = scaled
            i += 1
        self.volume = volume

    def getCoord(self, mousePosX, mousePosY, index, showSagCor=False):
        vp = self.frames.get_viewport(index)
        image = self.cache[vp.image_ind]['data']
        geo = vp.geometry()
        point = vp.mapToGlobal(QPoint(geo.x(), geo.y()))
        coord = self.mousePos2Coord((mousePosX, mousePosY),
                                    (point.x(), point.y(), geo.width(), geo.height()),
                                    (image.shape[1], image.shape[0]))
        x, y = int(round(coord[0] * image.shape[1])), int(round(coord[1] * image.shape[0]))

        if showSagCor:
            self.localize(x, y)
        else:
            # print '%d, %d, %d' % (x, y, vp.image_ind+1)
            self.save_nodule(x, y, vp.image_ind + 1)

    def save_nodule(self, x, y, z):
        with codecs.open(os.path.join(self.app.base_dir, 'output.txt'), 'w', 'utf-8') as text_file:
            text_file.write('%s,%d,%d,%d\n' % (self.study_id, x, y, z))

    def mousePos2Coord(self, mousePos=(0, 0), viewport_dim=(0, 0, 0, 0), image_dim=(0, 0)):
        x, y, w, h = viewport_dim
        im_w, im_h = image_dim
        mx, my = mousePos
        if im_w * 1.0 / im_h > w * 1.0 / h:
            if not x <= mx <= x + w - 1:
                return None
            i_w = w
            i_h = i_w * 1.0 / im_w * im_h

            if not y + (h - i_h) * 1.0 / 2 <= my <= y + (h + i_h) * 1.0 / 2:
                return None

            coord0 = (mx - x) * 1.0 / w
            coord1 = (my - (y + (h - i_h) * 1.0 / 2)) / i_h
        else:
            if not y <= my <= y + h - 1:
                return None
            i_h = h
            i_w = i_h * 1.0 / im_h * im_w

            if not (x + (w - i_w) * 1.0 / 2) <= mx <= x(w + i_w) * 1.0 / 2:
                return None

            coord1 = (my - y) * 1.0 / h
            coord0 = (mx - (x + (w - i_w) * 1.0 / 2)) / i_w

        return (coord0, coord1)

    def getRange(self, coord, dim, size):

        ret = []
        for i in range(3):
            if coord[i] < int((size[i] - 1) / 2):
                ret.append((0, size[i]))
            elif coord[i] > dim[i] - int((size[i] - 1) / 2):
                ret.append((dim[i] - size[i], dim[i]))
            else:
                ret.append((coord[i] - int((size[i] - 1) / 2), coord[i] + int((size[i] - 1) / 2) + 1))

        return (ret[0], ret[1], ret[2])

    def localize(self, x, y):
        try:
            z_sp = getattr(self.dicom_info, 'SpacingBetweenSlices', self.dicom_info.SliceThickness)
            x_sp, y_sp = self.dicom_info.PixelSpacing
        except:
            z_sp = x_sp = y_sp = 1

        if z_sp > x_sp:
            factor_z, factor_x = z_sp * 1.0 / x_sp, 1
        elif x_sp > z_sp:
            factor_z, factor_x = 1, x_sp * 1.0 / z_sp
        else:
            factor_z = factor_x = 1

        if z_sp > y_sp:
            factor_z, factor_y = z_sp * 1.0 / y_sp, 1
        elif y_sp > z_sp:
            factor_z, factor_y = 1, y_sp * 1.0 / z_sp
        else:
            factor_z = factor_y = 1

        d = self.cache[0]['data']
        # sag = np.zeros((len(self.cache), d.shape[0]), np.uint8)
        # cor = np.zeros((len(self.cache), d.shape[1]), np.uint8)

        l = len(self.cache)
        cube_size = max(int(l / 4), 128)
        if cube_size % 2 == 0:
            cube_size += 1
        z_cube_size = int(round(factor_x * 1.0 * cube_size / factor_z))
        if z_cube_size % 2 == 0:
            z_cube_size += 1

        sag = np.zeros((z_cube_size, cube_size), np.uint8)
        cor = np.zeros((z_cube_size, cube_size), np.uint8)

        image_ind = self.frames.get_viewport(0).image_ind
        x_range, y_range, z_range = self.getRange((x, y, image_ind),
                                                  (d.shape[1], d.shape[0], l),
                                                  (cube_size, cube_size, z_cube_size))
        i = 0
        for z_ind in range(z_range[0], z_range[1]):
            try:
                d = np.array(self.cache[z_ind]['gray'])
            except:
                continue
            sag[i, :] = d[y_range[0]:y_range[1], x]
            cor[i, :] = d[y, x_range[0]:x_range[1]]
            i += 1

        if i != z_cube_size:
            sag = np.delete(sag, np.s_[i - 1:], 0)
            cor = np.delete(cor, np.s_[i - 1:], 0)

        # print x_range
        # print y_range
        vp = self.frames.get_viewport(1)
        qi = QImage(sag.data, sag.shape[1], sag.shape[0], sag.shape[1], QImage.Format_Indexed8)
        qpx = QPixmap.fromImage(qi).scaled(int(round(sag.shape[1] * factor_y)),
                                           int(round(sag.shape[0] * factor_z)))
        qpx = qpx.scaled(vp.width(), vp.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        vp.setPixmap(qpx)

        vp = self.frames.get_viewport(2)
        qi = QImage(cor.data, cor.shape[1], cor.shape[0], cor.shape[1], QImage.Format_Indexed8)
        qpx = QPixmap.fromImage(qi).scaled(int(round(cor.shape[1] * factor_x)),
                                           int(round(cor.shape[0] * factor_z)))
        qpx = qpx.scaled(vp.width(), vp.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        vp.setPixmap(qpx)

    def show_enable(self):
        self.setEnabled(True)
        # self.setWindowFlags(Qt.Tool | Qt.Widget | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowStaysOnTopHint)
        self.show()
        # self.activateWindow()

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
            map(lambda x: x.terminate(), self.load_threads)
            map(lambda x: x.clear(), self.series_labels)
            map(lambda x: x.count_label.setText(''), self.series_labels)
        except:
            pass
        self.old_hx = ''
        self.load_threads = []
        self.folder = ''
        self.expected_image_count = []
        self.total_image_count = 0
        self.image_ind = 0
        self.loaded_image = OrderedDict()
        self.ind = OrderedDict()
        self.AccNo = ''
        self.ChartNo = ''
        self.timers = []
        self.preprocessed = []
        self.frames = [[0, 0, 0.5, 1], [0.5, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]]

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

        # self.load_dir()
        for i, d in enumerate(self.expected_image_count):
            if len(d.keys()) < 1:
                continue
            acc, image_count = d.keys()[0], d.values()[0]
            if i >= len(self.series_labels):
                break
            self.loaded_image[acc] = {}
            self.ind[acc] = ''
            threading.Thread(target=partial(self.load_image, acc, image_count))
            self.next_image(acc, i)
        self.AccNo = AccNo
        self.info_label.setText(self.AccNo + '\n' + self.ChartNo)

    def load_image(self, AccNo, expected_count):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        AccNo, index = self.whichLabel(AccNo=AccNo)
        last_k = 0
        while True:
            self.load_lock.acquire()
            for k, image_path in enumerate(glob.glob(os.path.join(self.folder, AccNo + ' *.jpeg'))):
                if image_path not in self.loaded_image[AccNo]:
                    # self.loaded_image[AccNo][image_path] = QImage(image_path)
                    self.loaded_image[AccNo][image_path] = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    last_k = k
                    break
            is_done = len(self.loaded_image[AccNo]) < self.expected_image_count[index][AccNo]
            self.load_lock.release()
            if is_done:
                sleep(0.5 if last_k == 0 else 0.1)
            else:
                break

        return

    def load_single_image(self, AccNo, image_path):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        self.load_lock.acquire()
        self.loaded_image[AccNo][image_path] = image = QImage(image_path)
        self.load_lock.release()
        return image

    def load_old_hx(self, old_hx):
        print self.AccNo
        print old_hx

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
            for i, (acc, _) in enumerate(self.ind.iteritems()):
                if acc == AccNo:
                    return (AccNo, i)
        else:
            return self.ind.items()[index][0], index

    def change_image(self, is_next, mouseX, mouseY):
        index = 0
        g = self.geometry()
        mouseX -= g.left()
        mouseY -= g.top()
        for i, image_label in enumerate(self.series_labels):
            if image_label.geometry().contains(mouseX, mouseY):
                index = i
                break
        if is_next:
            self.emit(SIGNAL('next_image'), '', index)
        else:
            self.emit(SIGNAL('prior_image'), '', index)

    def next_image(self, index=0, from_ind=None):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')

        vp = self.frames.get_viewport(index)
        if from_ind is not None:
            image_ind = int(from_ind)
        else:
            image_ind = vp.image_ind + 1

        if not 0 <= image_ind <= len(self.cache) - 1:
            print 'No next image!'
            return

        self.show_lock.acquire()
        vp.image_ind = image_ind

        # self.show_image(ind, AccNo=AccNo)
        self.emit(SIGNAL('show_image'), index)
        # self.after(1000, self.next_image)

    def prior_image(self, index=0):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        vp = self.frames.get_viewport(index)
        if vp.image_ind == 0:
            return

        self.show_lock.acquire()
        vp.image_ind -= 1

        # self.show_image(ind, AccNo=AccNo)
        self.emit(SIGNAL('show_image'), index)
        # self.after(1000, self.next_image)

    def show_curtain(self, index=0, curtain_label=None):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        if curtain_label is None:
            curtain_label = self.series_labels[index].curtain_label
        curtain_label.raise_()
        curtain_label.show()
        # curtain_label.activateWindow()

    def hide_count_label(self, index):
        self.series_labels[index].count_label.hide()

    def show_count_label(self, index):
        self.series_labels[index].count_label.show()

    def show_image(self, index=0):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        vp = self.frames.get_viewport(index)
        image_ind = vp.image_ind
        qpx = self.px_cache.get(image_ind, None)
        if qpx is None:
            qpx = self.preloading(image_ind)

        vp.setPixmap(qpx)

        self.show_lock.release()
        threading.Thread(target=self.preload_image()).start()

        # Send_WM_COPYDATA(self.app.bridge_hwnd, json.dumps({'activateSimpleRIS': 1}), self.app.dwData)

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

    def __init__(self, list):
        super(ImageViewerApp, self).__init__(list)
        self.screen_count = QDesktopWidget().screenCount()
        self.WM_COPYDATA_Listener = WM_COPYDATA_Listener(receiver=self.listener)
        # self.folder_path = folderPath
        self.viewers = []
        self.viewer_index = -1
        self.study_index = -1
        self.total_viewer_count = 1
        self.study_list = OrderedDict()
        self.preload_threads = []
        # self.study_list_lock = threading.Lock()
        self.show_study_lock = threading.Lock()
        self.load_thread_lock = threading.Lock()
        self.bridge_hwnd = 0
        self.old_hx_list = {}
        self.AccNo = ''
        self.old_hx_threads = []
        self.fast_mode = False
        self.total_study_count = 0
        self.x = self.y = self.h = 0
        self.cache = {}

        # if self.total_viewer_count > 2:
        #     self.preload_count = self.total_viewer_count - 2
        # elif self.total_viewer_count > 1:
        #     self.preload_count = 1
        # else:
        #     self.preload_count = 0
        #
        # for _ in range(totalViewer):
        #     self.viewers.append(MainViewer(app=self))

        self.viewers.append(MainViewer(app=self))
        self.viewers[0].show()
        # self.progressWin = ProgressWin(app=self)

        # oldHxLabel = QLabel()
        # oldHxLabel.setGeometry(0, 0, 0, 0)
        # oldHxLabel.hide()
        # self.oldHx_label = oldHxLabel

        self.connect(self, SIGNAL('show_study'), self.show_study)
        self.connect(self, SIGNAL('next_study'), self.next_study)
        self.connect(self, SIGNAL('activate_main'), self.activate_main)
        self.connect(self, SIGNAL('hide_all'), self.hide_all)
        self.connect(self, SIGNAL('show_dialog'), self.show_dialog)

        # self.base_dir = r'E:\Nodule Detection\case CT'
        self.base_dir = r'C:\CT_DICOM'
        # self.file_list=OrderedDict
        # self.file_list_ind=-1

        self.load_local_dir()
        threading.Timer(0.5, lambda s: s.emit(SIGNAL('next_study'), self.study_index), [self]).start()

    def load_local_dir(self, from_study_name=''):
        found_study_name_index = 0
        i = 0
        for study in os.listdir(self.base_dir):
            if not os.path.isdir(os.path.join(self.base_dir, study)):
                continue
            if from_study_name == study and found_study_name_index == 0:
                found_study_name_index = i
            i += 1
            self.study_list[study] = OrderedDict()
            for series in os.listdir(os.path.join(self.base_dir, study)):
                self.study_list[study][series] = []
                for images in sorted(glob.glob(os.path.join(self.base_dir, study, series, '*.dcm'))):
                    self.study_list[study][series].append(images)
        self.total_study_count = len(self.study_list)
        self.study_index = found_study_name_index - 1

    def load(self, jsonStr):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        return self.listener(dwData=ImageViewerApp.dwData, lpData=jsonStr)

    def listener(self, *args, **kwargs):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        try:
            if kwargs['dwData'] != ImageViewerApp.dwData:
                return

            json_data = json.loads(kwargs['lpData'])

            if 'study_data' in json_data:
                for k, l in json_data.iteritems():
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
                elif json_data['from'] == 'sendReport':
                    self.progressWin.next_study()
            elif 'fast_mode' in json_data:
                self.fast_mode = bool(json_data['fast_mode'])
            elif 'start_from' in json_data:
                self.emit(SIGNAL('next_study'), json_data['start_from'])


        except Exception as e:
            print e
            return

    def next_study(self, from_ind=None):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')

        # if not from_ind in self.study_list:
        #     threading.Timer(0.5, self.next_study, [from_ind]).start()
        #     return

        self.show_study_lock.acquire()
        if from_ind is not None:
            from_ind = int(from_ind)
            # self.study_index = from_ind - 1
            thisStudyInd = from_ind
        else:
            thisStudyInd = self.study_index + 1

        if not thisStudyInd < self.total_study_count:
            self.show_study_lock.release()
            # self.viewers[self.viewer_index].emit(SIGNAL('hide_disable'))
            self.emit(SIGNAL('show_dialog'))
            return
        # if not thisStudyInd in self.study_list:
        #     threading.Timer(0.5, self.next_study, [from_ind]).start()
        #     self.show_study_lock.release()
        #     return

        # thisViewerInd = self.next_index(self.viewer_index, self.total_viewer_count)

        self.show_study(study=thisStudyInd, from_next=True)
        # self.emit(SIGNAL('show_study'), thisViewerInd, thisStudyInd)
        #
        # try:
        #     map(lambda t: t.cancel(), self.preload_threads)
        #     map(lambda t: t.terminate(), self.preload_threads)
        # except:
        #     pass
        # finally:
        #     self.preload_threads = []
        #     for i in range(self.preload_count):
        #         th = threading.Timer(i + 1, partial(self.preload, i + 1))
        #         th.start()
        #         self.preload_threads.append(th)

    def prior_study(self):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        self.show_study_lock.acquire()
        thisStudyInd = self.study_index - 1
        if thisStudyInd < 0:
            print 'Beyond first study!'
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

        if msg.exec_() == QMessageBox.Ok:
            Send_WM_COPYDATA(self.bridge_hwnd, json.dumps({'exit': 1}), ImageViewerApp.dwData)

    def read_dicom(self, path):
        f = dicom.read_file(path)

        s, i = f.RescaleSlope, f.RescaleIntercept
        return np.array(f.pixel_array) * s + i

    def show_study(self, study, from_next=''):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        # viewer = int(viewer)
        study = int(study)
        # w = self.viewers[viewer]
        study_name = self.study_list.keys()[study]
        s = self.study_list[study_name]

        self.viewers[0].cache = []
        # self.viewers.
        count = 0
        # cl=clock()
        for series, images in s.items():
            # self.viewers[0].cache[series] = OrderedDict()
            for image in images:
                filename = os.path.basename(image)
                # self.viewers[0].cache[series][filename] = OrderedDict()
                data = self.read_dicom(image)
                vp = self.viewers[0].frames.get_viewport(0)
                gray = vp.apply_window(data)
                qi = QImage(gray, gray.shape[1], gray.shape[0], gray.shape[1], QImage.Format_Indexed8)
                # scaled=QPixmap.fromImage(qi).scaled(
                #     vp.width(), vp.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # qpx = QPixmap.fromImage(qi)
                # scaled = qpx.scaled(vp.width(), vp.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                # w = image_label.width()
                # h = image_label.height()
                # scaled = px.scaled(w, h, Qt.KeepAspectRatio)
                #
                dic = {}
                dic['data'] = data
                dic['gray'] = gray
                dic['qimage'] = qi
                # dic['qpixmap'] = qpx
                dic['fullpath'] = image
                # dic['scaled'] = scaled
                self.viewers[0].cache.append(dic)

                if count == 0:
                    qpx = QPixmap.fromImage(qi)
                    scaled = qpx.scaled(vp.width(), vp.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

                    vp.setPixmap(scaled)
                    # print clock()-cl
                    self.viewers[0].px_cache[count] = scaled
                    vp.image_ind = 0
                    self.viewers[0].dicom_info = dicom.read_file(image)
                    self.viewers[0].study_id = study_name
                elif count <= 10:
                    qpx = QPixmap.fromImage(qi)
                    scaled = qpx.scaled(vp.width(), vp.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.viewers[0].px_cache[count] = scaled
                # elif count > 30:
                #     break
                count += 1

        # self.viewers[0].process_cache()
        self.viewers[0].preload_image()
        self.study_index = study
        self.show_study_lock.release()

    def load_old_hx(self, AccNo=None, win=None):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        if AccNo is None:
            AccNo = self.AccNo
        if win is None:
            win = self.viewers[self.viewer_index]
        while True:
            if AccNo in self.old_hx_list:
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
            viewer.emit(SIGNAL('load'), study)
            viewer.preloading_AccNo = study['AccNo']
            viewer.old_hx = ''

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
    app = ImageViewerApp(sys.argv)
    # app.load(r'[{"AccNo":"T0173515899", "ChartNo":"6380534", "expected_image_count":[{"T0173515899":1}]}]')
    # app.load(
    #     r'[{"AccNo":"T0173580748", "ChartNo":"5180465", "expected_image_count":[{"T0173580748":1}, {"T0173528014":1}]}]')
    sys.exit(app.exec_())

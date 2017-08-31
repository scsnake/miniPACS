import os, glob, sys
import threading
from collections import OrderedDict
from screeninfo import get_monitors
from PyQt4.QtGui import QApplication, QMainWindow, QTextEdit, QMessageBox, QGraphicsScene, QLabel, QPalette, QImage
from PyQt4.QtGui import QPixmap, QPainter, QGraphicsPixmapItem, QAction, QKeySequence, QDesktopWidget, QFont
from PyQt4.QtGui import QVBoxLayout, QWidget, QSizePolicy, QFrame, QBrush, QColor
from PyQt4.QtCore import QTimer, QObject, QSize, Qt, QRectF, SIGNAL
from time import sleep, clock
from win32func import WM_COPYDATA_Listener, Send_WM_COPYDATA
import SetWindowPos
import json
import logging, inspect
from functools import partial
import cv2
import numpy as np
import ctypes.wintypes

class ImageLabel(QLabel):
    def __init__(self, *args):
        super(ImageLabel, self).__init__(*args)
        self.mousePressed=False
        self.mouseMoving = False
        self.z_pressed = False

        mag_label = QLabel(self)
        mag_label.setFixedSize(400, 400)
        mag_label.setStyleSheet('border: 1px solid white;')
        mag_label.hide()
        self.mag_label = mag_label

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button()==Qt.LeftButton:
            self.mousePressed=True
            self.mousePressedPosX = QMouseEvent.x()
            self.mousePressedPosY = QMouseEvent.y()

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.setPixmap(self.original_px)
            self.mousePressed=False

    def mouseMoveEvent(self, QMouseEvent):
        if not self.mousePressed and not self.z_pressed:
            return
        if self.mouseMoving:
            return
        self.mouseMoving=True

        if not self.z_pressed:
            cl=clock()
            contrast = self.sigmoid((QMouseEvent.x() - self.mousePressedPosX+0.0)/self.fixedWidth)
            brightness = self.sigmoid((QMouseEvent.y() - self.mousePressedPosY+0.0)/self.fixedHeight)
            # print contrast, brightness
            v = (self.original_image * contrast + brightness)
            v[v<0]=0
            v[v>255]=255
            v=v.astype(np.uint8)
            im2 = QImage(v.data, self.original_image.shape[1], self.original_image.shape[0],
                         self.original_image.shape[1], QImage.Format_Indexed8)
            px2 = QPixmap.fromImage(im2)
            self.setPixmap(px2.scaled(self.fixedWidth, self.fixedHeight, Qt.KeepAspectRatio))
            print clock()-cl
        else:
            pass
        self.mouseMoving=False

    # def keyPressEvent(self, QKeyEvent):
    #     if QKeyEvent.key()==Qt.Key_Z:
    #         self.show_mag()
    #         self.z_pressed=True
    #
    # def keyReleaseEvent(self, QKeyEvent):
    #     if QKeyEvent.key()==Qt.Key_Z:
    #         self.z_pressed=False
    #         self.hide_mag()
    #
    # def show_mag(self):


    def sigmoid(self, x):
        return 1+x/(1+abs(x))

class ImageViewer(QMainWindow):
    def __init__(self, use_monitor=(1, -1)):
        '''
        :param use_monitor: tuple, default with second to last monitor, use (None, None) for all monitors
        '''
        super(ImageViewer, self).__init__()
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name)
        self.load_lock = threading.Lock()
        self.show_lock = threading.Lock()
        self.setStyleSheet('background-color: black;')
        self.setWindowFlags(Qt.Tool)
        self.setWindowFlags(Qt.FramelessWindowHint)

        self.reset()

        w_w = w_h = w_x = w_y = 0
        tmp_i = 0
        self.image_labels = []
        for i, m in enumerate(sorted(get_monitors(), key=lambda m: m.x)):
            if i < use_monitor[0]:
                continue
            if i == use_monitor[0]:
                w_x, w_y = m.x, m.y

            tmp_i += 1

            if m.height > w_h:
                w_h = m.height

            imageLabel = ImageLabel(self)
            imageLabel.setStyleSheet('background-color: black;')
            imageLabel.setFixedSize(m.width, m.height)
            imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            # imageLabel.setScaledContents(True)
            imageLabel.setGeometry(w_w, m.y, m.width, m.height)
            imageLabel.setAlignment(Qt.AlignCenter)
            imageLabel.fixedWidth=m.width
            imageLabel.fixedHeight=m.height


            countLabel = QLabel(imageLabel)
            countLabel.setStyleSheet('background-color: transparent; color: rgba(255,255,255,100); ')
            countLabel.setFixedSize(200, 100)
            countLabel.setGeometry(50, 50, 250, 100)
            countLabel.setFont(QFont("Verdana", 50, QFont.Normal))

            curtainLabel = QLabel(imageLabel)
            curtainLabel.setStyleSheet('background-color: rgba(0,0,0,100); ')
            curtainLabel.setFixedSize(m.width, m.height)
            curtainLabel.setGeometry(w_w, m.y, m.width, m.height)
            curtainLabel.hide()

            imageLabel.count_label = countLabel
            imageLabel.curtain_label = curtainLabel
            self.image_labels.append(imageLabel)

            if tmp_i == 2:
                oldHxLabel = QTextEdit(self)
                oldHxLabel.setStyleSheet('background-color: rgb(0,0,0,50); color: rgb(255,255,255,200); ')
                oldHxLabel.viewport().setAutoFillBackground(False)
                oldHxLabel.setGeometry(w_w, m.y + m.height - 200, m.width, 200)
                oldHxLabel.setFont(QFont('Verdana', 24, QFont.Normal))
                # oldHxLabel.setText('test\ntest\ntest\ntest\ntest\ntest')
                # oldHxLabel.show()
                oldHxLabel.hide()
                self.old_hx_label = oldHxLabel

            w_w += m.width

            if i == use_monitor[1]:
                break

        infoLabel = QLabel(self)
        infoLabel.setStyleSheet('background-color: transparent; color: rgba(255,255,255,100); ')
        infoLabel.setFixedSize(200, 100)
        infoLabel.setGeometry(50, 150, 200, 100)
        infoLabel.setFont(QFont("Verdana", 24, QFont.Normal))
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

        # self._define_global_shortcuts()

    def show_enable(self):
        self.setEnabled(True)
        self.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowStaysOnTopHint)
        self.show()
        self.activateWindow()

    def hide_disable(self):
        self.hide()
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
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
            map(lambda x: x.clear(), self.image_labels)
            map(lambda x: x.count_label.setText(''), self.image_labels)
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
            acc, image_count = d.keys()[0], d.values()[0]
            if i >= len(self.image_labels):
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

        self.old_hx_label.clear()
        if old_hx.strip() == '':
            self.old_hx_label.hide()
        else:
            self.old_hx_label.setText(old_hx.strip())
            self.old_hx_label.show()

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
        curtain_label.activateWindow()

    def show_image(self, image_ind, AccNo='', index=0):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        AccNo, index = self.whichLabel(AccNo, index)

        image_label = self.image_labels[index]
        image_label.count_label.setText('%d / %d' % (image_ind + 1,
                                                     self.expected_image_count[index][AccNo]))

        try:
            image_path = glob.glob(os.path.join(self.folder, AccNo + ' ??????? ' + str(image_ind + 1) + '.jpeg'))[0]
        except:
            self.show_curtain(index=index)
            print('Image %d not found!' % image_ind)
            return

        try:
            if image_path in self.loaded_image[AccNo]:
                image = self.loaded_image[AccNo][image_path]
            else:
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                # self.load_single_image(Acc, image_path, image)
        except:
            self.show_curtain(index=index)
            print('Image %d not loaded!' % image_ind)
            return

        px = QPixmap.fromImage(QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Indexed8))
        w = image_label.width()
        h = image_label.height()
        scaled=px.scaled(w, h, Qt.KeepAspectRatio)
        image_label.setPixmap(scaled)
        image_label.setEnabled(True)
        image_label.show()
        image_label.activateWindow()
        image_label.original_image = image
        image_label.original_px = scaled
        self.setWindowTitle(image_path)
        self.show_lock.release()
        image_label.curtain_label.hide()


class ImageViewerApp(QApplication):
    dwData = 17

    def __init__(self, list, folderPath, totalViewer=4):
        super(ImageViewerApp, self).__init__(list)
        self.screen_count = QDesktopWidget().screenCount()
        self.WM_COPYDATA_Listener = WM_COPYDATA_Listener(receiver=self.listener)
        self.folder_path = folderPath
        self.viewers = []
        self.viewer_index = -1
        self.study_index = -1
        self.total_viewer_count = totalViewer
        self.study_list = []
        self.preload_threads = []
        # self.study_list_lock = threading.Lock()
        self.show_study_lock = threading.Lock()
        self.load_thread_lock = threading.Lock()
        self.bridge_hwnd = 0
        self.old_hx_list = {}
        self.AccNo = ''
        self.old_hx_threads = []

        if self.total_viewer_count > 2:
            self.preload_count = 2
        elif self.total_viewer_count > 1:
            self.preload_count = 1
        else:
            self.preload_count = 0

        for _ in range(totalViewer):
            self.viewers.append(ImageViewer())

        oldHxLabel = QLabel()
        oldHxLabel.setGeometry(0, 0, 0, 0)
        oldHxLabel.hide()
        self.oldHx_label = oldHxLabel

        self.connect(self, SIGNAL('show_study'), self.show_study)

        # self.next_study()

    def load(self, jsonStr):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        return self.listener(dwData=ImageViewerApp.dwData, lpData=jsonStr)

    def listener(self, *args, **kwargs):
        logging.debug(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        try:
            if kwargs['dwData'] != ImageViewerApp.dwData:
                return

            json_data = json.loads(kwargs['lpData'])

            if 'setBridgeHwnd' in json_data:
                logging.debug('set bridge: ' + str(json_data['setBridgeHwnd']))
                self.bridge_hwnd = int(json_data['setBridgeHwnd'])
                return
            # if self.bridge_hwnd != kwargs['hwnd']:
            #     return
            if 'oldHx' in json_data:
                self.old_hx_list.update(json_data['oldHx'])
                # logging.warning(self.old_hx_list)
                return
            if 'next_study' in json_data:
                case = int(json_data['next_study'])
                if case == 1:
                    self.next_study()
                else:
                    self.prior_study()
                # self.emit(SIGNAL('next_study'))
                # QTimer.singleShot(0, self.next_study)
                return
            if 'next_image' in json_data:
                case = int(json_data['next_image'])

                self.viewers[self.viewer_index].emit(SIGNAL('change_image'),
                                                     case,
                                                     json_data['x'],
                                                     json_data['y'])
                return

            if 'request_info' in json_data:
                v = self.viewers[self.viewer_index]
                d = {}
                d['request_info'] = 1
                d['from'] = json_data['from']
                d['AccNo'] = v.AccNo
                d['ChartNo'] = v.ChartNo
                Send_WM_COPYDATA(self.bridge_hwnd, json.dumps(d), ImageViewerApp.dwData)
                return

            # self.study_list_lock.acquire()
            for l in json_data:
                l['folder_path'] = os.path.join(self.folder_path, l['AccNo'] + ' ' + l['ChartNo'])
                self.study_list.append(l)  # list.append is atomic
                # self.study_list_lock.release()
            if len(self.study_list) > 0 and self.study_index == -1:
                # QTimer.singleShot(0, self.next_study)
                # self.emit(SIGNAL('next_study'))
                self.next_study()
        except Exception as e:
            print e
            return

    def next_study(self):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        self.show_study_lock.acquire()
        thisStudyInd = self.study_index + 1
        if not thisStudyInd < len(self.study_list):
            self.showdialog()
            self.show_study_lock.release()
            return
        thisViewerInd = self.next_index(self.viewer_index, self.total_viewer_count)

        self.show_study(viewer=thisViewerInd, study=thisStudyInd)
        # self.emit(SIGNAL('show_study'), thisViewerInd, thisStudyInd)

        try:
            map(lambda t: t.cancel(), self.preload_threads)
            map(lambda t: t.terminate(), self.preload_threads)
        except:
            pass
        self.preload_threads = []
        for i in range(self.preload_count):
            th = threading.Timer(i + 1, partial(self.preload, i + 1))
            th.start()
            self.preload_threads.append(th)

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

    def showdialog():
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)

        msg.setText("Already last study!\nExit?")
        msg.setWindowTitle("miniPACS")
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Ok)

        if msg.exec_()==QMessageBox.Ok:
            Send_WM_COPYDATA(self.bridge_hwnd, json.dumps({'exit': 1}), ImageViewerApp.dwData)
            sys.exit()


    def show_study(self, viewer, study):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        w = self.viewers[viewer]
        s = self.study_list[study]
        AccNo = s['AccNo']
        c = self.viewers[self.viewer_index]

        if w.AccNo != AccNo:
            self.load_thread_lock.acquire()
            logging.info('load now')
            # w.load(**s)
            w.emit(SIGNAL('load'), s)
            # SetWindowPos.insertAfter(w.winId(), c.winId())
            self.load_thread_lock.release()

        w.emit(SIGNAL('show_enable'))
        # w.setEnabled(True)
        # w.setWindowFlags(Qt.Widget | Qt.FramelessWindowHint | Qt.WindowSystemMenuHint | Qt.WindowStaysOnTopHint)
        # w.show()
        # w.activateWindow()

        c.emit(SIGNAL('hide_disable'))
        # c.hide()
        # c.setEnabled(False)


        self.viewer_index = viewer
        self.study_index = study
        self.show_study_lock.release()
        self.AccNo = AccNo

        try:
            map(lambda t: t.terminate(), self.old_hx_threads)
        except:
            pass
        finally:
            th = threading.Thread(target=partial(self.load_old_hx, AccNo, w))
            th.start()
            self.old_hx_threads.append(th)
        Send_WM_COPYDATA(self.bridge_hwnd, json.dumps({'activateSimpleRIS':1}), ImageViewerApp.dwData)

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

                win.emit(SIGNAL('load_old_hx'), old_hx)
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

        self.load_thread_lock.acquire()

        preload_ind = (self.viewer_index + inc) % self.total_viewer_count
        preload_prior_ind = (self.viewer_index + inc - 1) % self.total_viewer_count
        study_ind = self.study_index + inc

        # hwndInsertAfter = self.viewers[preload_prior_ind].winId()
        # self.study_list_lock needed?
        while not study_ind < len(self.study_list):
            sleep(0.5)

        study = self.study_list[study_ind]
        viewer = self.viewers[preload_ind]
        if viewer.AccNo != study['AccNo']:
            # viewer.load(**study)
            viewer.emit(SIGNAL('load'), study)

        # SetWindowPos.insertAfter(viewer.winId(), hwndInsertAfter)

        # viewer.show()
        if inc == 1:
            viewer.emit(SIGNAL('show'))
        viewer.old_hx=''

        self.load_thread_lock.release()

    def save_report(self, next=True):
        logging.info(str(self) + ': ' + inspect.currentframe().f_code.co_name + '\n' + str(locals()) + '\n')
        study = self.study_list[self.study_index]
        try:
            Send_WM_COPYDATA(self.bridge_hwnd, json.dumps(study), ImageViewerApp.dwData)
            if next:
                self.next_study()
        except:
            return

def getMyDocPath():
    CSIDL_PERSONAL = 5  # My Documents

    SHGFP_TYPE_CURRENT = 0  # Want current, not default value
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_PERSONAL, 0, SHGFP_TYPE_CURRENT, buf)
    return buf.value

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    app = ImageViewerApp(sys.argv, os.path.join(getMyDocPath(), 'feedRIS'))
    # app.load(r'[{"AccNo":"T0173278453", "ChartNo":"4587039", "expected_image_count":[{"T0173278453":2}]}]')
    # app.load(
    #     r'[{"AccNo":"T0173580748", "ChartNo":"5180465", "expected_image_count":[{"T0173580748":1}, {"T0173528014":1}]}]')
    sys.exit(app.exec_())

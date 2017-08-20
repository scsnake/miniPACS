import os, glob, sys
import threading
from collections import OrderedDict
from functools import partial
from screeninfo import get_monitors
from PyQt4.QtGui import QApplication, QMainWindow, QTextEdit, QGraphicsView, QGraphicsScene, QLabel, QPalette, QImage
from PyQt4.QtGui import QPixmap, QPainter, QGraphicsPixmapItem, QAction, QKeySequence, QDesktopWidget, QFont
from PyQt4.QtGui import QVBoxLayout, QWidget, QSizePolicy, QFrame, QBrush, QColor
from PyQt4.QtCore import QTimer, QObject, QSize, Qt, QRectF, SIGNAL
from time import sleep
from win32func import WM_COPYDATA_Listener, Send_WM_COPYDATA
import SetWindowPos
import json
import psutil


class ImageViewer(QMainWindow):
    def __init__(self, use_monitor=(1, -1)):
        '''
        :param use_monitor: tuple, default with second to last monitor, use (None, None) for all monitors
        '''
        super(ImageViewer, self).__init__()

        self.load_lock = threading.Lock()
        self.show_lock = threading.Lock()
        self.setStyleSheet('background-color: black;')
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.countLabelFont = QFont("Verdana", 50, QFont.Normal)
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

            imageLabel = QLabel(self)
            imageLabel.setStyleSheet('background-color: black;')
            imageLabel.setFixedSize(m.width, m.height)
            imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            imageLabel.setScaledContents(True)
            imageLabel.setGeometry(w_w, m.y, m.width, m.height)

            countLabel = QLabel(imageLabel)
            countLabel.setStyleSheet('background-color: transparent; color: rgba(255,255,255,100); ')
            countLabel.setFixedSize(200, 100)
            countLabel.setGeometry(50, 50, 200, 100)
            countLabel.setFont(self.countLabelFont)

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
                oldHxLabel.setText('test\ntest\ntest\ntest\ntest\ntest')
                oldHxLabel.show()
                # oldHxLabel.hide()
                self.oldhx_label = oldHxLabel

            w_w += m.width

            if i == use_monitor[1]:
                break

        self.setFixedSize(w_w, w_h)
        self.move(w_x, w_y)
        self.hide()
        self.setEnabled(False)
        # self._define_global_shortcuts()

    def wheelEvent(self, QWheelEvent):
        ind = -1
        for i, imageLabel in enumerate(self.image_labels):
            if imageLabel.contentsRect().contains(QWheelEvent.pos()):
                ind = i
                break
        if ind == -1:
            return
        if QWheelEvent.delta() < 0:
            self.next_image(index=ind)
        elif QWheelEvent.delta() > 0:
            self.prior_image(index=ind)

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
        self.load_threads = []
        self.folder = ''
        self.expected_image_count = OrderedDict()
        self.total_image_count = 0
        self.loaded_image = OrderedDict()
        self.ind = OrderedDict()
        self.AccNo = ''
        self.ChartNo = ''

    def load(self, folder_path,
             expected_image_count, AccNo, ChartNo):

        self.AccNo = AccNo
        self.ChartNo = ChartNo
        self.folder = folder_path
        self.expected_image_count = OrderedDict(expected_image_count)
        self.total_image_count = sum(self.expected_image_count.values())

        # self.load_dir()
        for i, (AccNo, image_count) in enumerate(self.expected_image_count.iteritems()):
            if i >= len(self.image_labels):
                break
            self.loaded_image[AccNo] = {}
            self.ind[AccNo] = ''
            load_thread = threading.Thread(target=self.load_image, args=(AccNo, image_count))
            load_thread.start()
            self.load_threads.append(load_thread)
            self.next_image(AccNo, i)

    def load_dir(self):
        self.load_lock.acquire()
        self.images_path = glob.glob(os.path.join(self.folder, '*.jpeg'))
        if len(self.images_path) < self.total_image_count:
            threading.Timer(100, function=self.load_dir)
        self.load_lock.release()

    def load_image(self, AccNo, expected_count):

        last_k = 0
        while True:
            self.load_lock.acquire()
            for k, image_path in enumerate(glob.glob(os.path.join(self.folder, AccNo + ' *.jpeg'))):
                if image_path not in self.loaded_image[AccNo]:
                    self.loaded_image[AccNo][image_path] = QImage(image_path)
                    last_k = k
                    break
            is_done = len(self.loaded_image[AccNo]) < self.expected_image_count[AccNo]
            self.load_lock.release()
            if is_done:
                sleep(0.5 if last_k == 0 else 0.1)
            else:
                break

        return

    def load_single_image(self, AccNo, image_path):
        self.load_lock.acquire()
        self.loaded_image[AccNo][image_path] = image = QImage(image_path)
        self.load_lock.release()
        return image

    def whichLabel(self, AccNo='', index=0):
        if AccNo != '':
            for i, (acc, _) in enumerate(self.ind.iteritems()):
                if acc == AccNo:
                    return (AccNo, i)
        else:
            return (self.ind.items()[index][0], index)

    def next_image(self, AccNo='', index=0):
        self.show_lock.acquire()
        AccNo, index = self.whichLabel(AccNo, index)
        expected_image_count = self.expected_image_count[AccNo]
        ind = self.ind[AccNo]

        if ind == '':
            ind = 0
        else:
            ind = (ind + 1) % expected_image_count
        self.ind[AccNo] = ind

        self.show_image(ind, AccNo=AccNo)
        # self.after(1000, self.next_image)

    def prior_image(self, AccNo='', index=0):
        self.show_lock.acquire()
        AccNo, index = self.whichLabel(AccNo, index)
        expected_image_count = self.expected_image_count[AccNo]
        ind = self.ind[AccNo]

        if ind == '':
            ind = expected_image_count - 1
        else:
            ind = (ind + expected_image_count - 1) % expected_image_count
        self.ind[AccNo] = ind
        self.show_image(ind, AccNo=AccNo)

    def show_curtain(self, index=0, curtain_label=None):
        if curtain_label is None:
            curtain_label = self.image_labels[index].curtain_label
        curtain_label.show()
        curtain_label.raise_()
        curtain_label.activateWindow()

    def show_image(self, image_ind, AccNo='', index=0):
        AccNo, index = self.whichLabel(AccNo, index)

        image_label = self.image_labels[index]
        image_label.count_label.setText('%d / %d' % (image_ind + 1,
                                                     self.expected_image_count[AccNo]))
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
                image = QImage(image_path)
                # self.load_single_image(Acc, image_path, image)
        except:
            self.show_curtain(index=index)
            print('Image %d not loaded!' % image_ind)
            return

        image_label.setPixmap(QPixmap.fromImage(image))
        image_label.setEnabled(True)
        image_label.show()
        image_label.activateWindow()
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
        self.old_hx = {}

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


        # self.next_study()

    def load(self, jsonStr):
        return self.listener(dwData=ImageViewerApp.dwData, lpData=jsonStr)

    def listener(self, *args, **kwargs):
        try:
            if kwargs['dwData'] != ImageViewerApp.dwData:
                return

            json_data = json.loads(kwargs['lpData'])

            if 'setBridgeHwnd' in json_data:
                self.bridge_hwnd = json_data['setBridgeHwnd']
                return

            if 'oldHx' in json_data:
                self.old_hx.update(json_data['oldHx'])
                return

            # self.study_list_lock.acquire()
            for l in json_data:
                l['folder_path'] = os.path.join(self.folder_path, l['AccNo'] + ' ' + l['ChartNo'])
                self.study_list.append(l)  # list.append is atomic
                # self.study_list_lock.release()
            if len(self.study_list) > 0 and self.study_index == -1:
                self.next_study()

        except Exception as e:
            print e
            return

    def next_study(self):
        self.show_study_lock.acquire()
        thisStudyInd = self.study_index + 1
        if not thisStudyInd < len(self.study_list):
            print 'Beyond current study list!'
            self.show_study_lock.release()
            return
        thisViewerInd = self.next_index(self.viewer_index, self.total_viewer_count)

        self.show_study(viewer=thisViewerInd, study=thisStudyInd)
        self.preload()

    def prior_study(self):
        self.show_study_lock.acquire()
        thisStudyInd = self.study_index - 1
        if thisStudyInd < 0:
            print 'Beyond current study list!'
            self.show_study_lock.release()
            return
        thisViewerInd = self.prior_index(self.viewer_index, self.total_viewer_count)

        self.show_study(viewer=thisViewerInd, study=thisStudyInd)

    def show_study(self, viewer, study):
        w = self.viewers[viewer]
        s = self.study_list[study]
        if w.AccNo != s['AccNo']:
            self.load_thread_lock.acquire()
            w.load(**s)
            # w.load(folder_path=self.get_folder_path(study), **s)
            # expected_image_count=s['expected_image_count'],
            # AccNo=s['AccNo'],
            # ChartNo=s['ChartNo'])
            self.load_thread_lock.release()

        c = self.viewers[self.viewer_index]
        c.hide()
        c.setEnabled(False)

        w.setEnabled(True)
        w.show()
        w.raise_()
        w.activateWindow()
        self.viewer_index = viewer
        self.study_index = study
        self.show_study_lock.release()

    def hide(self):
        self.show_study_lock.acquire()
        with self.viewers[self.viewer_index] as c:
            c.hide()
            c.setEnabled(False)
        self.show_study_lock.release()

    def next_index(self, ind, total):
        return (ind + 1) % total

    def prior_index(self, ind, total):
        return (ind - 1 + total) % total

    def preload(self):
        if self.preload_count == 0:
            return

        self.load_thread_lock.acquire()

        try:
            map(lambda t: t.cancel(), self.preload_threads)
            map(lambda t: t.terminate(), self.preload_threads)
        except:
            pass

        self.preload_threads = []
        hwndInsertAfter = self.viewers[self.viewer_index].winId()
        # self.study_list_lock needed?
        for i in range(self.preload_count):
            if not self.study_index + i + 1 < len(self.study_list):
                break
            nextInd = self.next_index(self.viewer_index + i, self.total_viewer_count)
            study = self.study_list[self.study_index + 1 + i]
            viewer = self.viewers[nextInd]
            th = threading.Timer(1000 * (i + 1),
                                 function=viewer.load,
                                 kwargs=study)
            # kwargs={'folder_path': self.get_folder_path(self.study_index + 1 + i),
            #         'expected_image_count': study['expected_image_count'],
            #         'AccNo': study['AccNo'],
            #         'ChartNo': study['ChartNo']})
            th.start()
            self.preload_threads.append(th)

            thisViewerHwnd = viewer.winId()
            SetWindowPos.insertAfter(thisViewerHwnd, hwndInsertAfter)
            hwndInsertAfter = thisViewerHwnd
            viewer.show()

        self.load_thread_lock.release()

    def save_report(self, next=True):
        study = self.study_list[self.study_index]
        try:
            Send_WM_COPYDATA(self.bridge_hwnd, json.dumps(study), ImageViewerApp.dwData)
            if next:
                self.next_study()
        except:
            return


if __name__ == '__main__':
    app = ImageViewerApp(sys.argv, r'C:\Users\IDI\Documents\feedRIS')
    app.load(r'[{"AccNo":"T0173278453", "ChartNo":"4587039", "expected_image_count":{"T0173278453":2}}]')
    sys.exit(app.exec_())

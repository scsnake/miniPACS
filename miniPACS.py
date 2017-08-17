import os, glob, sys
import threading
from collections import OrderedDict
from functools import partial
from screeninfo import get_monitors
from PyQt4.QtGui import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QLabel, QPalette, QImage
from PyQt4.QtGui import QPixmap, QPainter, QGraphicsPixmapItem, QAction, QKeySequence, QDesktopWidget, QFont
from PyQt4.QtGui import QVBoxLayout, QWidget, QSizePolicy, QFrame, QBrush, QColor
from PyQt4.QtCore import QTimer, QObject, QSize, Qt, QRectF
from time import sleep


class ImageViewer(QMainWindow):
    def __init__(self, use_monitor=(1,-1)):
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
        self.image_labels = []
        for i, m in enumerate(sorted(get_monitors(), key=lambda m: m.x)):
            if i < use_monitor[0]:
                continue
            if i == use_monitor[0]:
                w_x, w_y = m.x, m.y

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

            imageLabel.count_label = countLabel
            self.image_labels.append(imageLabel)



            w_w += m.width

            if i == use_monitor[1]:
                break
        self.setFixedSize(w_w, w_h)
        self.move(w_x, w_y)
        # self._define_global_shortcuts()

    def wheelEvent(self, QWheelEvent):
        # map(lambda x: x.clear(), self.image_labels)
        ind = -1
        for i, imageLabel in enumerate(self.image_labels):
            if imageLabel.contentsRect().contains(QWheelEvent.pos()):
                ind = i
                break
        if ind==-1:
            return
        if QWheelEvent.delta() < 0:
            self.next_image(index=ind)
        elif QWheelEvent.delta() > 0:
            self.prior_image(index=ind)

    def _define_global_shortcuts(self):

        shortcuts = []

        sequence = {
            'Ctrl+Shift+Left': self.on_action_previous_comic_triggered,
            'Ctrl+Left': self.on_action_first_page_triggered,
            'Left': self.on_action_previous_page_triggered,
            'Right': self.on_action_next_page_triggered,
            'Ctrl+Right': self.on_action_last_page_triggered,
            'Ctrl+Shift+Right': self.on_action_next_comic_triggered,
            'Ctrl+R': self.on_action_rotate_left_triggered,
            'Ctrl+Shift+R': self.on_action_rotate_right_triggered,
        }

        for key, value in list(sequence.items()):
            s = QWidget.QShortcut(QKeySequence(key),
                                  self.ui.qscroll_area_viewer, value)
            s.setEnabled(False)
            shortcuts.append(s)

        return shortcuts

    def reset(self):
        try:
            map(lambda x: x.terminate(), self.load_threads)
            map(lambda x: x.clear(), self.image_labels)
            map(lambda x: x.count_label.clear(), self.image_labels)
        except:
            pass
        self.load_threads = []
        self.folder = ''
        self.expected_image_count = OrderedDict()
        self.total_image_count = 0
        self.loaded_image = OrderedDict()
        self.ind = OrderedDict()

    def load(self, folder_path,
             expected_image_count):
        '''
        :param folder_path: str
        :param expected_image_count: dict with keys of AccNo, values of total image count
        '''



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
            QTimer.singleShot(100, self.load_dir)
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

    def show_image(self, image_ind, AccNo='', index=0):
        AccNo, index = self.whichLabel(AccNo, index)

        try:
            image_path = glob.glob(os.path.join(self.folder, AccNo + ' ??????? ' + str(image_ind + 1) + '.jpeg'))[0]
        except:
            print('No path for index: %d' % image_ind)
            return

        if image_path in self.loaded_image[AccNo]:
            image = self.loaded_image[AccNo][image_path]
        else:
            image = QImage(image_path)
            # self.load_single_image(Acc, image_path, image)

        self.image_labels[index].setPixmap(QPixmap.fromImage(image))
        self.image_labels[index].count_label.setText('%d / %d' % (image_ind+1,
                                                                       self.expected_image_count[AccNo]))

        self.setWindowTitle(image_path)
        self.show_lock.release()


class ImageViewerApp(QApplication):
    def __init__(self):
        super(ImageViewerApp, self).__init__()
        self.screen_count = QDesktopWidget.screenCount()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    imageViewer = ImageViewer()
    imageViewer.load(r'C:\Users\IDI\Documents\feedRIS\T0173278453 4587039',
                     {'T0173278453': 2})
    imageViewer.show()
    sys.exit(app.exec_())

import os, glob, sys
import threading
from collections import OrderedDict
from functools import partial
from screeninfo import get_monitors
from PyQt4.QtGui import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QLabel, QPalette, QImage
from PyQt4.QtGui import QPixmap, QGraphicsPixmapItem, QAction, QKeySequence, QDesktopWidget
from PyQt4.QtGui import QVBoxLayout, QWidget, QSizePolicy, QFrame, QBrush, QColor
from PyQt4.QtCore import QTimer, QObject, QSize, Qt, QRectF


class ImageViewer(QMainWindow):
    def __init__(self):
        super(ImageViewer, self).__init__()

        self.load_lock = threading.Lock()
        self.show_lock = threading.Lock()
        self.setStyleSheet('background-color: black;')
        self.reset()

    def reset(self):
        try:
            for label in self.image_labels:
                label.setParent(None)
        except:
            pass
        self.image_labels = []
        self.folder = ''
        self.expected_image_count = OrderedDict()
        self.total_image_count = 0
        self.loaded_image = OrderedDict()
        self.ind = OrderedDict()

    def load(self, folder_path,
             expected_image_count,
             use_monitor=(1, -1)):
        '''
        :param folder_path: str
        :param expected_image_count: dict with keys of AccNo, values of total image count
        :param use_monitor: tuple, default with second to last monitor, use (None, None) for all monitors
        '''

        w_w = w_h = w_x = w_y = 0
        for i, m in enumerate(sorted(get_monitors(), key=lambda m: m.x)):
            if i<use_monitor[0]:
                continue
            if i==use_monitor[0]:
                w_x, w_y = m.x, m.y


            if m.height > w_h:
                w_h = m.height

            imageLabel = QLabel()
            imageLabel.setStyleSheet('background-color: black;')
            imageLabel.setFixedSize(m.width, m.height)
            imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            imageLabel.setScaledContents(True)
            imageLabel.move(w_w, m.y)
            self.image_labels.append(imageLabel)

            w_w += m.width

            if i==use_monitor[1]:
                break

        self.folder = folder_path
        self.expected_image_count = OrderedDict(expected_image_count)
        self.total_image_count = sum(self.expected_image_count.values())
        self.setFixedSize(w_w, w_h)
        self.move(w_x, w_y)
        # self.load_dir()
        for i, (AccNo, image_count) in enumerate(self.expected_image_count.iteritems()):
            if i>=len(self.image_labels):
                break
            self.loaded_image[AccNo] = {}
            self.ind[AccNo] = ''
            self.load_image(AccNo, image_count)
            self.next_image(AccNo, i)

    def load_dir(self):
        self.load_lock.acquire()
        self.images_path = glob.glob(os.path.join(self.folder, '*.jpeg'))
        if len(self.images_path) < self.total_image_count:
            QTimer.singleShot(100, self.load_dir)
        self.load_lock.release()

    def load_image(self, AccNo, expected_count):
        self.load_lock.acquire()
        last_k = 0
        for k, image_path in enumerate(glob.glob(os.path.join(self.folder, AccNo + ' *.jpeg'))):
            if image_path not in self.loaded_image[AccNo]:
                self.loaded_image[AccNo][image_path] = QImage(image_path)
                last_k = k
                break
        if len(self.loaded_image[AccNo]) < self.expected_image_count[AccNo]:
            QTimer.singleShot(500 if last_k == 0 else 100, partial(self.load_image, AccNo, expected_count))
        self.load_lock.release()
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
            image_path = glob.glob(os.path.join(self.folder, AccNo + ' ??????? ' + str(image_ind+1) + '.jpeg'))[0]
        except:
            print('No path for index: %d' % image_ind)
            return

        if image_path in self.loaded_image[AccNo]:
            image = self.loaded_image[AccNo][image_path]
        else:
            image = QImage(image_path)
            # self.load_single_image(Acc, image_path, image)

        self.image_labels[index].setPixmap(QPixmap.fromImage(image))

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
                     {'T0173278453':2})
    imageViewer.show()
    sys.exit(app.exec_())

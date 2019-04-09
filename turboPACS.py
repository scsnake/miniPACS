import ctypes.wintypes
import logging
import os
import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from screeninfo import get_monitors


# from win32func import WM_COPYDATA_Listener, Send_WM_COPYDATA

def _get_monitors():
    global app
    d = app.desktop()
    return [d.screenGeometry(i) for i in range(d.screenCount())]


def getMyDocPath():
    CSIDL_PERSONAL = 5  # My Documents

    SHGFP_TYPE_CURRENT = 0  # Want current, not default value
    buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
    ctypes.windll.shell32.SHGetFolderPathW(0, CSIDL_PERSONAL, 0, SHGFP_TYPE_CURRENT, buf)
    return buf.value


class ImageLabel(QLabel):
    def __init__(self, *args, parent=None):
        super(ImageLabel, self).__init__(*args)
        self.border_state = False
        self.setMouseTracking(True)
        self.id = 0
        self.parent = parent

        self.border_toggle(False)
        self.setFrameShape(QFrame.Panel)

    def border_toggle(self, to_state=None):
        '''
        :param state: False to hide border
        :return:
        '''

        if to_state is not None:
            self.border_state = to_state
        else:
            self.border_state = ~self.border_state
        border_width = 5 if self.border_state else 0
        self.setLineWidth(border_width)


    def enterEvent(self, event):
        logging.debug('Enter label #{}'.format(self.id))
        self.parent.active_image_label_id = self.id
        self.border_toggle(True)

    def leaveEvent(self, event):
        logging.debug('Leave label #{}'.format(self.id))
        self.border_toggle(False)


class ImageViewer(QMainWindow):
    def __init__(self, app=None):
        super(ImageViewer, self).__init__()

        self.setStyleSheet('background-color: black;')
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)

        self.app = app

        self.rows = 4
        self.cols = 4
        self.active_image_label_id = 0

        monitors = sorted(get_monitors(), key=lambda m: m.x)
        x = y = w = h = 0
        self.image_labels = []

        l_id = 0
        if len(monitors) > 1:
            self.screen_count = len(monitors) - 1
            for i, m in enumerate(monitors[1:]):
                if i == 0:
                    x = m.x
                    y = m.y
                w += m.width
                h = m.height
                l_w = int(m.width / self.cols)
                l_h = int(m.height / self.rows)

                l_id = self.create_image_labels(self.rows, self.cols, m, (l_w, l_h), l_id)
        else:
            self.screen_count = 0
            m = monitors[0]
            x, y, w, h = m.x, m.y, m.width, m.height
            l_w = int(m.width / self.cols)
            l_h = int(m.height / self.rows)

            self.create_image_labels(self.rows, self.cols, m, (l_w, l_h), l_id)

        self._base_counts = np.array((self.cols * self.rows, self.cols), dtype=np.int)

        self.setFixedSize(w, h)
        self.move(x, y)
        self.show_enable()

    @property
    def active_image_label(self):
        return self.image_labels[self.active_image_label_id]

    def wheelEvent(self, event):
        d = event.angleDelta()
        if d < 0:
            if IS_DEBUG:
                # self.active_image_label.setText(self.active_image_label.text() + '+')
                logging.debug('test+')
        elif d > 0:
            if IS_DEBUG:
                # self.active_image_label.setText(self.active_image_label.text()[:-1])
                logging.debug('test-')

    def id2coord(self, image_label_id):
        which_screen, remainder = np.divmod(image_label_id, self._base_counts[0])
        which_row, which_col = np.divmod(remainder, self._base_counts[1])
        return which_screen, which_row, which_col

    def create_image_labels(self, rows, cols, monitor, label_w_h, init_id):
        m = monitor
        l_w, l_h = label_w_h

        for r in range(rows):
            for c in range(cols):
                imageLabel = ImageLabel(self, parent=self)
                imageLabel.setStyleSheet('background-color: black;')
                imageLabel.setFixedSize(l_w, l_h)
                imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                # imageLabel.setScaledContents(True)
                imageLabel.setGeometry(m.x + c * l_w, m.y + r * l_h, l_w, l_h)
                imageLabel.setAlignment(Qt.AlignCenter)
                imageLabel.fixedWidth = l_w
                imageLabel.fixedHeight = l_h
                imageLabel.id = init_id

                init_id += 1
                if IS_DEBUG:
                    imageLabel.setFont(QFont("Verdana", 50, QFont.Normal))
                    imageLabel.setStyleSheet('background-color: black; color: rgba(255,255,255,100); ')
                    imageLabel.setText('({},{})'.format(r + 1, c + 1))
                self.image_labels.append(imageLabel)
        return init_id

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


class ImageViewerApp(QApplication):
    dwData = 17

    def __init__(self, args, folderPath):
        super(ImageViewerApp, self).__init__(args)

        self.screen_count = QDesktopWidget().screenCount()
        self.folder_path = folderPath

        self.viewer = ImageViewer(app=self)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    IS_DEBUG = logging.getLogger().isEnabledFor(logging.DEBUG)
    app = ImageViewerApp(sys.argv, os.path.join(getMyDocPath(), 'feedRIS'))
    sys.exit(app.exec_())

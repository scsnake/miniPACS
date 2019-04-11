import ctypes.wintypes
import logging
import os
import sys
import csv
import re
import numpy as np
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from screeninfo import get_monitors
from pathlib import Path
import dflite as df
# import tempfile
import threading
from queue import Queue, Empty
from collections import OrderedDict
import cv2
import time
from functools import partial
import glob

# import win32clipboard
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


report_key_mapping = {
    'Key_1': '',
    'Key_2': '',
    'Key_3': '',
    'Key_4': '',
    'Key_5': '',
    'Key_6': '',
    'Key_Q': '',
    'Key_W': '',
    'Key_E': ''
}


def report_contents(study_info, state):
    # accNo, chartNo, examName = study_info
    side, pressed_valid_key = state

    info = 'Accession Number: {}\r\nChart Number: {}\r\nExamination Name: {}\r\n'.format(*study_info)
    report = report_key_mapping[pressed_valid_key]
    if side in report:
        report = report[side]
    return info + report

class ImageLabel(QLabel):
    show_image_sig = pyqtSignal()
    clear_sig = pyqtSignal()
    load_path_sig = pyqtSignal([str])

    def __init__(self, *args, parent=None):
        super(ImageLabel, self).__init__(*args)
        self.border_state = False
        self.setMouseTracking(True)
        self.id = 0
        self.parent = parent
        self.load_files = None
        self.loaded = OrderedDict()
        self.image_ind = -1
        self.study_info = None
        self.is_reported = False
        self.border_toggle(False)
        self.setFrameShape(QFrame.Panel)

        self.submit_lock = threading.Lock()
        self.show_lock = threading.Lock()

        self.show_image_sig.connect(self.show_image)

        self.clear_sig.connect(self.clear)
        self.load_path_sig.connect(self.load_path)

    def submit_state(self):
        if Qt.LeftButton:
            if Qt.RightButton:
                side = 'b'
            else:
                side = 'l'
        elif Qt.RightButton:
            side = 'r'
        else:
            side = 'b'

        return (side, self.parent.pressed_valid_key)

    def submit(self):
        if self.is_reported:
            return
        try:
            self.submit_lock.acquire()
            submit_state = self.submit_state()

            for k, v in self.loaded.items():
                image_dir = Path(k).parent
                accNo_chartNo = image_dir.name
                report_txt = Path(self.parent.app.report_dir).joinpath(accNo_chartNo + '.txt')
                break

            if IS_DEBUG:
                logging.debug(
                    'Image label #{}: \nstudy_info: {}, state: {}'.format(self.id, self.study_info, submit_state))
            else:
                with open(str(report_txt.resolve()), 'w', encoding='utf-8') as f:
                    f.write(report_contents(self.study_info, submit_state))

        except Exception as e:
            logging.error(e)
        finally:
            self.load_files = None
            self.study_info = None
            self.loaded = OrderedDict()
            self.image_ind = 0
            self.clear_sig.emit()
            self.is_reported=True
            self.submit_lock.release()

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
        # logging.debug('Enter label #{}'.format(self.id))
        self.parent.active_image_label_id = self.id
        self.border_toggle(True)
        logging.debug('enter at {}'.format(time.time()))
        if self.parent.pressed_valid_key:
            self.submit()

    def leaveEvent(self, event):
        # logging.debug('Leave label #{}'.format(self.id))
        self.border_toggle(False)

    def load_path(self, path=''):
        if path is None:
            self.load_files = None
            self.loaded = OrderedDict()
            return
        elif path == '':
            path = self.load_files



        for f in sorted(list(glob.glob(path)), key = lambda p: Path(p).stat().st_ctime):
            f = Path(f)
            file_name = f.name
            if file_name in self.loaded:
                continue
            self.loaded[file_name] = cv2.imread(str(f.resolve()), cv2.IMREAD_GRAYSCALE)

        self.load_files = path

        th = threading.Timer(0.5, self.load_path)
        th.start()
        self.load_path_task = th

        if self.image_ind == -1:
            self.next_image()

    def show_image(self):
        try:
            self.show_lock.acquire()
            image = list(self.loaded.items())[self.image_ind][1]
            w = self.width()
            h = self.height()
            px = QPixmap.fromImage(
                QImage(image.data, image.shape[1], image.shape[0], image.shape[1], QImage.Format_Indexed8)
                    .scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            self.setPixmap(px)
            self.show()
        except Exception as e:
            logging.error(e)
            self.clear_sig.emit()
        finally:
            self.show_lock.release()

    def next_image(self):
        l = len(self.loaded)
        if l <= 1:
            return
        self.image_ind = (self.image_ind + 1) % l
        self.show_image_sig.emit()

    def prior_image(self):
        l = len(self.loaded)
        if l <= 1:
            return
        self.image_ind = (self.image_ind - 1 + l) % l
        self.show_image_sig.emit()


class ImageViewer(QMainWindow):
    def __init__(self, app=None):
        super(ImageViewer, self).__init__()

        self.setStyleSheet('background-color: black;')
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)

        self.app = app
        self.pressed_valid_key = ''
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

                l_id = self.create_image_labels(self.rows, self.cols, m, (l_w, l_h), (x, y), l_id)
        else:
            self.screen_count = 0
            m = monitors[0]
            x, y, w, h = m.x, m.y, m.width, m.height
            l_w = int(m.width / self.cols)
            l_h = int(m.height / self.rows)

            self.create_image_labels(self.rows, self.cols, m, (l_w, l_h), (x, y), l_id)

        self._base_counts = np.array((self.cols * self.rows, self.cols), dtype=np.int)

        self.setFixedSize(w, h)
        self.move(x, y)
        self.show_enable()

    @property
    def active_image_label(self):
        return self.image_labels[self.active_image_label_id]

    def keyPressEvent(self, event):
        for k in report_key_mapping:
            if getattr(Qt, k) == event.key():
                self.pressed_valid_key = k
                self.active_image_label.submit()
                break

    def keyReleaseEvent(self, event):
        try:
            if getattr(Qt, self.pressed_valid_key) == event.key():
                self.pressed_valid_key = ''
        except:
            pass

    def dispatch_study(self):

        q = self.app.study_queue
        df = self.app.ris_data
        for label in self.image_labels:
            if label.load_files:
                continue
            while 1:
                try:
                    accNo, chartNo, examName = q.get()
                except Empty:
                    th = threading.Timer(1.0, self.dispatch_study)
                    th.start()
                    self.dispatch_study_task = th
                    return
                accNo_chartNo = accNo + ' ' + chartNo

                self.app.df_lock.acquire()

                if Path(self.app.folder_path).joinpath(accNo_chartNo + '.txt').exists():
                    df.iloc[np.where(df['照會單號'] == accNo)]['reported'] = 1
                    continue
                label.study_info = (accNo, chartNo, examName)
                label.load_path_sig.emit(str(Path(self.app.folder_path).joinpath(accNo_chartNo).joinpath(accNo_chartNo + ' *.jpeg')))
                df.iloc[np.where(df['照會單號'] == accNo)]['loaded'] = 1
                self.app.df_lock.release()
                break

        th = threading.Timer(1.0, self.dispatch_study)
        th.start()
        self.dispatch_study_task = th

    def wheelEvent(self, event):
        d = event.delta()

        if d < 0:
            if IS_DEBUG:
                self.active_image_label.setText(self.active_image_label.text() + '+')
            self.active_image_label.next_image()

        elif d > 0:
            if IS_DEBUG:
                self.active_image_label.setText(self.active_image_label.text()[:-1])
            self.active_image_label.prior_image()

    def id2coord(self, image_label_id):
        which_screen, remainder = np.divmod(image_label_id, self._base_counts[0])
        which_row, which_col = np.divmod(remainder, self._base_counts[1])
        return which_screen, which_row, which_col

    def create_image_labels(self, rows, cols, monitor, label_w_h, win_pos, init_id):
        m = monitor
        l_w, l_h = label_w_h
        w_x, w_y = win_pos

        for r in range(rows):
            for c in range(cols):
                imageLabel = ImageLabel(self, parent=self)
                imageLabel.setStyleSheet('background-color: black;')
                imageLabel.setFixedSize(l_w, l_h)
                imageLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                # imageLabel.setScaledContents(True)
                imageLabel.setGeometry(m.x + c * l_w - w_x, m.y + r * l_h - w_y, l_w, l_h)
                imageLabel.setAlignment(Qt.AlignCenter)
                imageLabel.fixedWidth = l_w
                imageLabel.fixedHeight = l_h
                imageLabel.id = init_id

                init_id += 1
                # if IS_DEBUG:
                #     imageLabel.setFont(QFont("Verdana", 50, QFont.Normal))
                #     imageLabel.setStyleSheet('background-color: black; color: rgba(255,255,255,100); ')
                #     imageLabel.setText('({},{})'.format(r + 1, c + 1))
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
        self.df_lock = threading.Lock()
        self.load_clipboard()
        self.viewer = ImageViewer(app=self)

        self.study_queue = Queue()
        self.scan_dir()
        th = threading.Timer(1.0, self.viewer.dispatch_study)
        th.start()
        self.viewer.dispatch_study_task = th

    def scan_dir(self):
        self.df_lock.acquire()
        df = self.ris_data
        for i, row in df.iterrows():
            if not ((row['狀態']!='已發報告') and (row['queued']=='0') and (row['reported']=='0')):
                continue

            accNo, chartNo, examName = str(row['照會單號']), str(row['病歷號']), str(row['檢查名稱'])
            accNo_chartNo = '{} {}'.format(accNo, chartNo)
            if not self.folder_path.joinpath(accNo_chartNo).joinpath('{} 1.jpeg'.format(accNo_chartNo)).exists():
                continue
            if self.folder_path.joinpath('{}.txt'.format(accNo_chartNo)).exists():
                self.ris_data[i, 'reported'] = True
                continue

            self.study_queue.put((accNo, chartNo, examName))
            self.ris_data.iloc[i]['queued'] = 1

        th = threading.Timer(1.0, self.scan_dir)
        th.start()
        self.scan_dir_task = th
        self.df_lock.release()

    def load_clipboard(self):
        logging.info('Wait RIS data copied ... ')
        while 1:
            try:
                clipboard = QApplication.clipboard().text()

                assert '病歷號' in clipboard
                assert '報告醫師' in clipboard
                assert '照會單號' in clipboard
                assert '檢查名稱' in clipboard
                assert '狀態' in clipboard

                data = np.array([[cell for cell in row.split('\t')] for row in clipboard.splitlines()])

                data[0, 0] = 'queued'
                data[1:, 0] = 0

                data = np.insert(data, 0, np.repeat(0, data.shape[0]), axis=1)
                data[0, 0] = 'reported'

                data = np.insert(data, 0, np.repeat(0, data.shape[0]), axis=1)
                data[0, 0] = 'loaded'

                self.ris_data = df.DataFrame(data[1:], columns=data[0])
                return

            except Exception as e:
                # logging.info('RIS data error: \n' + str(e))
                # self.quit()
                logging.debug('Waiting for RIS data...')
                time.sleep(1)



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    IS_DEBUG = logging.getLogger().isEnabledFor(logging.DEBUG)
    app = ImageViewerApp(sys.argv, Path(getMyDocPath()).joinpath('feedRIS'))
    sys.exit(app.exec_())

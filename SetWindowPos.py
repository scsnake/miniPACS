import win32con, win32api, win32gui, ctypes, ctypes.wintypes


# def SetWindowPos(hwnd, insertAfter=0, x=None, y=None, w=None, h=None, flags=0):

def insertAfter(hwnd, hwndInsertAfter):
    rect = win32gui.GetWindowRect(hwnd)
    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    win32gui.SetWindowPos(
        hwnd, hwndInsertAfter, x, y, w, h, 0)


def moveToTop(hwnd):
    win32gui.SetWindowPos(hwnd,
                          win32con.HWND_TOPMOST,
                          # = always on top. only reliable way to bring it to the front on windows
                          0, 0, 0, 0,
                          win32con.SWP_DRAWFRAME | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)
    win32gui.SetWindowPos(hwnd,
                          win32con.HWND_NOTOPMOST,  # disable the always on top, but leave window at its top position
                          0, 0, 0, 0,
                          win32con.SWP_DRAWFRAME | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)

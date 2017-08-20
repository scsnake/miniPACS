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
import win32con, win32api, win32gui, ctypes, ctypes.wintypes
import threading


class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_wchar_p)
    ]


PCOPYDATASTRUCT = ctypes.POINTER(COPYDATASTRUCT)


def Send_WM_COPYDATA(hwnd, str, dwData=0):
    cds = COPYDATASTRUCT()
    cds.dwData = dwData
    cds.cbData = ctypes.sizeof(ctypes.create_unicode_buffer(str))
    cds.lpData = ctypes.c_wchar_p(str)

    return ctypes.windll.user32.SendMessageW(hwnd, win32con.WM_COPYDATA, 0, ctypes.byref(cds))


class WM_COPYDATA_Listener:
    def __init__(self, receiver=None):
        message_map = {
            win32con.WM_COPYDATA: self.__OnCopyData
        }
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = message_map
        wc.lpszClassName = 'MyWindowClass'
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        classAtom = win32gui.RegisterClass(wc)
        self.hwnd = win32gui.CreateWindow(
            classAtom,
            "WM_COPYDATA_Listener",
            0,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None
        )
        self.receiver = receiver
        threading.Thread(target=win32gui.PumpMessages).start()
        # print self.hwnd

    def OnCopyData(self, *args, **kwargs):
        for k in ['hwnd', 'msg', 'wparam', 'lparam', 'dwData', 'cbData', 'lpData']:
            print kwargs[k]

    def __OnCopyData(self, hwnd, msg, wparam, lparam):
        pCDS = ctypes.cast(lparam, PCOPYDATASTRUCT)

        t = threading.Thread(target=self.OnCopyData if self.receiver is None else self.receiver,
                             kwargs={'hwnd': hwnd, 'msg': msg,
                                     'wparam': wparam, 'lparam': lparam,
                                     'dwData': pCDS.contents.dwData,
                                     'cbData': pCDS.contents.cbData,
                                     'lpData': pCDS.contents.lpData})
        t.start()

        return 1




if __name__ == '__main__':
    COPYDATA_Listener = WM_COPYDATA_Listener()

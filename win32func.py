import ctypes
import ctypes.wintypes
import threading
import win32api
import win32gui

import win32con


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
    def __init__(self, receiver=None, title="WM_COPYDATA_Listener", useThread=True):
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
            title,
            win32con.WS_OVERLAPPEDWINDOW,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None
        )
        # win32gui.ShowWindow(self.hwnd, win32con.SW_SHOWNORMAL)
        # win32gui.UpdateWindow(self.hwnd)
        self.receiver = self.OnCopyData if receiver is None else receiver
        if useThread:
            threading.Thread(target=win32gui.PumpMessages).start()


        # print self.hwnd
    def quit(self):
        pass

    def start(self):
        win32gui.PumpMessages()
    def OnCopyData(self, *args, **kwargs):
        for k in ['hwnd', 'msg', 'wparam', 'lparam', 'dwData', 'cbData', 'lpData']:
            print(kwargs[k])

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
    #
    # def wndProc(hWnd, message, wParam, lParam):
    #
    #     if message == win32con.WM_PAINT:
    #         hDC, paintStruct = win32gui.BeginPaint(hWnd)
    #
    #         rect = win32gui.GetClientRect(hWnd)
    #         win32gui.DrawText(
    #             hDC,
    #             'Hello send by Python via Win32!',
    #             -1,
    #             rect,
    #             win32con.DT_SINGLELINE | win32con.DT_CENTER | win32con.DT_VCENTER)
    #
    #         win32gui.EndPaint(hWnd, paintStruct)
    #         return 0
    #
    #     elif message == win32con.WM_DESTROY:
    #         print('Being destroyed')
    #         win32gui.PostQuitMessage(0)
    #         return 0
    #
    #     else:
    #         return win32gui.DefWindowProc(hWnd, message, wParam, lParam)


if __name__ == '__main__':
    # Send_WM_COPYDATA(1973580,'test')
    COPYDATA_Listener = WM_COPYDATA_Listener()
    print(COPYDATA_Listener.hwnd)

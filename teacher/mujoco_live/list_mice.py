"""List the mouse devices Windows Raw Input can see (each one can drive its own arm)."""
import ctypes
from ctypes import wintypes as wt

user32 = ctypes.WinDLL("user32")


class RAWINPUTDEVICELIST(ctypes.Structure):
    _fields_ = [("hDevice", wt.HANDLE), ("dwType", wt.DWORD)]


n = wt.UINT(0)
user32.GetRawInputDeviceList(None, ctypes.byref(n), ctypes.sizeof(RAWINPUTDEVICELIST))
arr = (RAWINPUTDEVICELIST * n.value)()
user32.GetRawInputDeviceList(arr, ctypes.byref(n), ctypes.sizeof(RAWINPUTDEVICELIST))
RIDI_DEVICENAME = 0x20000007
for d in arr:
    if d.dwType != 0:  # 0 = mouse
        continue
    size = wt.UINT(0)
    user32.GetRawInputDeviceInfoW(d.hDevice, RIDI_DEVICENAME, None, ctypes.byref(size))
    buf = ctypes.create_unicode_buffer(size.value + 1)
    user32.GetRawInputDeviceInfoW(d.hDevice, RIDI_DEVICENAME, buf, ctypes.byref(size))
    print(hex(d.hDevice or 0), buf.value)

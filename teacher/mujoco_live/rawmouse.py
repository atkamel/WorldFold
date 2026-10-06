"""Read several physical mice separately on Windows (Raw Input API).
Windows merges all mice into one cursor; WM_INPUT still tells which device each move/click came from.
"""
import ctypes, threading
from ctypes import wintypes as wt

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32")

WM_INPUT = 0x00FF
RID_INPUT = 0x10000003
RIM_TYPEMOUSE = 0
RIDEV_INPUTSINK = 0x00000100
HWND_MESSAGE = wt.HWND(-3)
LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)

user32.DefWindowProcW.argtypes = [wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID]
user32.CreateWindowExW.restype = wt.HWND
user32.GetRawInputData.argtypes = [wt.HANDLE, wt.UINT, wt.LPVOID, ctypes.POINTER(wt.UINT), wt.UINT]
user32.GetRawInputData.restype = wt.UINT


class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wt.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                ("hInstance", wt.HINSTANCE), ("hIcon", wt.HICON), ("hCursor", wt.HANDLE), ("hbrBackground", wt.HBRUSH),
                ("lpszMenuName", wt.LPCWSTR), ("lpszClassName", wt.LPCWSTR)]


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [("usUsagePage", wt.USHORT), ("usUsage", wt.USHORT), ("dwFlags", wt.DWORD), ("hwndTarget", wt.HWND)]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [("dwType", wt.DWORD), ("dwSize", wt.DWORD), ("hDevice", wt.HANDLE), ("wParam", wt.WPARAM)]


class _BTN(ctypes.Structure):
    _fields_ = [("usButtonFlags", wt.USHORT), ("usButtonData", wt.USHORT)]


class _BTNU(ctypes.Union):
    _fields_ = [("ulButtons", wt.ULONG), ("b", _BTN)]


class RAWMOUSE(ctypes.Structure):
    _fields_ = [("usFlags", wt.USHORT), ("u", _BTNU), ("ulRawButtons", wt.ULONG),
                ("lLastX", wt.LONG), ("lLastY", wt.LONG), ("ulExtraInformation", wt.ULONG)]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]


class MiceReader:
    """Per-device state: dx, dy (accumulated since last take()), wheel notches, buttons held, click events."""

    def __init__(self):
        self.lock = threading.Lock()
        self.dev = {}  # handle -> dict
        self.ok = False
        threading.Thread(target=self._run, daemon=True).start()

    def _state(self, h):
        if h not in self.dev:
            self.dev[h] = dict(dx=0, dy=0, wheel=0, left=False, right=False, middle=False, events=[])
        return self.dev[h]

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            size = wt.UINT(0)
            user32.GetRawInputData(lparam, RID_INPUT, None, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER))
            buf = ctypes.create_string_buffer(size.value)
            if user32.GetRawInputData(lparam, RID_INPUT, buf, ctypes.byref(size), ctypes.sizeof(RAWINPUTHEADER)) == size.value:
                ri = ctypes.cast(buf, ctypes.POINTER(RAWINPUT)).contents
                if ri.header.dwType == RIM_TYPEMOUSE:
                    m = ri.mouse
                    f = m.u.b.usButtonFlags
                    with self.lock:
                        s = self._state(int(ri.header.hDevice or 0))
                        if not (m.usFlags & 0x01):  # relative motion
                            s["dx"] += m.lLastX
                            s["dy"] += m.lLastY
                        if f & 0x0001: s["left"] = True; s["events"].append("left_down")
                        if f & 0x0002: s["left"] = False; s["events"].append("left_up")
                        if f & 0x0004: s["right"] = True; s["events"].append("right_down")
                        if f & 0x0008: s["right"] = False; s["events"].append("right_up")
                        if f & 0x0010: s["middle"] = True; s["events"].append("middle_down")
                        if f & 0x0020: s["middle"] = False; s["events"].append("middle_up")
                        if f & 0x0400: s["wheel"] += ctypes.c_short(m.u.b.usButtonData).value / 120.0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def register(self):
        """(Re)claim raw mouse input for our hidden window. Windows allows ONE raw-mouse listener per
        program; SDL/pygame takes it over in relative-mouse mode, so call this after display changes."""
        if not getattr(self, "hwnd", None):
            return False
        rid = RAWINPUTDEVICE(0x01, 0x02, RIDEV_INPUTSINK, self.hwnd)
        self.ok = bool(user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)))
        return self.ok

    def listening(self):
        """True if raw mouse input is currently delivered to our window (not taken over by someone else)."""
        n = wt.UINT(0)
        user32.GetRegisteredRawInputDevices(None, ctypes.byref(n), ctypes.sizeof(RAWINPUTDEVICE))
        if not n.value:
            return False
        arr = (RAWINPUTDEVICE * n.value)()
        user32.GetRegisteredRawInputDevices(arr, ctypes.byref(n), ctypes.sizeof(RAWINPUTDEVICE))
        return any(d.usUsagePage == 1 and d.usUsage == 2 and d.hwndTarget == self.hwnd for d in arr)

    def _run(self):
        self._proc = WNDPROC(self._wndproc)
        hinst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSW(lpfnWndProc=self._proc, hInstance=hinst, lpszClassName="FoldRawMice")
        user32.RegisterClassW(ctypes.byref(wc))
        self.hwnd = user32.CreateWindowExW(0, "FoldRawMice", "", 0, 0, 0, 0, 0, HWND_MESSAGE, None, hinst, None)
        self.register()
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    @staticmethod
    def device_name(h):
        """Windows device path, e.g. '\\\\?\\HID#VID_3151&PID_3020...' for a USB mouse."""
        RIDI_DEVICENAME = 0x20000007
        size = wt.UINT(0)
        user32.GetRawInputDeviceInfoW(wt.HANDLE(h), RIDI_DEVICENAME, None, ctypes.byref(size))
        if not size.value:
            return ""
        buf = ctypes.create_unicode_buffer(size.value + 1)
        user32.GetRawInputDeviceInfoW(wt.HANDLE(h), RIDI_DEVICENAME, buf, ctypes.byref(size))
        return buf.value

    @classmethod
    def is_external_mouse(cls, h):
        """USB/Bluetooth mice have a vendor id (VID_) in their path; laptop touchpads (I2C HID) don't."""
        return bool(h) and "VID_" in cls.device_name(h).upper()

    def take(self):
        """Return {device: (dx, dy, wheel, left_held, right_held, events)} and clear the accumulators.
        events also contains "middle_held" while the wheel button is held down."""
        with self.lock:
            out = {}
            for h, s in self.dev.items():
                out[h] = (s["dx"], s["dy"], s["wheel"], s["left"], s["right"], list(s["events"]) + (["middle_held"] if s["middle"] else []))
                s["dx"] = s["dy"] = 0
                s["wheel"] = 0
                s["events"].clear()
            return out


if __name__ == "__main__":
    import time
    r = MiceReader()
    time.sleep(0.3)
    print("raw input registered:", r.ok, "- move/click your mice for 8 s")
    t = time.time()
    while time.time() - t < 8:
        for h, v in r.take().items():
            if any(v[:3]) or v[5]:
                print(hex(h), v)
        time.sleep(0.25)

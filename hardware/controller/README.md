# PowerA EnWired USB controller

Python user-space reader for the Lavender Swirl Xbox Series X|S controller (`20d6:2026`). It reads Microsoft Game Input Protocol (GIP) over USB and makes the state available to Python programs.

## Run on macOS

Run these commands from the WorldFold repository root. Use the project's Python environment for the pip and reader commands.

```sh
brew install libusb
python -m pip install -r hardware/controller/requirements.txt
python -m hardware.controller
```

Move a stick or press a button to get JSON lines on stdout. The controller sends input when its state changes. Stop with Ctrl-C or pass `--count N` to stop after N reports. If the controller appears in System Information but the reader cannot find it, run from macOS Terminal. Sandboxed launchers can hide USB devices from libusb.

Pass `--raw` to print the USB packets as hex. Pass a quoted hex line to `--decode HEX` to replay an individual input report without the controller, libusb, or PyUSB. Offline decoding of a Guide packet starts with a neutral state. The live reader retains the most recent stick, trigger, and button values across Guide events.

## Python interface

Use `hardware.controller.controller.read_controller()` as a context manager. It yields an iterator of `Reading` objects with `raw` bytes and an immutable `state`. The state is `None` for status, startup placeholder, and other non-input reports. The USB interface is released when the context closes. For offline use, `hardware.controller.gip.parse_input_report(packet)` returns an `InputReport` whose `state(previous)` method applies the report to the prior state.

Sticks are signed 16-bit values in the controller's native orientation. Triggers use the native 0–1023 range. `buttons` lists currently pressed buttons, and `guide` is the Xbox button state. Values are raw, with no dead zone applied. See the [Linux xpad driver](https://github.com/torvalds/linux/blob/master/drivers/input/joystick/xpad.c) and the [SDL GIP reader](https://github.com/libsdl-org/SDL/blob/main/src/joystick/hidapi/SDL_hidapi_xboxone.c) for the protocol references.

#!/usr/bin/env python3

from evdev import InputDevice, UInput, ecodes, list_devices
import sys

# 自动寻找真正的 Pro Controller，排除 IMU
source = None

for path in list_devices():
    dev = InputDevice(path)

    if dev.name in (
        "Pro Controller",
        "Nintendo Co., Ltd. Pro Controller",
        "Nintendo Switch Pro Controller",
    ):
        source = dev
        break

if source is None:
    print("ERROR: Cannot find Switch Pro Controller")
    print("Available input devices:")

    for path in list_devices():
        dev = InputDevice(path)
        print(f"  {path}: {dev.name}")

    sys.exit(1)

print(f"Physical controller: {source.path}")
print(f"Name: {source.name}")

caps = source.capabilities(absinfo=True)

# 只复制 joystick 所需要的按钮和模拟轴
virtual_caps = {}

if ecodes.EV_KEY in caps:
    virtual_caps[ecodes.EV_KEY] = caps[ecodes.EV_KEY]

if ecodes.EV_ABS in caps:
    virtual_caps[ecodes.EV_ABS] = caps[ecodes.EV_ABS]

ui = UInput(
    virtual_caps,
    name="Virtual Switch Pro Controller",
    bustype=ecodes.BUS_USB,
    vendor=0x057e,
    product=0x2009,
    version=0x8001,
)

print(f"Virtual input device: {ui.device.path}")
print("Relay running. Press Ctrl+C to stop.")

try:
    for event in source.read_loop():

        if event.type in (ecodes.EV_KEY, ecodes.EV_ABS):
            ui.write(event.type, event.code, event.value)

        elif event.type == ecodes.EV_SYN:
            ui.syn()

except KeyboardInterrupt:
    print("\nStopping bridge.")

finally:
    ui.close()
    source.close()

# Sim-to-Sim Utilities

This directory contains additional utilities and configuration used by the
MuJoCo Sim-to-Sim deployment.

## Structure

```text
sim2sim/
├── switch_bridge/
│   └── switch_to_js.py
│
├── dds_test/
│   ├── CMakeLists.txt
│   └── ...
│
└── unitree_mujoco/
    ├── config.yaml
    └── physics_joystick.h
```

## Switch Controller Bridge

Nintendo Switch Pro Controller may be detected by Linux as an `evdev`
device without automatically creating:

```text
/dev/input/js0
```

The bridge converts the physical controller input into a virtual Linux
joystick:

```text
Switch Pro Controller
        ↓
/dev/input/eventXX
        ↓
switch_to_js.py
        ↓
Virtual Switch Pro Controller
        ↓
/dev/input/js0
```

Run:

```bash
cd sim2sim/switch_bridge
sudo python3 switch_to_js.py
```

Then verify:

```bash
jstest /dev/input/js0
```

---

## DDS Test

The DDS test subscribes only to:

```text
rt/lowstate
```

and is intended to verify the communication path before enabling robot
control:

```text
MuJoCo
  ↓
unitree_mujoco
  ↓
LowState
  ↓
CycloneDDS
  ↓
DDS Test
```

Build:

```bash
cd sim2sim/dds_test

mkdir build
cd build

cmake .. -DCMAKE_PREFIX_PATH=/opt/unitree_robotics
make -j4
```

Run the generated DDS test executable while `unitree_mujoco` is running.

---

## unitree_mujoco Configuration

The files under:

```text
sim2sim/unitree_mujoco/
```

record the configuration used by this project.

Important DDS settings:

```yaml
domain_id: 1
interface: "lo"
```

Simulation therefore uses:

```text
DDS Domain 1
+
Linux loopback interface
```

to keep simulation communication isolated from a real Go2.

Joystick configuration:

```yaml
use_joystick: 1
joystick_type: "switch"
joystick_device: "/dev/input/js0"
```

`physics_joystick.h` contains the tested Nintendo Switch Pro Controller
button and axis mapping.

These files are reference/override files for the external
`unitree_mujoco` repository.
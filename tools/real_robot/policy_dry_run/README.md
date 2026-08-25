# Go2 Real Policy Dry Run (Task 5)

First-contact tool for running a trained RL policy against **real Go2 sensor
data** without sending a single byte of command. The policy truly consumes live
`rt/lowstate` (12 joint states + IMU), runs the full production deploy stack,
and prints the `q_des` it *would* send.

```text
real Go2 -> rt/lowstate (domain 0)
  -> BaseArticulation::update()            (deploy/include/unitree_articulation.h)
  -> ObservationManager                    (30-frame history -> 1350-D "obs")
  -> OrtRunner                             (exported/policy.onnx -> 12-D action)
  -> ActionManager / JointPositionAction   (x 0.25 + default_joint_pos) == q_des
  -> terminal print                        (never LowCmd)
```

## Safety contract

This program **never creates a LowCmd publisher**. Only the subscription side of
the SDK (`go2_sub.h`) is compiled in. Verify on the built binary:

```bash
nm -C build/go2_policy_dry_run | grep -iE "ChannelPublisher|publisher::LowCmd|RealTimePublisher"
# -> must print nothing
strings build/go2_policy_dry_run | grep -xE "rt/[a-z]+"
# -> only: rt/lowstate
```

(The `LowCmd_` IDL type-support symbols visible in `strings` come from SDK
header includes and are never used — no endpoint is created.)

## Build

```bash
cd tools/real_robot/policy_dry_run
cmake -B build -S .
cmake --build build -j
```

## 1) Offline pre-flight (no robot, no DDS)

Validates the whole chain except DDS with a mock standing pose measured on the
real robot:

```bash
./build/go2_policy_offline_check            # default: ../../../pretrained/example
```

Expect `Observation: 1350`, `Action: 12`, `all finite: YES`, `RESULT: PASS`.
Note: with the frozen mock (no physics), the policy's actions drift into a
limit cycle — that is expected open-loop behavior, not a model defect.

## 2) Real robot dry run (READ-ONLY)

```bash
./build/go2_policy_dry_run --network enp6s0
```

Options:

| option | default | meaning |
|---|---|---|
| `--network, -n` | `""` | DDS interface (real Go2: `enp6s0`) |
| `--domain` | `0` | DDS domain (real Go2: 0; MuJoCo sim2sim used 1) |
| `--policy, -p` | `<repo>/pretrained/example` | policy dir (`exported/policy.onnx` + `params/deploy.yaml`) |
| `--print-period` | `0.5` | seconds between status prints |
| `--steps` | `0` | stop after N steps (0 = until Ctrl-C) |
| `--use-joystick` | off | read commands from remote; **default: forced zero commands** |

What to check in the output:

- `Observation: 1350`, `Action: 12` in the banner
- `projected_gravity` ≈ `[0, 0, -1]` when the robot is level
- `q (policy order)` close to the measured standing pose
  (hips `±0.0x`, thighs `~0.66`, calves `~-1.36`)
- `q_des` finite and in a plausible range — reject NaN, `17 rad`, hip/calf swaps
- `rate ≈ 50 Hz`, `lowstate: OK`, `all finite: YES`

Ctrl-C stops the loop and prints a summary. Nothing is ever sent to the robot.

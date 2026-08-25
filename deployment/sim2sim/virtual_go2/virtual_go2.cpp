// Virtual Go2 — sim2sim stand-in for closed-loop rehearsal (domain 1, lo).
//
//   - publishes rt/lowstate at 500 Hz: level IMU, joints tracking rt/lowcmd
//     with a first-order lag (k = 15 1/s), lying initial pose
//   - subscribes rt/lowcmd (read-only cache, never replies with commands)
//   - injects scripted remote button presses into wireless_remote so the
//     controller FSM can be driven through Passive -> FixStand -> Velocity
//     without the physical remote
//
// Together with go2_ctrl --domain 1 --network lo [--pc-command] this exercises
// the full chain: FSM transitions, warm-up, blend, safety gate, PC commands.
//
// Usage:
//   ./virtual_go2 [--lt-a 2.0] [--start 6.0] [--lt-b 9999] [--hz 500]

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/robots/go2/go2.h>

using LowStatePub = unitree::robot::go2::publisher::LowState;
using LowCmdSub   = unitree::robot::go2::subscription::LowCmd;

static std::atomic<bool> g_run{true};
static void on_signal(int) { g_run = false; }

int main(int argc, char** argv)
{
    double t_lt_a = 2.0, t_start = 6.0, t_lt_b = 1e9;
    int hz = 500;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> double { return (i + 1 < argc) ? std::atof(argv[++i]) : 0.0; };
        if      (a == "--lt-a")  t_lt_a  = next();
        else if (a == "--start") t_start = next();
        else if (a == "--lt-b")  t_lt_b  = next();
        else if (a == "--hz")    hz      = static_cast<int>(next());
    }

    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("==============================================\n"
                " Virtual Go2 (domain 1, interface lo)\n"
                " publish rt/lowstate @ %d Hz, subscribe rt/lowcmd\n"
                " script: LT+A @ %.1fs, START @ %.1fs, LT+B @ %.1fs\n"
                "==============================================\n",
                hz, t_lt_a, t_start, t_lt_b);

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    unitree::robot::ChannelFactory::Instance()->Init(1, "lo");

    auto lowstate = std::make_shared<LowStatePub>();  // rt/lowstate
    auto lowcmd   = std::make_shared<LowCmdSub>();    // rt/lowcmd

    // joint state: start lying ([0, 1.36, -2.65] per leg), track cmd with lag
    float q[12], dq[12] = {0};
    long trylock_ok_ = 0;
    for (int leg = 0; leg < 4; ++leg)
    {
        q[3 * leg + 0] = 0.0f;
        q[3 * leg + 1] = 1.36f;
        q[3 * leg + 2] = -2.65f;
    }

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    const std::chrono::duration<double> dt(1.0 / hz);
    const float track_k = 15.0f;
    auto next_tick = t0 + dt;
    long tick = 0;

    while (g_run)
    {
        const double t = std::chrono::duration<double>(clock::now() - t0).count();

        // scripted remote presses, each held 0.5 s
        // BtnUnion bit order: R1 L1 Start Select R2 L2 f1 f2 | A B X Y up right down left
        unsigned btn = 0;
        const auto held = [&](double ts) { return t >= ts && t < ts + 0.5; };
        if (held(t_lt_a))  btn = (1u << 5) | (1u << 8);  // L2 + A -> FixStand
        if (held(t_start)) btn = (1u << 2);              // Start  -> RL
        if (held(t_lt_b))  btn = (1u << 5) | (1u << 9);  // L2 + B -> Passive

        // joints track the latest lowcmd target (first-order lag)
        float track_err = 0.0f;
        float q_cmd_dbg = 0.0f;
        {
            std::lock_guard<std::mutex> lk(lowcmd->mutex_);
            const float alpha = 1.0f - std::exp(-static_cast<float>(dt.count()) * track_k);
            q_cmd_dbg = lowcmd->msg_.motor_cmd()[1].q();
            for (int i = 0; i < 12; ++i)
            {
                const float e = lowcmd->msg_.motor_cmd()[i].q() - q[i];
                q[i] += alpha * e;
                dq[i] = e * track_k;
                track_err = std::max(track_err, std::fabs(e));
            }
        }

        bool locked = lowstate->trylock();
        trylock_ok_ += locked;
        if (locked)
        {
            auto& msg = lowstate->msg_;

            // level IMU, zero gyro
            msg.imu_state().quaternion()[0] = 1.0f;
            msg.imu_state().quaternion()[1] = 0.0f;
            msg.imu_state().quaternion()[2] = 0.0f;
            msg.imu_state().quaternion()[3] = 0.0f;
            for (int i = 0; i < 3; ++i)
            {
                msg.imu_state().gyroscope()[i] = 0.0f;
                msg.imu_state().accelerometer()[i] = 0.0f;
                msg.imu_state().rpy()[i] = 0.0f;
            }

            for (int i = 0; i < 12; ++i)
            {
                msg.motor_state()[i].q() = q[i];
                msg.motor_state()[i].dq() = dq[i];
                msg.motor_state()[i].tau_est() = 0.0f;
            }

            // scripted remote (40-byte REMOTE_DATA_RX)
            unitree::common::REMOTE_DATA_RX rx{};
            rx.RF_RX.btn.value = static_cast<uint16_t>(btn);
            std::memcpy(msg.wireless_remote().data(), rx.buff, 40);

            lowstate->unlockAndPublish();
        }

        if (++tick % hz == 0)
            std::printf("[virtual_go2] t=%4.0fs  q1=%.3f  q_cmd1=%.3f  err=%.3f  btn=0x%x  trylock=%ld/%ld\n",
                        t, q[1], q_cmd_dbg, track_err, btn, trylock_ok_, tick);

        std::this_thread::sleep_until(next_tick);
        next_tick += dt;
    }

    std::printf("[virtual_go2] stopped\n");
    return 0;
}

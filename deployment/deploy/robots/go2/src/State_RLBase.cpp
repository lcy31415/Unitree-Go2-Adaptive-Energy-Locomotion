#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

#include <chrono>
#include <cstdio>
#include <string>

// ---------------------------------------------------------------------------
// PC keyboard -> velocity command injection (opt-in via --pc-command)
//
// The keyboard may ONLY set vx/vy/wz: they are written into a merged joystick
// that feeds the policy's `velocity_commands` observation (vx=ly, vy=-lx,
// wz=-rx, clamped by deploy.yaml ranges). Nothing here can touch q_des,
// motor_cmd, gains or LowCmd, so the warm-up / blend / safety-gate chain stays
// fully in command. The physical remote keeps priority at all times: the
// LT+B -> Passive transition reads lowstate->joystick, not this object.
//
//   1: vx=+speed, 5 s     2: vx=-speed, 5 s
//   3: wz=+yaw,   5 s     4: wz=-yaw,   5 s
//   0: immediate zero
//
// Commands auto-zero on expiry (expire_time checked every update, never a
// blocking sleep) and are dropped when the RL state is left/re-entered
// (detected as a gap between update() calls).
// ---------------------------------------------------------------------------
class PcCommandArticulation : public unitree::BaseArticulation<LowState_t::SharedPtr>
{
public:
    using Base = unitree::BaseArticulation<LowState_t::SharedPtr>;
    using clock = std::chrono::steady_clock;

    PcCommandArticulation(LowState_t::SharedPtr lowstate, float lin_speed, float yaw_rate, float duration)
    : Base(lowstate), lin_speed_(lin_speed), yaw_rate_(yaw_rate), duration_(duration)
    {
        data.joystick = &merged_;   // policy velocity_commands reads merged axes
        merged_.lx.smooth = 1.0f;   // crisp steps, no remote-style axis smoothing
        merged_.ly.smooth = 1.0f;
        merged_.rx.smooth = 1.0f;
        last_update_ = clock::now();
    }

    void update() override
    {
        Base::update();  // IMU + joint states from real lowstate

        const auto now = clock::now();

        // resumed after a pause (RL state re-entered): drop any stale command
        if (std::chrono::duration<float>(now - last_update_).count() > 0.5f)
            active_ = false;
        last_update_ = now;

        handle_keys(now);

        if (active_ && now >= expire_)
        {
            active_ = false;
            std::printf("[PC CMD] timeout -> ZERO   (vx=0.000 vy=0.000 wz=0.000)\n");
        }
        else if (active_ && ++print_div_ % 25 == 0)  // 0.5 s at 50 Hz
        {
            std::printf("[PC CMD] %.1fs remaining\n",
                        std::chrono::duration<float>(expire_ - now).count());
        }

        if (active_)
        {
            merged_.ly( vx_);   // vx = ly
            merged_.lx(-vy_);   // vy = -lx
            merged_.rx(-wz_);   // wz = -rx
        }
        else
        {
            // no PC command: pass the physical remote through unchanged
            merged_.ly(lowstate->joystick.ly());
            merged_.lx(lowstate->joystick.lx());
            merged_.rx(lowstate->joystick.rx());
        }
    }

private:
    void handle_keys(const clock::time_point& now)
    {
        const std::string k = FSMState::keyboard ? FSMState::keyboard->key() : std::string();
        if (k == last_key_)
            return;  // edge-triggered only
        last_key_ = k;

        if      (k == "1") arm(now,  lin_speed_, 0.0f, 0.0f, "FORWARD");
        else if (k == "2") arm(now, -lin_speed_, 0.0f, 0.0f, "BACKWARD");
        else if (k == "3") arm(now, 0.0f, 0.0f,  yaw_rate_, "TURN LEFT");
        else if (k == "4") arm(now, 0.0f, 0.0f, -yaw_rate_, "TURN RIGHT");
        else if (k == "0")
        {
            active_ = false;
            std::printf("[PC CMD] STOP -> ZERO   (vx=0.000 vy=0.000 wz=0.000)\n");
        }
    }

    void arm(const clock::time_point& now, float vx, float vy, float wz, const char* name)
    {
        vx_ = vx; vy_ = vy; wz_ = wz;
        expire_ = now + std::chrono::duration_cast<clock::duration>(
                            std::chrono::duration<float>(duration_));
        active_ = true;
        print_div_ = 0;
        std::printf("[PC CMD] %-10s vx=%+.3f vy=%+.3f wz=%+.3f duration=%.1fs\n",
                    name, vx, vy, wz, duration_);
    }

    unitree::common::UnitreeJoystick merged_;
    const float lin_speed_, yaw_rate_, duration_;
    float vx_ = 0.0f, vy_ = 0.0f, wz_ = 0.0f;
    bool active_ = false;
    clock::time_point expire_{}, last_update_{};
    std::string last_key_;
    int print_div_ = 0;
};

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    std::shared_ptr<isaaclab::Articulation> articulation;
    if (param::vm.count("pc-command"))
    {
        const float lin = cfg["pc_cmd_speed"]    ? cfg["pc_cmd_speed"].as<float>()    : 0.1f;
        const float yaw = cfg["pc_cmd_yaw_rate"] ? cfg["pc_cmd_yaw_rate"].as<float>() : 0.3f;
        const float dur = cfg["pc_cmd_duration"] ? cfg["pc_cmd_duration"].as<float>() : 5.0f;
        articulation = std::make_shared<PcCommandArticulation>(FSMState::lowstate, lin, yaw, dur);
        spdlog::info("PC command enabled: [1] fwd  [2] back  [3] left  [4] right  [0] stop "
                     "({:.2f} m/s, {:.2f} rad/s, auto-zero after {:.1f} s)", lin, yaw, dur);
    }
    else
    {
        articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
    }

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // RL entry protocol and runtime safety limits (optional config keys)
    if (cfg["warmup_time"])           warmup_time_    = cfg["warmup_time"].as<float>();
    if (cfg["blend_time"])            blend_time_     = cfg["blend_time"].as<float>();
    if (cfg["safety_max_action_abs"]) max_action_abs_ = cfg["safety_max_action_abs"].as<float>();
    if (cfg["safety_max_q_des_dev"])  max_q_des_dev_  = cfg["safety_max_q_des_dev"].as<float>();

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

    // any safety violation flagged by run() -> leave RL immediately
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return safety_violation_.load(); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    const auto& map = env->robot->data.joint_ids_map;
    const int n = static_cast<int>(map.size());

    const auto q_rl = env->action_manager->processed_actions();  // policy order
    const auto raw  = env->alg->get_action();                    // policy order

    // ---- phase: warm-up (hold entry pose, history fills) -> blend -> RL ----
    const float t = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - t_enter_).count();
    float alpha = 1.0f;
    if (t < warmup_time_)
        alpha = 0.0f;
    else if (blend_time_ > 0.0f && t < warmup_time_ + blend_time_)
        alpha = (t - warmup_time_) / blend_time_;

    for (int i(0); i < n; ++i)
        q_cmd_[i] = (1.0f - alpha) * q_start_[i] + alpha * q_rl[i];

    // ---- safety gate: validate BEFORE anything reaches motor_cmd ----
    const char* reason = nullptr;
    if (lowstate->isTimeout())
    {
        reason = "lowstate timeout";
    }
    else
    {
        float max_abs = 0.0f;
        for (float a : raw)
        {
            if (!std::isfinite(a)) { reason = "non-finite policy action"; break; }
            max_abs = std::max(max_abs, std::fabs(a));
        }
        if (!reason && max_abs > max_action_abs_)
            reason = "policy action magnitude";

        if (!reason)
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            for (int i(0); i < n; ++i)
            {
                if (!std::isfinite(q_cmd_[i])) { reason = "non-finite q_des"; break; }
                const float q_real = lowstate->msg_.motor_state()[map[i]].q();
                if (std::fabs(q_cmd_[i] - q_real) > max_q_des_dev_)
                {
                    reason = "q_des deviation from measured q";
                    break;
                }
            }
        }
    }

    if (reason)
    {
        if (!violation_logged_)
        {
            spdlog::warn("State_{}: safety violation [{}] -> Passive",
                         getStateString(), reason);
            violation_logged_ = true;
        }
        safety_violation_ = true;

        // hold measured positions with current gains until Passive takes over
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        for (int i(0); i < n; ++i)
            q_cmd_[i] = lowstate->msg_.motor_state()[map[i]].q();
    }

    for (int i(0); i < n; ++i) {
        lowcmd->msg_.motor_cmd()[map[i]].q() = q_cmd_[i];
    }
}

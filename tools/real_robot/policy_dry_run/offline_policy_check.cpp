// Offline policy pre-flight check — NO DDS, NO robot.
//
// Runs the exact production deploy stack (deploy.yaml -> ManagerBasedRLEnv ->
// ObservationManager 30-frame history -> OrtRunner -> ActionManager) against a
// MOCK articulation filled with the real Go2 standing pose measured in task 4.
//
// Purpose: validate everything except the DDS subscription before touching the
// robot — YAML parsing, joint mapping, history layout (1350-D), ONNX load +
// inference, action post-processing, finiteness of q_des.

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

namespace fs = std::filesystem;

// Feeds the same fields BaseArticulation::update() produces, but from static
// mock values: level base, zero gyro, robot resting in its measured stand pose.
class MockArticulation : public isaaclab::Articulation
{
public:
    MockArticulation() { data.joystick = &joystick_; }

    void update() override
    {
        // Policy-order standing pose measured on the real Go2 (task 4):
        // hips [+0.03,-0.02,+0.07,-0.07], thighs ~0.66, calves ~-1.36
        static const float q[12] = {
            0.03f, -0.02f, 0.07f, -0.07f,
            0.66f, 0.66f, 0.66f, 0.66f,
            -1.37f, -1.36f, -1.35f, -1.36f};
        for (int i = 0; i < 12; ++i)
        {
            data.joint_pos[i] = q[i];
            data.joint_vel[i] = 0.0f;
        }
        data.root_quat_w = Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        data.root_ang_vel_b = Eigen::Vector3f::Zero();
    }

private:
    unitree::common::UnitreeJoystick joystick_;  // stays zero => zero commands
};

template <typename Vec>
static std::string fmt_vec(const Vec& v, int prec = 3)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec) << "[";
    for (size_t i = 0; i < v.size(); ++i)
        os << (i ? ", " : "") << v[i];
    os << "]";
    return os.str();
}

int main(int argc, char** argv)
{
    std::string policy_arg = (argc > 1) ? argv[1] : "../../../pretrained/example";
    const long steps = (argc > 2) ? std::atol(argv[2]) : 150;

    const fs::path policy_dir = fs::weakly_canonical(fs::path(policy_arg));
    if (!fs::exists(policy_dir / "exported" / "policy.onnx"))
    {
        std::cerr << "ERROR: " << (policy_dir / "exported" / "policy.onnx").string() << " not found\n";
        return 1;
    }

    auto env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile((policy_dir / "params" / "deploy.yaml").string()),
        std::make_shared<MockArticulation>());
    env->alg = std::make_unique<isaaclab::OrtRunner>(
        (policy_dir / "exported" / "policy.onnx").string());

    const size_t obs_dim = env->observation_manager->compute().at("obs").size();
    const size_t act_dim = env->action_manager->processed_actions().size();
    env->reset();

    std::cout << "==============================================\n"
              << " Offline policy pre-flight (no DDS, no robot)\n"
              << " Policy       : " << policy_dir.string() << "\n"
              << " step_dt      : " << env->step_dt << " s\n"
              << " Observation  : " << obs_dim << " (expect 45 x 30 = 1350)\n"
              << " Action       : " << act_dim << " (expect 12)\n"
              << " default pose : " << fmt_vec(env->robot->data.default_joint_pos, 2) << "\n"
              << " mock q       : " << fmt_vec(env->robot->data.joint_pos) << "\n"
              << "==============================================\n";

    bool ok = true;
    double max_abs = 0.0;
    for (long s = 1; s <= steps; ++s)
    {
        env->step();
        const auto raw = env->alg->get_action();
        const auto qdes = env->action_manager->processed_actions();

        for (float a : raw)
            if (std::isfinite(a))
                max_abs = std::max(max_abs, (double)std::fabs(a));
            else
                ok = false;
        for (float q : qdes)
            if (!std::isfinite(q))
                ok = false;

        if (s == 1 || s == 5 || s % 25 == 0 || s == steps)
            std::cout << " step " << std::setw(4) << s
                      << "  raw action : " << fmt_vec(raw)
                      << "  q_des : " << fmt_vec(qdes) << "\n";
    }

    std::cout << "----------------------------------------------\n"
              << " steps run    : " << steps << "\n"
              << " max|raw act| : " << std::fixed << std::setprecision(3) << max_abs << "\n"
              << " all finite   : " << (ok ? "YES" : "NO  <-- !!") << "\n"
              << " RESULT       : " << (ok ? "PASS" : "FAIL") << "\n"
              << "==============================================\n";

    return ok ? 0 : 1;
}

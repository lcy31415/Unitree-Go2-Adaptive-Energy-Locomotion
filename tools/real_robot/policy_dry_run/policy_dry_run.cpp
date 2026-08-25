// Go2 REAL Policy Dry Run — READ-ONLY (Task 5)
//
// Feeds real rt/lowstate into the exact production deploy stack:
//
//   LowState (rt/lowstate, domain 0)
//     -> BaseArticulation::update()          (deploy/include/unitree_articulation.h)
//     -> ObservationManager                  (30-frame history -> 1350-D "obs")
//     -> OrtRunner                           (exported/policy.onnx -> 12-D raw action)
//     -> ActionManager / JointAction         (x scale + default_joint_pos, clip) == q_des
//     -> terminal print
//
// Safety contract: this program NEVER creates a LowCmd publisher.
// Only go2_sub.h (subscription side) is included from the SDK, so this binary
// contains no ChannelPublisher<LowCmd>, no rt/lowcmd endpoint, and no Write().
// Verify after build:
//   nm -C go2_policy_dry_run | grep -iE "ChannelPublisher|publisher::LowCmd|RealTimePublisher"
//     -> must print nothing (the LowCmd_ IDL type-support symbols that strings
//        shows come from go2_sub.h's includes and are never used)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <boost/program_options.hpp>
#include <yaml-cpp/yaml.h>

// DDS: subscription only — deliberately NOT go2.h / go2_pub.h (publisher side)
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/dds_wrapper/robots/go2/go2_sub.h>

// Production deploy stack (header-only, same files go2_ctrl compiles)
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"  // registers obs terms
#include "isaaclab/envs/mdp/actions/joint_actions.h"       // registers action terms
#include "unitree_articulation.h"

namespace fs = std::filesystem;

using LowState_t = unitree::robot::go2::subscription::LowState;

static std::atomic<bool> g_running{true};
static void handle_signal(int) { g_running = false; }

// ---------- small formatting helpers ----------

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

static std::string fmt_scalar_vec(const std::vector<float>& v, int prec = 1)
{
    // "25 x 12" if uniform, otherwise the full vector
    const bool uniform = std::adjacent_find(v.begin(), v.end(), std::not_equal_to<>()) == v.end();
    std::ostringstream os;
    os << std::fixed << std::setprecision(prec);
    if (uniform && !v.empty())
        os << v.front() << " x " << v.size();
    else
        os << fmt_vec(v, prec);
    return os.str();
}

// ---------- policy directory resolution ----------

// Same rule as deploy param::parser_policy_dir(): if the directory itself has no
// "exported/" folder, pick the last subdirectory (alphabetical) that has one.
static fs::path resolve_policy_dir(fs::path dir)
{
    std::error_code ec;
    if (!fs::exists(dir, ec))
    {
        std::cerr << "ERROR: policy directory not found: " << dir.string() << "\n";
        std::exit(1);
    }
    if (!fs::exists(dir / "exported"))
    {
        std::vector<fs::path> subdirs;
        for (const auto& entry : fs::directory_iterator(dir))
            if (entry.is_directory())
                subdirs.push_back(entry.path());
        std::sort(subdirs.begin(), subdirs.end());
        for (auto it = subdirs.rbegin(); it != subdirs.rend(); ++it)
        {
            if (fs::exists(*it / "exported"))
            {
                dir = *it;
                break;
            }
        }
    }
    return fs::weakly_canonical(dir);
}

// <repo>/pretrained/example located relative to the executable, so the binary
// works from its build directory without any argument.
static fs::path default_policy_dir()
{
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    fs::path dir = exe.parent_path();
    for (int i = 0; i < 6 && !dir.empty(); ++i)
    {
        if (fs::exists(dir / "deploy") && fs::exists(dir / "pretrained"))
            return dir / "pretrained" / "example";
        dir = dir.parent_path();
    }
    return fs::path("pretrained/example");
}

int main(int argc, char** argv)
{
    // ---------- command line ----------
    int domain = 0;
    std::string network;
    std::string policy_arg;
    double print_period = 0.5;
    long max_steps = 0;        // 0 = run until Ctrl-C
    bool use_joystick = false; // default: forced zero commands

    namespace po = boost::program_options;
    po::options_description desc("Go2 Policy Dry Run (READ-ONLY: no LowCmd is ever created)");
    desc.add_options()
        ("help,h", "show this help message")
        ("network,n", po::value<std::string>(&network)->default_value(""),
         "DDS network interface, e.g. enp6s0 (empty = default interface)")
        ("domain", po::value<int>(&domain)->default_value(0),
         "DDS domain (real Go2: 0)")
        ("policy,p", po::value<std::string>(&policy_arg)->default_value(""),
         "policy dir with exported/policy.onnx and params/deploy.yaml "
         "(default: <repo>/pretrained/example)")
        ("print-period", po::value<double>(&print_period)->default_value(0.5),
         "seconds between status prints")
        ("steps", po::value<long>(&max_steps)->default_value(0),
         "stop after N policy steps (0 = run until Ctrl-C)")
        ("use-joystick", po::bool_switch(&use_joystick),
         "read velocity commands from the remote (default: forced zero commands)")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }
    if (print_period <= 0.0)
    {
        std::cerr << "ERROR: --print-period must be > 0\n";
        return 1;
    }

    const fs::path policy_dir =
        resolve_policy_dir(policy_arg.empty() ? default_policy_dir() : fs::path(policy_arg));

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    std::cout << "\n Go2 REAL Policy Dry Run — READ-ONLY\n"
              << " Connecting: DDS domain " << domain
              << ", interface '" << (network.empty() ? "<default>" : network) << "'\n" << std::endl;

    // ---------- DDS: subscription only ----------
    try
    {
        unitree::robot::ChannelFactory::Instance()->Init(domain, network);
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: DDS init failed on domain " << domain
                  << ", interface '" << network << "'"
                  << " (check: ip link — interface must exist and be UP)\n"
                  << "       " << e.what() << "\n";
        return 1;
    }

    auto lowstate = std::make_shared<LowState_t>();  // subscribes rt/lowstate
    std::cout << " Waiting for rt/lowstate ...\n";
    lowstate->wait_for_connection();
    std::cout << " Connected.\n";

    // ---------- deploy stack (identical to State_RLBase) ----------
    auto env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile((policy_dir / "params" / "deploy.yaml").string()),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(lowstate));
    env->alg = std::make_unique<isaaclab::OrtRunner>(
        (policy_dir / "exported" / "policy.onnx").string());

    // Measure model interface dims once, then reset history buffers exactly like
    // the production policy thread does at State_RLBase::enter().
    const size_t obs_dim = env->observation_manager->compute().at("obs").size();
    const size_t act_dim = env->action_manager->processed_actions().size();
    env->reset();

    const auto& jd = env->robot->data;
    const float rate_hz = 1.0f / env->step_dt;

    std::cout <<
        "==============================================\n"
        " Go2 REAL Policy Dry Run\n"
        "\n"
        " MODE           : DRY-RUN (READ-ONLY)\n"
        " DDS Domain     : " << domain << "\n"
        " Interface      : " << (network.empty() ? "<default>" : network) << "\n"
        "\n"
        " LowState       : ENABLED   (rt/lowstate)\n"
        " LowCmd         : NOT CREATED (no publisher, no rt/lowcmd)\n"
        "\n"
        " Policy         : " << policy_dir.string() << "\n"
        " step_dt        : " << env->step_dt << " s\n"
        " Policy rate    : " << rate_hz << " Hz\n"
        " Observation    : " << obs_dim << "\n"
        " Action         : " << act_dim << "\n"
        "\n"
        " Commands       : " << (use_joystick ? "JOYSTICK (--use-joystick)" : "FORCED ZERO") << "\n"
        "\n"
        " joint_ids_map  : " << fmt_vec(jd.joint_ids_map, 0) << "\n"
        " Kp             : " << fmt_scalar_vec(jd.joint_stiffness) << "\n"
        " Kd             : " << fmt_scalar_vec(jd.joint_damping) << "\n"
        " default pose   : " << fmt_vec(jd.default_joint_pos, 2) << "\n"
        "\n"
        " policy joint order:\n"
        "   0 FL_hip    1 FR_hip    2 RL_hip    3 RR_hip\n"
        "   4 FL_thigh  5 FR_thigh  6 RL_thigh  7 RR_thigh\n"
        "   8 FL_calf   9 FR_calf  10 RL_calf  11 RR_calf\n"
        "==============================================\n"
        " Press Ctrl-C to stop.\n" << std::endl;

    // ---------- policy loop at step_dt (same timing scheme as State_RLBase) ----------
    using clock = std::chrono::steady_clock;
    const auto dt = std::chrono::duration_cast<clock::duration>(
        std::chrono::duration<double>(env->step_dt));
    const long print_every = std::max(1L, static_cast<long>(std::llround(print_period / env->step_dt)));

    long step = 0;
    long nonfinite_events = 0;
    long timeout_events = 0;
    double max_abs_action = 0.0;
    const auto t0 = clock::now();
    auto sleep_till = t0 + dt;

    const auto& cmd_ranges = env->cfg["commands"]["base_velocity"]["ranges"];

    try
    {
        while (g_running && (max_steps <= 0 || step < max_steps))
        {
            // With --use-joystick, refresh remote state exactly like the FSM
            // pre_run() does. Without it we never call update(), so the joystick
            // stays at its zero-initialized values => velocity_commands = [0,0,0].
            if (use_joystick)
                lowstate->update();

            env->step();
            ++step;

            const bool lowstate_ok = !lowstate->isTimeout();
            if (!lowstate_ok)
                ++timeout_events;

            const auto raw  = env->alg->get_action();                    // 12-D onnx output
            const auto qdes = env->action_manager->processed_actions();  // scale + offset + clip

            bool all_finite = true;
            double cur_max = 0.0;
            for (float a : raw)
            {
                if (!std::isfinite(a)) all_finite = false;
                else cur_max = std::max(cur_max, static_cast<double>(std::fabs(a)));
            }
            for (float q : qdes)
                if (!std::isfinite(q)) all_finite = false;
            if (!all_finite)
                ++nonfinite_events;
            max_abs_action = std::max(max_abs_action, cur_max);

            if (step % print_every == 0)
            {
                const auto& d = env->robot->data;

                std::vector<float> cmd = {0.f, 0.f, 0.f};
                if (use_joystick)
                {
                    // identical to mdp::velocity_commands, display only
                    auto* joy = d.joystick;
                    cmd[0] = std::clamp(joy->ly(),  cmd_ranges["lin_vel_x"][0].as<float>(), cmd_ranges["lin_vel_x"][1].as<float>());
                    cmd[1] = std::clamp(-joy->lx(), cmd_ranges["lin_vel_y"][0].as<float>(), cmd_ranges["lin_vel_y"][1].as<float>());
                    cmd[2] = std::clamp(-joy->rx(), cmd_ranges["ang_vel_z"][0].as<float>(), cmd_ranges["ang_vel_z"][1].as<float>());
                }

                const double wall = std::chrono::duration<double>(clock::now() - t0).count();
                const Eigen::VectorXf pos_rel = d.joint_pos - d.default_joint_pos;

                std::cout << "--- t = " << std::fixed << std::setprecision(1) << wall
                          << " s   step " << step
                          << "   rate " << std::setprecision(1) << (step / wall)
                          << " Hz ---\n";
                std::cout << " gyro                : " << fmt_vec(d.root_ang_vel_b) << "\n"
                          << " projected_gravity   : " << fmt_vec(d.projected_gravity_b) << "\n"
                          << " commands vx,vy,wz   : " << fmt_vec(cmd)
                          << (use_joystick ? "" : "   (forced zero)") << "\n"
                          << " q (policy order)    : " << fmt_vec(d.joint_pos) << "\n"
                          << " joint_pos_rel       : " << fmt_vec(pos_rel) << "\n"
                          << " raw action          : " << fmt_vec(raw) << "\n"
                          << " q_des               : " << fmt_vec(qdes) << "\n"
                          << " max|raw action|     : " << std::setprecision(3) << cur_max << "\n"
                          << " all finite          : " << (all_finite ? "YES" : "NO  <-- !!") << "\n"
                          << " lowstate            : " << (lowstate_ok ? "OK" : "TIMEOUT <-- !!") << "\n"
                          << std::endl;
            }

            std::this_thread::sleep_until(sleep_till);
            sleep_till += dt;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nERROR during policy loop: " << e.what() << "\n";
        return 1;
    }

    // ---------- summary ----------
    const double wall = std::chrono::duration<double>(clock::now() - t0).count();
    std::cout <<
        "==============================================\n"
        " Dry run finished (" << (max_steps > 0 && step >= max_steps ? "step limit" : "Ctrl-C") << ")\n"
        " steps             : " << step << "\n"
        " wall time         : " << std::fixed << std::setprecision(1) << wall << " s\n"
        " avg policy rate   : " << std::setprecision(1) << (wall > 0 ? step / wall : 0.0)
        << " Hz (target " << rate_hz << ")\n"
        " max|raw action|   : " << std::setprecision(3) << max_abs_action << "\n"
        " non-finite events : " << nonfinite_events << "\n"
        " lowstate timeouts : " << timeout_events << "\n"
        " No command was ever sent to the robot.\n"
        "==============================================\n";

    return 0;
}

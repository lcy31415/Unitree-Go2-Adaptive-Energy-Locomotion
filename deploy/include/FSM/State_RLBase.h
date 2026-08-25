// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);

    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

        // ---- RL entry protocol ----
        // Capture the pose the motors are currently commanded to hold (policy
        // order). Warm-up holds this pose while the policy thread fills the
        // 30-frame observation history; blend then interpolates from this pose
        // to the live RL target, so the command stream stays continuous.
        const auto& map = env->robot->data.joint_ids_map;
        q_start_.resize(map.size());
        for (size_t i = 0; i < map.size(); ++i)
            q_start_[i] = lowcmd->msg_.motor_cmd()[map[i]].q();
        q_cmd_.assign(map.size(), 0.0f);
        t_enter_ = std::chrono::steady_clock::now();
        safety_violation_ = false;
        violation_logged_ = false;
        spdlog::info("State_{}: warm-up {:.2f} s (fill history), blend {:.2f} s, then RL control",
                     getStateString(), warmup_time_, blend_time_);

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    // ---- RL entry protocol (config keys under FSM/<state>, all optional;
    //      missing keys keep the legacy immediate-RL behavior) ----
    float warmup_time_ = 0.0f;    // hold entry pose while history fills (>= 0.6 s for 30 x 0.02 s)
    float blend_time_  = 0.0f;    // q_cmd = (1-a) q_start + a q_rl, a: 0 -> 1
    std::chrono::steady_clock::time_point t_enter_;
    std::vector<float> q_start_;  // policy order
    std::vector<float> q_cmd_;    // policy order

    // ---- runtime safety gate (checked in run() before writing motor_cmd) ----
    float max_action_abs_ = std::numeric_limits<float>::infinity();
    float max_q_des_dev_  = std::numeric_limits<float>::infinity();
    std::atomic<bool> safety_violation_{false};
    bool violation_logged_ = false;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)

#include <array>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <thread>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>

using namespace unitree::robot;

int main()
{
    std::cout
        << "=============================================\n"
        << " Go2 Real LowState Inspector\n"
        << " READ-ONLY: no LowCmd publisher\n"
        << " DDS domain : 0\n"
        << " Interface  : enp6s0\n"
        << " Topic      : rt/lowstate\n"
        << "=============================================\n";

    ChannelFactory::Instance()->Init(0, "enp6s0");

    auto subscriber =
        std::make_shared<
            ChannelSubscriber<unitree_go::msg::dds_::LowState_>
        >("rt/lowstate");

    subscriber->InitChannel(
        [](const void* message)
        {
            static std::atomic<unsigned long long> counter{0};

            if (++counter % 50 != 0)
                return;

            const auto* state =
                static_cast<
                    const unitree_go::msg::dds_::LowState_*
                >(message);

            const auto& imu = state->imu_state();

            std::cout << std::fixed << std::setprecision(4);

            std::cout
                << "\n================ LowState ================\n";

            std::cout
                << "Quaternion : ["
                << imu.quaternion()[0] << ", "
                << imu.quaternion()[1] << ", "
                << imu.quaternion()[2] << ", "
                << imu.quaternion()[3] << "]\n";

            std::cout
                << "Gyro       : ["
                << imu.gyroscope()[0] << ", "
                << imu.gyroscope()[1] << ", "
                << imu.gyroscope()[2] << "]\n";

            std::cout
                << "Accel      : ["
                << imu.accelerometer()[0] << ", "
                << imu.accelerometer()[1] << ", "
                << imu.accelerometer()[2] << "]\n";

            std::cout
                << "RPY        : ["
                << imu.rpy()[0] << ", "
                << imu.rpy()[1] << ", "
                << imu.rpy()[2] << "]\n";

            static const std::array<const char*, 12> joint_names = {
                "FR_hip",   "FR_thigh", "FR_calf",
                "FL_hip",   "FL_thigh", "FL_calf",
                "RR_hip",   "RR_thigh", "RR_calf",
                "RL_hip",   "RL_thigh", "RL_calf"
            };

            std::cout
                << "\nID  Joint        q(rad)       dq(rad/s)    tau_est\n"
                << "-------------------------------------------------\n";

            const auto& motors = state->motor_state();

            for (int i = 0; i < 12; ++i)
            {
                std::cout
                    << std::setw(2) << i << "  "
                    << std::setw(10) << joint_names[i] << "  "
                    << std::setw(10) << motors[i].q() << "  "
                    << std::setw(11) << motors[i].dq() << "  "
                    << std::setw(10) << motors[i].tau_est()
                    << "\n";
            }

            std::cout
                << "==========================================\n"
                << std::flush;
        },
        1
    );

    std::cout << "Waiting for rt/lowstate..." << std::endl;

    while (true)
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}

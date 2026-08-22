#include <chrono>
#include <iostream>
#include <memory>
#include <thread>

#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>

using namespace unitree::robot;

int main()
{
    std::cout << "Initializing DDS: domain=1, interface=lo" << std::endl;

    ChannelFactory::Instance()->Init(1, "lo");

    auto subscriber =
        std::make_shared<
            ChannelSubscriber<unitree_go::msg::dds_::LowState_>
        >("rt/lowstate");

    subscriber->InitChannel(
        [](const void* message)
        {
            const auto* state =
                static_cast<
                    const unitree_go::msg::dds_::LowState_*
                >(message);

            std::cout
                << "FR_hip q = "
                << state->motor_state()[0].q()
                << " | gyro = ["
                << state->imu_state().gyroscope()[0] << ", "
                << state->imu_state().gyroscope()[1] << ", "
                << state->imu_state().gyroscope()[2] << "]"
                << std::endl;
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

import sys
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import time

def multiple_points(bot, point_list):
    for point in point_list:
        print(f'going to point {point}')
        bot.arm.set_ee_pose_components(x=point[0], y=point[1], z=point[2])
        bot.arm.set_trajectory_time(moving_time=0.5)
        
        # time.sleep(1)

def pta_to_ptb(bot, pta, ptb):
    bot.arm.set_ee_pose_components(x=pta[0], y=pta[1], z=pta[2])
    bot.arm.set_ee_pose_components(x=ptb[0], y=ptb[1], z=ptb[2])

def set_gripper(bot, closed: bool):
    if closed:
        bot.gripper.grasp()
    else:
        bot.gripper.release()

    
def main():
    bot = InterbotixManipulatorXS(
        robot_model='px150',
        group_name='arm',
        gripper_name='gripper',
    )

    robot_startup()

    bot.arm.set_trajectory_time(moving_time=1)
    bot.arm.go_to_sleep_pose()
    bot.arm.go_to_home_pose()

    #Move from one point to another, points are in meters
    #pta = [0.2, 0.1, 0.2]
    #ptb = [0.1, 0.2, 0.1]

    #pta_to_ptb(bot, pta, ptb)

    #Check if points move relative to eachother or on a global scale
    # pta = [0.1, 0.2, 0.1]
    # ptb = [0.1, 0.2, 0.1]

    #pta_to_ptb(bot, pta, ptb)

    #Figuring out where the points are
    #Half a meter too big!
    #0 values error out
    points = [
        # [0.2, 0.1, 0.1],
        # [0.1, 0.2, 0.1],
        # [0.1, 0.1, 0.2],

        #[0.1, 0.1, 0.1], # not valid
        #Tested x values
        #[0.2, 0.1, 0.1],
        #[0.3, 0.1, 0.1],
        #[0.4, 0.1, 0.1],

        #Tested y values    
        # [0.2, 0.2, 0.1],
        # [0.2, 0.3, 0.1],
        # [0.2, 0.4, 0.1],

        #[0.2, 0, 0.1],
        [0.3, 0, 0.06],
        [0.4, 0, 0.06],

        # [0.2, 0.1, 0.2],
        
        [0.25, -0.1, 0.05],
        [0.30, -0.15, 0.05]

        # [0.03, 0.0, 0.1],
        # [0.04, 0.0, 0.1],

        # [0.025, -0.05, 0.1],
        # [0.027, -0.07, 0.1],
    ]

    bot.arm.go_to_home_pose()
    # time.sleep(1)

    # multiple_points(bot, points)
    #Ball 1
    multiple_points(bot, [[0.3, 0, 0.07],[0.4, 0, 0.07]])
    time.sleep(0.5)
    bot.arm.set_trajectory_time(moving_time=1)
    bot.arm.go_to_sleep_pose()
    time.sleep(0.5)
    bot.arm.set_trajectory_time(moving_time=1)
    multiple_points(bot, [[0.25, -0.1, 0.07], [0.35, -0.2, 0.07]])
    time.sleep(0.5)
    bot.arm.set_trajectory_time(moving_time=1)
    bot.arm.go_to_sleep_pose()
    multiple_points(bot, [[0.25, 0.1, 0.07], [0.35, 0.2, 0.07]])

    
    bot.arm.set_trajectory_time(moving_time=1)
    bot.arm.go_to_sleep_pose()

    robot_shutdown()

if __name__ == '__main__':
    main()

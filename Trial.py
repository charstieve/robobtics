import math
import sys
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import time
from RealBallPosition import BallDetection
import cv2
import threading

ball_position = None

def set_gripper(bot, closed: bool):
    if closed:
        bot.gripper.grasp()
    else:
        bot.gripper.release()

# def robot_thread_func(bot):
#     tag_offset_x = 0.0
#     tag_offset_y = 5 * 0.0254
#     z = 0.06
#     while True:
#         print("hello!")
#         try:
#             if ball_position == (None, None, None) or ball_position == None:
#                 print("No ball detected")
#                 continue
#             x, y, z = ball_position
#             print(f"Ball position thread: {ball_position}")
#             print(-x, y + tag_offset_y, 0.06)

#             bot.arm.set_ee_pose_components(y = -x, x = y - tag_offset_y, z=0.07)
#             print("moved to hit ball")
#             #bot.arm.set_trajectory_time(moving_time=0.3)
#             hit_distance = 0.05 # The distance to move forwards so that the arm hits the ball
#             #bot.arm.set_ee_pose_components(y = -x, x = y - tag_offset_y + hit_distance, z=0.07)
#             #time.sleep(1)
#         except Exception as e:
#             print(f"Error in robot thread: {e}")
#             continue


def increase_motor_accuracies(bot: InterbotixManipulatorXS):
    # ensure the bot is in a safe pose to torque off
    print("Checking values of motor 'accuracy' registers to attempt updates.")
    JOINT_DEFAULTS = {'waist': 640, 'shoulder': 800, 'elbow': 800, 'wrist_angle': 640, 'wrist_rotate': 640}
    torque = False
    # update register values for each motor
    for joint in bot.arm.group_info.joint_names:
         bot.core.robot_set_motor_pid_gains(
                cmd_type='single',
                name=joint,
                kp_pos=JOINT_DEFAULTS[joint] + 150,
            )
        # read the old value
        
    print("Enabling torque for robot arm")
    bot.core.robot_torque_enable(cmd_type="group", name="all", enable=True)
    time.sleep(1)

def robot_thread_func(bot):
    LOWER = 3 * 0.0254
    UPPER = 5 * 0.0254
    # HOME_DISTANCE = 15 * 0.0254
    HOME_DISTANCE = 7 * 0.0254
    tag_offset_x = 0.0
    tag_offset_y = 5 * 0.0254
    # tag_offset_y = 5
    hit_distance = 0.15 # The distance to move forwards so that the arm hits the ball
    z = 0.06
    previous_angle = None
    while True:
        # bot.arm.go_to_home_pose()
        # time.sleep(0.5)
        bot.arm.set_trajectory_time(moving_time=0.5)
        print("hello!")
        try:
            if ball_position == (None, None, None) or ball_position == None:
                print("No ball detected")
                continue

            x, y, z = ball_position
            ax = y - tag_offset_y
            ay = -x
            az = 0.07

            print(f"Ball position thread: {ball_position}")
            print(-x, y + tag_offset_y, 0.06)
            angle = math.atan(ay/ax)
            print(f"Angle: {angle * 2* 180 / math.pi}")
            distance = math.sqrt((ax ** 2) + (ay ** 2))

            if LOWER < distance < UPPER:
                print("In range")
                bot.arm.set_ee_pose_components(x=ax, y=ay, z=az)
                time.sleep(0.3)
                bot.arm.set_ee_pose_components(x=ax + hit_distance, y=ay, z=az)
                # time.sleep(10)
                previous_angle = None
                continue

            # if previous_angle is not None and abs(previous_angle - angle) < 0.1:
            #     previous_angle = angle
            #     print("Angle is similar")
            #     continue
                
            previous_angle = angle

            # Lowerbound coordinates
            hx = HOME_DISTANCE * math.cos(angle)
            hy = HOME_DISTANCE * math.sin(angle)

            print(f"hx: {hx}, hy: {hy}")
            print(f"ax: {ax}, ay: {ay}")
            # bot.arm.set_ee_pose_components(x=hx, y=hy, z=az)
            bot.arm.set_ee_pose_components(x=ax, y=ay, z=az)

            print("moved to hit ball")
        except Exception as e:
            print(f"Error in robot thread: {e}")
            continue
        
def main():
    global ball_position
    bot = None
    bot = InterbotixManipulatorXS(
        robot_model='rx200',
        group_name='arm',
        gripper_name='gripper',
    )

    cam = cv2.VideoCapture(0)
    # ball_detection = BallDetection()

    robot_startup()
    increase_motor_accuracies(bot)
    bot.arm.go_to_sleep_pose()
    bot.arm.set_trajectory_time(moving_time=3.5)
    
    # bot.arm.go_to_home_pose()
    print('Going to second home')
    bot.arm.set_ee_pose_components(y=0.05, x = 10 * 0.0254 , z=0.07)
    # 0.25, 0.12, 0.05
    # bot.arm.set_ee_pose_components(y=0.25, x = 0.12 , z=0.05)
    print(0.1, 5 * 0.0254, 0.07)
    print(bot.arm.get_ee_pose())
    print('At Home')
    ball_detection = BallDetection(tag_size=2 * 0.0254)

    
    bot_thread = threading.Thread(target=robot_thread_func, args=([bot]))
    bot_thread.start()

    while True:
        #print("here")
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame")
            continue
        ball_position = ball_detection.get_position(frame)
        # print("Ball position is: ", ball_position)
        cv2.imshow("Ball Detection", frame)
        cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            bot.arm.go_to_sleep_pose()
            break



    print('Terminating ...')
    # while True:
    #     x = input("X")
    #     y = input("Y")
    #     x *= 0.0254
    #     y *= 0.0254



    cam.release()
    cv2.destroyAllWindows()
    # robot_shutdown()


if __name__ == '__main__':
    main()

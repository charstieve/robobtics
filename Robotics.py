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
paused = False

def set_gripper(bot, closed: bool):
    if closed:
        bot.gripper.grasp()
    else:
        bot.gripper.release()

def robot_thread_func(bot):
    LOWER = 3 * 0.0254
    UPPER = 12 * 0.0254
    HOME_DISTANCE = 7 * 0.0254
    tag_offset_x = 0.0
    tag_offset_y = 5 * 0.0254
    hit_distance = 0.05  # The distance to move forwards so that the arm hits the ball
    z = 0.06
    previous_angle = None
    while True:
        if paused:
            continue
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
            angle = math.atan(x/y)
            distance = math.sqrt((ax ** 2) + (ay ** 2))

            if LOWER < distance < UPPER:
                bot.arm.set_ee_pose_components(x=ax, y=ay, z=az)
                time.sleep(0.3)
                bot.arm.set_ee_pose_components(x=ax + hit_distance, y=ay, z=az)
                time.sleep(10)
                previous_angle = None

            if previous_angle == angle:
                continue

            # Lowerbound coordinates
            hx = HOME_DISTANCE * math.sin(angle)
            hy = HOME_DISTANCE * math.cos(angle)

            bot.arm.set_ee_pose_components(x=hx, y=hy, z=az)

            print("moved to hit ball")
        except Exception as e:
            print(f"Error in robot thread: {e}")
            continue

def main():
    global ball_position, paused
    bot = InterbotixManipulatorXS(
        robot_model='rx200',
        group_name='arm',
        gripper_name='gripper',
    )

    cam = cv2.VideoCapture(0)
    ball_detection = BallDetection()

    robot_startup()
    bot.arm.set_trajectory_time(moving_time=0.5)
    # bot.arm.go_to_sleep_pose()
    bot.arm.go_to_home_pose()
    LOWER = 3 * 0.0254
    bot.arm.set_ee_pose_components(x=0, y=LOWER, z=0.07)
    ball_detection = BallDetection()

    
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
        match cv2.waitKey(1):
            case ord('q'):
                bot.arm.go_to_sleep_pose()
                break
            case ord('p'):
                paused = not paused




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

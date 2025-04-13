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

def robot_thread_func(bot):
    tag_offset_x = 0.0
    tag_offset_y = 5 * 0.0254
    z = 0.06
    while True:
        print("hello!")
        try:
            if ball_position == (None, None, None) or ball_position == None:
                print("No ball detected")
                continue
            x, y, z = ball_position
            print(f"Ball position thread: {ball_position}")
            print(-x, y + tag_offset_y, 0.06)

            bot.arm.set_ee_pose_components(y = -x, x = y - tag_offset_y, z=0.07)
            print("moved to hit ball")
            #bot.arm.set_trajectory_time(moving_time=0.3)
            hit_distance = 0.05 # The distance to move forwards so that the arm hits the ball
            #bot.arm.set_ee_pose_components(y = -x, x = y - tag_offset_y + hit_distance, z=0.07)
            #time.sleep(1)
        except Exception as e:
            print(f"Error in robot thread: {e}")
            continue

def main():
    global ball_position
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

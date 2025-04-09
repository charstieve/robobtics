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

def camera_thread_func(cam, ball_detection):
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture frame")
            continue
        ball_position = ball_detection.get_ball_position(frame)
        cv2.imshow("Ball Detection", frame)
        cv2.waitKey(1)
    
def main():

    bot = InterbotixManipulatorXS(
        robot_model='rx200',
        group_name='arm',
        gripper_name='gripper',
    )

    cam = cv2.VideoCapture(1)
    ball_detection = BallDetection()

    robot_startup()
    bot.arm.set_trajectory_time(moving_time=1)
    # bot.arm.go_to_sleep_pose()
    bot.arm.go_to_home_pose()
    ball_detection = BallDetection()

    tag_offset_x = 0.0
    tag_offset_y = 5 * 0.0254
    z = 0.06
    previous_position = None

    cam_thread = threading.Thread(target=camera_thread_func, args=(cam, ball_detection))
    cam_thread.start()

    while True:
        try:
            # for _ in range(50):
            #     _, _ = cam.read()
            

            if ball_position == (None, None, None) or ball_position == None:
                print("No ball detected")
                continue
            x, y, z = ball_position
            print(f"Ball position: {ball_position}")
            print(-x, y + tag_offset_y, 0.06)
            # y=-x, x= y + tag_offset_y, z=0.06
            # Go to ball
            # user_input = input("Press Enter to continue...")
            
            print(ball_position)
            # if user_input.lower() == 'q':
            #     print("Exiting...")
            #     cam.release()
            #     cv2.destroyAllWindows()
            #     robot_shutdown()
            #     break
            bot.arm.set_ee_pose_components(y=-x, x= y - tag_offset_y, z=0.06)

            # bot.arm.set_ee_pose_components(y=-x, x= y - 0.07 - tag_offset_y, z=0.06)
            bot.arm.set_trajectory_time(moving_time=0.5)
            # hit_distance = 0.05 # The distance to move forwards so that the arm hits the ball
            # bot.arm.set_ee_pose_components(y=-x, x=y - tag_offset_y + hit_distance, z=0.06)
            # bot.arm.set_trajectory_time(moving_time=1)
            # bot.arm.go_to_home_pose()
        except Exception as e:
            print(e)
    print('Terminating ...')
    # while True:
    #     x = input("X")
    #     y = input("Y")
    #     x *= 0.0254
    #     y *= 0.0254



    cam.release()
    cv2.destroyAllWindows()
    robot_shutdown()


if __name__ == '__main__':
    main()

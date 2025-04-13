import sys
from interbotix_common_modules.common_robot.robot import robot_shutdown, robot_startup
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import numpy as np
import time
from RealBallPosition import BallDetection
import cv2


def set_gripper(bot, closed: bool):
    if closed:
        bot.gripper.grasp()
    else:
        bot.gripper.release()


def main():
    bot = InterbotixManipulatorXS(
        robot_model='rx200',
        group_name='arm',
        gripper_name='gripper'
    )

    cam = cv2.VideoCapture(0)
    ball_detection = BallDetection()

    robot_startup()
    bot.arm.set_trajectory_time(moving_time=1)
    bot.arm.go_to_sleep_pose()
    bot.arm.go_to_home_pose()

    tag_offset_x = 0.0
    tag_offset_y = 5.6 * 0.0254
    z = 0.06

    while True:
        try:

            for i in range(30):
                cam.read()
            _, frame = cam.read()

            cv2.imshow("Ball Detection", frame)
            cv2.waitKey(1)

            if not _:
                print("Failed to capture frame")
                continue
            ball_position = ball_detection.get_position(frame)

            if ball_position == (None, None, None):
                print("No ball detected")
                continue

            x, y, z = ball_position
            print(f"Ball position: {ball_position}")

            # Go to ball
            #user_input = input("Press Enter to continue...")
            #time.sleep(1)
            print(ball_position)
            if cv2.waitKey(1) == ord('q'):
                print("Exiting...")
                cam.release()
                cv2.destroyAllWindows()
                robot_shutdown()
                break

            print("first pose")
            bot.arm.set_ee_pose_components(y = -x, x = y - tag_offset_y + 0.06, z=0.068)
            bot.arm.set_trajectory_time(moving_time=0.3)
            hit_distance = 0.05 # The distance to move forwards so that the arm hits the ball
            print("second pose")
            #bot.arm.set_ee_pose_components(y = -x, x = y - tag_offset_y + hit_distance, z=0.07)
        except:
            break
    print('Terminating ...')
    cam.release()
    cv2.destroyAllWindows()
    robot_shutdown()


if __name__ == '__main__':
    main()

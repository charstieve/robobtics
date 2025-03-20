import cv2
import numpy as np

class BallTracker:
    '''
    Class to keep track of ball from a video stream and offset it's position to a robot using a AprilTag
    '''
    def __init__(self, tag_size, robot_tag_offset_x, robot_tag_offset_y, apriltag_family, camera_number=0):
        self.tag_size = tag_size
        self.robot_tag_offset_x = robot_tag_offset_x
        self.robot_tag_offset_y = robot_tag_offset_y
        self.apriltag_family = apriltag_family
        self.camera_number = camera_number
        self._detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))

        pass

    def april_tag_corners(self, frame) -> tuple[float, float, float, float] | None:
        '''
        Returns the corners of the april tag in the frame
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detector.detectMarkers(gray)
        return corners[0][0]
    
    def find_robot_apriltag(self, frame):
        '''
        Finds the april tag position on the robot
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detector.detectMarkers(gray)
        cv2.aruco.estimatePos
        return self._detector.detectMarkers(gray)
     
    #Find in pixels
    def find_orange_ball_pixels(self, frame) -> tuple[float, float] | None:
        '''
        Finds the orange ball position in the frame using cv2 and hue circle filter to find the largest orange circle in the frame
        '''
        # img = frame.copy()
        

        # #Only take orange from the frame
        # lower_orange = np.array([10, 100, 100])
        # upper_orange = np.array([25, 255, 255])
        # mask = cv2.inRange(frame, lower_orange, upper_orange)

        gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blurred_gray_frame = cv2.GaussianBlur(gray_scale_frame, ksize=(7,7), sigmaX=0)
        
        balls_in_frame = cv2.HoughCircles(blurred_gray_frame, cv2.HOUGH_GRADIENT, 1.8, 100)

        cimg = cv2.cvtColor(blurred_gray_frame, cv2.COLOR_GRAY2BGR)
        print('_______')
        print(balls_in_frame)
        print('_______')
        if balls_in_frame is None:
            return
        # for ball in balls_in_frame[0]:
        #     print("ball: ", ball)
        # # return
        for ball in balls_in_frame[0]:
        #     # print(ball)
        #     #Draw circle around ball
        #     print('----------------')
        #     print(ball[0], ball[1])
            cv2.circle(cimg, (int(ball[0]), int(ball[1])), int(ball[2]), (0,0,255), 2)
            
        #     #Draw circle in center
            cv2.circle(cimg, (int(ball[0]), int(ball[1])), 2, (0,0,255), 3)

            cv2.imshow('detected circles',cimg)

        ball_tuple = (ball[0])
        return ball_tuple
        
    #Convert to meters and take into account april tag to find the position of the ball for the robot
    def find_ball_position(self, frame) -> tuple[float, float] | None:
        '''
        Returns the ball position in the frame
        '''
        
        ball_position = self._find_ball_position(frame)

    
while True:
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    ball_tracker = BallTracker(tag_size=0.05, robot_tag_offset_x=0.1, robot_tag_offset_y=0.1, apriltag_family="tag36h11", camera_number=1)
    ball_tracker.find_orange_ball_pixels(frame)
    # frame = cv2.imread("apriltag_36h11.png")
    # tag = ball_tracker.april_tag_corners(frame)
    # print(tag)
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)

            
            
        
    
    
        

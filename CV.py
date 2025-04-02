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
    # TODO: Currently sends the frame back with the highlight. The pixel coordinates of the balls
    # are also generated here, just have to figure out how to return them along with the frame when 
    # they are sometimes None, which seems to cause an issue when you try to grab the frame 
    # at the bottom where the webcam is run
    def find_orange_ball_pixels(self, frame): #-> tuple[float, float] | None:
        '''
        Finds the orange ball position in the frame using cv2 and hue circle filter to find the largest orange circle in the frame
        '''

        # #Only take orange from the frame
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # https://stackoverflow.com/questions/48528754/what-are-recommended-color-spaces-for-detecting-orange-color-in-open-cv
        # Lower range (darker orange) values
        lower_orange = np.array([10, 50, 70])
        # Upper range (lighter orange) values
        upper_orange = np.array([35, 255, 255])

        # Apply a mask that removes any color not within the orange range
        # It sets everything that is not orange to black and anything that is orange 
        # to white
        mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

        # Apply the mask to the frame, so anything that is white keeps its color
        # and anything that was previously turned black stays black
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert to greyscale with the image, so the orange ping pong balls will be turned
        # into greyscale so that they can be used with Hough Circles
        gray_scale_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian Blur to the grey-scaled fram for use with Hough Circles
        blurred_gray_frame = cv2.GaussianBlur(gray_scale_frame, ksize=(7,7), sigmaX=0)
        
        # Find the coordinates of the balls in the frames using Hough Circles
        balls_in_frame = cv2.HoughCircles(blurred_gray_frame, cv2.HOUGH_GRADIENT, 1.8, 25)

        # Test statement to show when the balls are being detected in case it is not shown 
        # visually
        print('_______')
        print(balls_in_frame)
        print('_______')

        # If there are no balls in the frame then return the frame normally
        if balls_in_frame is None:
            return frame

        # Iterate through every ball in the frame
        for ball in balls_in_frame[0]:

            #Draw circle around ball
            cv2.circle(frame, (int(ball[0]), int(ball[1])), int(ball[2]), (0,0,255), 2)
            
            #Draw circle in center of the ball
            cv2.circle(frame, (int(ball[0]), int(ball[1])), 2, (0,0,255), 3)

        # These are the coordinates that should also be returned, or can be returned in 
        # another function
        ball_tuple = (ball[0])
        # Return the frame with the circles drawn around the balls.
        return frame
        
    #Convert to meters and take into account april tag to find the position of the ball for the robot
    def find_ball_position(self, frame) -> tuple[float, float] | None:
        '''
        Returns the ball position in the frame
        '''
        
        ball_position = self._find_ball_position(frame)

# https://stackoverflow.com/questions/2601194/displaying-a-webcam-feed-using-opencv-and-python
cv2.namedWindow("preview")
vc = cv2.VideoCapture(1)

if vc.isOpened(): # try to get the first frame
    ret, frame = vc.read()
else:
    ret = False

while ret:

    ret, frame = vc.read()

    ball_tracker = BallTracker(tag_size=0.05, robot_tag_offset_x=0.1, robot_tag_offset_y=0.1, apriltag_family="tag36h11", camera_number=1)

    frame = ball_tracker.find_orange_ball_pixels(frame)

    cv2.imshow("preview", frame)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")
vc.release()

# cap = cv2.VideoCapture(1)
# while True:
#     #cap = cv2.VideoCapture(1)
#     ret, frame = cap.read()
    # ball_tracker = BallTracker(tag_size=0.05, robot_tag_offset_x=0.1, robot_tag_offset_y=0.1, apriltag_family="tag36h11", camera_number=1)
    # ball_tracker.find_orange_ball_pixels(frame)
#     # frame = cv2.imread("apriltag_36h11.png")
#     # tag = ball_tracker.april_tag_corners(frame)
#     # print(tag)
#     cv2.imshow("Frame", frame)
#     cv2.waitKey(1)

# cap.release()  # Release camera when done
# cv2.destroyAllWindows()  # Close all OpenCV windows

            
            
        
    
    
        

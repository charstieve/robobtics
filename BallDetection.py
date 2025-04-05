import cv2
import numpy as np

# A class to detect a ball in the frame. Currently focuses on blue ping pong balls. 
# Finds the balls based on color then draws a contour around them and finds the center
class BallDetection:

    # Requires the lower range of the color, the upper range, as well as the camera being utilized
    def __init__(self, lower_range, upper_range, camera):
        self.lower_range = lower_range # Lower Blue
        self.upper_range = upper_range # Upper Blue
        self.camera = camera #
        pass

    # https://stackoverflow.com/questions/45095734/python-finding-contours-of-different-colors-on-an-image
    # Convert the frame from Blue Green Red color scale to HSV color scale, then find anything in the frame 
    # that is within the HSV color range in the footage
    def blue_mask(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv_frame, self.lower_range, self.upper_range)

        return mask_blue
    
    # Draw contours around the blue that was previously isolated from the frame
    # RETR_EXTERNAL returns only the outermost contour in case multiple rings of blue are detected
    # CHAIN_APPROX_SIMPLE returns only the endpoints of the contours instead of every single contour point
    def draw_contours(self, frame, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image = np.copy(frame)
        cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

        return image, contours
    
    # https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
    # https://dsp.stackexchange.com/questions/4868/contour-and-area-raw-spatial-and-central-image-moments#8521
    # https://en.wikipedia.org/wiki/Image_moment#Central_moments
    # Find the center of the contours drawn around the balls by finding the central moments of the contour
    # extracting the x and y value of the centroid from that, and drawing the center of the contour into the frame
    # as well as return both the frame and center.
    # Moments are used to provide information about an object in an image 
    def draw_centers(self, frame, contours):
        cX = 0
        cY = 0
        if contours:
            for contour in contours:

                center = cv2.moments(contour)

                if center["m00"] != 0:
                    cX = int(center["m10"] / center["m00"])
                    cY = int(center["m01"] / center["m00"])

                if (cX != 0 and cY != 0):
                    cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1)
            
        centroid = np.array([cX, cY])
        return frame, centroid
      
# Testing area

# Define boundaries, camera, then run the functions and display the final frame to ensure it works
lower_blue = np.array([90, 100, 70])
upper_blue = np.array([135, 255, 255])

cam = cv2.VideoCapture(1)  # Webcam

while True:

    _, frame = cam.read()

    ball_detection = BallDetection(lower_blue, upper_blue, cam)

    mask = ball_detection.blue_mask(frame)

    contours_frame, contours = ball_detection.draw_contours(frame, mask)

    centers_frame, center = ball_detection.draw_centers(contours_frame, contours)

    cv2.imshow("Ball Detection", centers_frame)


    if cv2.waitKey(1)==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()       
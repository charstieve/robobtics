import cv2
import numpy as np

class BallDetection:
    def __init__(self,

                 hsv_lower: tuple[float] = (37, 109, 132), #[90, 100, 70]
                 hsv_upper: tuple[float] =  (180, 255, 255), #[135, 255, 255]
                 calibration_matrix_path: str = 'calibration_matrix.npy',
                 distortion_path: str = 'distortion_coefficients.npy',
                 tag_size: float = 2, # * 0.0254,
                 tag_family: int = cv2.aruco.DICT_APRILTAG_36h11,
                 tag_id: int = 2):
        self._hsv_lower = np.array(hsv_lower)
        self._hsv_upper = np.array(hsv_upper)
        self._calibration_matrix = np.load(calibration_matrix_path)
        self._distortion_coefficient = np.load(distortion_path)

        self._tag_size = tag_size
        aruco_dict = cv2.aruco.getPredefinedDictionary(tag_family)
        parameters = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self._tag_id = tag_id
        self._rvec = None
        self._tvec = None

    def get_position(self, frame) -> tuple:
        """
        Get Ball positions as x,y,z in meters.
        Returns a tuple of the ball positions and the annotated frame
        """
        mask = self._mask(frame)
        contours = self._draw_contours(frame, mask)
        centers = self._draw_centers(frame, contours)
        self._pose_estimation(frame)
        # print(len(centers))
        # print(self._rvec is None)
        # print(self._tvec is None)
        if len(centers) == 0 or self._rvec is None or self._tvec is None:
            return None, None, None

        points = np.array([[centers[0], centers[1]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(points, self._calibration_matrix, self._distortion_coefficient)
        ray = np.array([undistorted[0][0][0], undistorted[0][0][1], 1.0])

        R, _ = cv2.Rodrigues(self._rvec)
        C = -np.matmul(R.T, self._tvec)
        ray_tag = np.matmul(R.T, ray.reshape(3, 1))
        t = -C[2] / ray_tag[2]
        point = C + t * ray_tag

        return float(point[0]), float(point[1]), float(point[2])

    # https://stackoverflow.com/questions/45095734/python-finding-contours-of-different-colors-on-an-image
    # Convert the frame from Blue Green Red color scale to HSV color scale, then find anything in the frame 
    # that is within the HSV color range in the footage
    def _mask(self, frame):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_frame, self._hsv_lower, self._hsv_upper)
        return mask
    
    # Draw contours around the blue that was previously isolated from the frame
    # RETR_EXTERNAL returns only the outermost contour in case multiple rings of blue are detected
    # CHAIN_APPROX_SIMPLE returns only the endpoints of the contours instead of every single contour point
    def _draw_contours(self, frame, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
        cv2.drawContours(frame, largest_contour, -1, (0, 255, 0), 1)

        return largest_contour
    
    # https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
    # https://dsp.stackexchange.com/questions/4868/contour-and-area-raw-spatial-and-central-image-moments#8521
    # https://en.wikipedia.org/wiki/Image_moment#Central_moments
    # Find the center of the contours drawn around the balls by finding the central moments of the contour
    # extracting the x and y value of the centroid from that, and drawing the center of the contour into the frame
    # as well as return both the frame and center.
    # Moments are used to provide information about an object in an image 
    def _draw_centers(self, frame, contours):
        cX = 0
        cY = 0
        if contours:
            for contour in contours:

                center = cv2.moments(contour)

                if center["m00"] != 0:
                    cX = int(center["m10"] / center["m00"])
                    cY = int(center["m01"] / center["m00"])

                if cX != 0 and cY != 0:
                    cv2.circle(frame, (cX, cY), 3, (255, 255, 255), -1)

        centroid = np.array([cX, cY])
        return centroid

    def _pose_estimation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self._detector.detectMarkers(gray)

        if ids is None or len(corners) <= 0:
            return

        for i, corner in enumerate(corners):
            if ids[i] != self._tag_id:
                continue

            half_size = self._tag_size / 2
            object_points = np.array([
                [-half_size, half_size, 0],
                [half_size, half_size, 0],
                [half_size, -half_size, 0],
                [-half_size, -half_size, 0]
            ])

            image_points = corner.reshape(4, 2).astype(np.float32)
            success, rvec, tvec = cv2.solvePnP(object_points,
                                               image_points,
                                               self._calibration_matrix,
                                               self._distortion_coefficient)
            if not success:
                return

            cv2.drawFrameAxes(frame, self._calibration_matrix, self._distortion_coefficient, rvec, tvec, half_size)

            self._rvec = rvec
            self._tvec = tvec


if __name__ == '__main__':

    # Initialize camera
    cam = cv2.VideoCapture(1)
    ball_detection = BallDetection()

    while True:

        # Capture frame
        ret, frame = cam.read()
        if not ret:
            break

        # Get ball position
        x, y, z = ball_detection.get_position(frame)
        print(f"Ball position: {x}, {y}, {z}")
        # Display frame
        cv2.imshow("Ball Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

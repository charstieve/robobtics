from filterpy.kalman import KalmanFilter
import cv2
from RealBallPosition import BallDetection
import numpy as np

'''
Kalman filter: (kf)

Limitations: 
- kf assumes system is linear
- non linear uses ekf or ukf
- kf assumes process and measuremnets = gaussian and white
- accuracy dependant on model accuracy

logic: 
- uses linear stochastic difference equation
- uses prev state to predict next

state transition matrix: link curr state to prev state
(optional: control input w control input matrix)

need to: 
state -> measurement deomain using transform matrix
process noise vector with covariance


best practices:
process noise covariance to avoid oscillations in the state estimate
measurement noise covariance to avoid overfitting to the data
initial state estimate to avoid divergence
'''


# class Trajectory:
measured = []
predicted = []

# load detector
kf = cv2.KalmanFilter(4,2)

kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
kf.transitionMatrix = np.array(
    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
)
kf.processNoiseCov = (
    np.array([[1, 0, 0, 0], 
              [0, 1, 0, 0], 
              [0, 0, 1, 0], 
              [0, 0, 0, 1]], np.float32)
    * 0.03
)

if __name__ == '__main__':
    cam = cv2.VideoCapture(0)
    ball_detection = BallDetection()
    while True:
        # Initialize camera
        print('Here')
        
        # Capture frame
        ret, frame = cam.read()
        if not ret:
            break

        # Get ball position
        cx, cy, cz = ball_detection.get_position(frame)
        print(f"Ball position: {cx}, {cy}, {cz}")
        ret, frame = cam.read()
        
        if not ret:
            #print("Failed to capture frame")
            continue
            
        if cx is None or cy is None or cz is None:
            print("No ball detected")
            continue

        measured.append((cx,cy))
        print(f"Ball position: {cx}, {cy}, {cz}")
        
        kf.correct(np.array([[cx], [cy]], np.float32))
        tp = kf.predict()
        predicted.append((int(tp[0].item()), int(tp[1].item())))
        
        cv2.circle(frame, (int(tp[0]), int(tp[1])), 20, (255,0,0), 4)
        # Display frame
        cv2.imshow("Ball Detection", frame)
        # Display frame
        cv2.imshow("Ball Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
# while True:
#     try: 
#         ret, frame = camera.read()
        
#         if not ret:
#             #print("Failed to capture frame")
#             continue
            
#         cx,cy,cz = bd.get_position(frame)
#         if cx is None or cy is None or cz is None:
#             print("No ball detected")
#             continue

#         measured.append((cx,cy))
#         print(f"Ball position: {cx}, {cy}, {cz}")
        
#         kf.correct(np.array([[cx], [cy]], np.float32))
#         tp = kf.predict()
#         predicted.append((int(tp[0].item()), int(tp[1].item())))
        
#         cv2.circle(frame, (int(tp[0]), int(tp[1])), 20, (255,0,0), 4)
#         # Display frame
#         cv2.imshow("Ball Detection", frame)
#         # if cv2.waitKey(1) == ord('q'):
#         #     break
#     except Exception as e:
#         print(f"Error: {e}")
#         break

    
#     print('Terminating ...')
#     camera.release()
#     cv2.destroyAllWindows()

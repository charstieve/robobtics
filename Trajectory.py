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

camera = cv2.VideoCapture(0)
measured = []
predicted = []

# load detector
bd = BallDetection()
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
while True:
    try: 
        _, frame = camera.read()
        
        if not _:
                print("Failed to capture frame")
                continue
            
        cx,cy,cz = bd.get_position(frame)
        measured.append((cx,cy))
        print(f"Ball position: {cx}, {cy}, {cz}")
        
        kf.correct((cx,cy))
        tp = kf.predict()
        predicted.append((int(tp[0]), int(tp[1])))
        
        cv2.circle(frame, (int(tp[0]), int(tp[1])), 20, (255,0,0), 4)
        # Display frame
        cv2.imshow("Ball Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        
        
        
    except:
        break
    
    print('Terminating ...')
    camera.release()
    cv2.destroyAllWindows()

#!/usr/bin/env python3
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
# init filter
kf = cv2.KalmanFilter(4,2)

# initial state = [x, y, delta_x, delta_y] 
# x, y =  position , delta_x, delta_y = velocity
kf.statePre = np.zeros((4, 1), dtype=np.float32)
# transition matrix -- 1 inch = 96 pixels so apply conversion?
kf.transitionMatrix = (np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=np.float32) * 96) 
# measurement matrix
kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]], dtype=np.float32)
# process noise convariance matrix
kf.processNoiseCov = (np.array([[1, 0, 0, 0], 
                                [0, 1, 0, 0], 
                                [0, 0, 1, 0], 
                                [0, 0, 0, 1]], np.float32) * 0.0001)
# measurement noise convariance matrix
kf.measurementNoiseCov = np.array([[1, 0],
                                       [0, 1]], dtype=np.float32)
# error covariance matrix
# Error covariance matrix
kf.errorCovPost = np.eye(4, dtype=np.float32)
print("hi")
if __name__ == '__main__':
    print("hi")
    # load live camera processing
    camera = cv2.VideoCapture(0)
    bd = BallDetection() # load detector
    
    # read first frame
    ret, frame = camera.read()
    # init first position
    cx,cy,cz = bd.get_position(frame)
    if cx is None or cy is None or cz is None:
        kf.statePre = np.array([[0], [0], [0], [0]], dtype=np.float32)
    else:
        kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

    while True:
        print("in here")
        # read each frame
        ret, frame = camera.read()
        if not ret:
            break
        
        # get prediction from kf
        prediction = kf.predict()
        pre_x, pre_y = int(prediction[0]), int(prediction[1])
        predicted.append((pre_x.item(), pre_y.item()))
        
        cv2.circle(frame, (pre_x, pre_y), 20, (255,0,0), -1)
        
        # TODO: do something with this prediction -- 
        # if pre_x, pre_y in robot range: 
        #   hit ball away
            
        cx,cy,cz = bd.get_position(frame)
        print(f"Ball position: {cx}, {cy}, {cz}")
        # unsure why repeat: ret, frame = camera.read()
        if cx is None or cy is None or cz is None:
            print("No ball detected")
            continue
        
        measured.append((cx,cy))
        
        # update kf with actual position
        kf.correct(np.array([[cx], [cy]], np.float32))
        
        # display frame
        cv2.imshow("Ball Tracking", frame)
        # unsure why repeat: cv2.imshow("Ball Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()
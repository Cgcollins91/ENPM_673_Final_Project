#!/usr/bin/env python
# -*- coding: utf-8 -*-

# detect_aruco.py

"""
 -----------------------------------------------------------
 Detect Aruco tags with camera and publish their pose relative to camera 

 Assumes intrinsic camera calibration is known and can be provided in CALIB_FILE
 -----------------------------------------------------------

See ArucoDetector class docstring
Topics
------
• **Subscribes**
  ├─ /csi_cam_0/image_raw    Image, BGR   Raw Camera Feed
• **Publishes**
    ├─ /aruco_detections       PoseStamped   Pose of detected Aruco tag

Referenced: Tommy Chang and Samer Charifa Lecture Notes, openCV documentation

 """

import rospy, cv2, cv_bridge
import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, CameraInfo
from pathlib import Path

aruco         = cv2.aruco
IMAGE_TOPIC   = '/csi_cam_0/image_raw'

# Camera Calibration Matrix and Distortion Coefficients from Project 3
CALIB_FILE    = Path('/home/jetson/ros_ws/src/detector/detector/calib_cam.npz')
ARUCO_DICT    = aruco.DICT_6X6_1000   # Dictionary used for ArUco markers
MARKER_SIZE   = 0.10                  # Marker side length (m)
DEBUG_WINDOW  = True


if not CALIB_FILE.is_file():
    raise IOError('Cannot find %s' % CALIB_FILE)

with np.load(CALIB_FILE) as X:
    cameraMatrix, distCoeffs = X['cameraMatrix'], X['distCoeffs']
rospy.loginfo('Loaded calibration from %s' % CALIB_FILE)

# ------------- Aruco Parameters ------------------------------------------------
# Tuned manually
aruco_dict     = aruco.getPredefinedDictionary(ARUCO_DICT)
aruco_params   = aruco.DetectorParameters_create()
aruco_params.adaptiveThreshWinSizeMin    = 3   
aruco_params.adaptiveThreshWinSizeMax    = 10   # default 23 
aruco_params.adaptiveThreshWinSizeStep   = 3    # default 10 
aruco_params.minMarkerPerimeterRate      = 0.02 # default 0.03
aruco_params.maxMarkerPerimeterRate      = 4.0    
aruco_params.minCornerDistanceRate       = 0.03  # default 0.05
aruco_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
aruco_params.cornerRefinementWinSize     = 4     # 3–7 work well
aruco_params.cornerRefinementMinAccuracy = 0.01


# ------------------------- Detector Node --------------------------------------
class ArucoDetector:
    def __init__(self):
        self.bridge   = cv_bridge.CvBridge()
        self.sub_img  = rospy.Subscriber(IMAGE_TOPIC, Image,
                                         self.cb_image, queue_size=1,
                                         buff_size=2**24)
        self.pub_pose = rospy.Publisher('aruco_detections',
                                        PoseStamped, queue_size=10)


    def cb_image(self, msg):
        img   = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        # Apply CLAHE to balance brightness
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray  = clahe.apply(gray)

        # Detect ArUco markers
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict,
                                              parameters=aruco_params)
        # Get pose vector of markers
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                                  corners, MARKER_SIZE,
                                  cameraMatrix, distCoeffs)
           
            # rvecs[i] is the rotation    vector of marker i
            # tvecs[i] is the translation vector of marker i
            rvecs = rvecs.reshape(-1,3)     # (N,3)
            tvecs = tvecs.reshape(-1,3)     # (N,3)
            ids   = ids .flatten()          # (N,)

            # Choose marker furthest away from camera for control
            dists = np.linalg.norm(tvecs, axis=1)      
            k     = np.argmax(dists)                   

            # Only use furthest marker 
            rvec, tvec, mid = rvecs[k], tvecs[k], ids[k]

            # Create Aruco Tag Pose Message
            pose                = PoseStamped()
            pose.header         = msg.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = tvec
            pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = rvec         
            pose.pose.orientation.w = 1.0

            # Publish Pose Message
            self.pub_pose.publish(pose)
            
            # Show live camera feed and detected markers if DEBUG_WINDOW = True
            if DEBUG_WINDOW: 
                aruco.drawAxis(img, cameraMatrix, distCoeffs,
                                   rvec, tvec, MARKER_SIZE * 0.5)
                aruco.drawDetectedMarkers(img, corners, ids)

        # Show live camera feed and detected markers if DEBUG_WINDOW = True
        # Topic: /aruco_detections
        if DEBUG_WINDOW:
            cv2.imshow('aruco', img)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                rospy.signal_shutdown('User exit')


if __name__ == '__main__':
    rospy.init_node('aruco_detector_file')
    ArucoDetector()
    rospy.loginfo('Aruco detector started.')
    rospy.spin()
    cv2.destroyAllWindows()


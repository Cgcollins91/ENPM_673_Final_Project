#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""""
optical_flow_node.py

ROS 1 node that detects dynamic obstacles and stop signs with dense
optical flow (Farneback) commanding brakes when either are detected in front of robot

Topics
------
• **Subscribes**
  ├─ <camera>/image_raw    (sensor_msgs/Image, BGR)  Raw Camera Feed
  ├─ /odom                 (nav_msgs/Odometry)       Robot Odometry
• **Publishes**
  ├─ /flow_image           (sensor_msgs/Image, BGR)  Residual-flow heatmap
  ├─ /debug/flow_image     (sensor_msgs/Image, BGR)  Raw dense-flow HSV
  ├─ /stop_sign            (sensor_msgs/Image, BGR)  Not Red filtered camera feed 
  ├─ /brake_flag           (std_msgs/Bool)           True if robot should remain stopped
  └─ /cmd_vel              (geometry_msgs/Twist)     Used to Stop Robot

Algorithm steps
---------------
1. Stop-sign detection:     HSV masking, then contour octagon fit (≥3 frames)
2. Dense Farneback flow:    Between consecutive greyscale frames
3. E matrix:                cv2.findEssentialMat  + recoverPose
4. Residual flow:           (observed - expected robot motion)
5. Hazard points:           Median absolute deviation threshold
6. Brake:                   If   |hazard| ≥ self.K_MIN_HAZARD,  |front| ≥ self.K_MIN_FRONT, |toward| ≥ self.K_MIN_TOWARD

Coordinate frames
-----------------
Camera Frame  : Camera intrinsics (``FX,FY,CX,CY``) from project-3 calib, translated by [0.09m, 0, 0.09m] from base_link (odom_frame)
Odom Frame    : Taken from /odom; delta pose used to scale translation

Referenced: Tommy Chang and Samer Charifa Lecture Notes, opencv documentation (i.e. https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0)
            https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
----------------------------------------------------------- 
"""

from std_msgs.msg import Bool         
import rospy, cv2, numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import tf.transformations as tr  

# Camera intrinsic parameters from project 3 calibration
FX = 382.27744214      
FY = 515.90384112
CX = 330.9000528     # Principal point X
CY = 185.86895897    # Principal point Y
K = np.array( [ [FX, 0, CX],
                [0, FY, CY],
                [0, 0,   1] 
              ], dtype=np.float64 )   

class FlowObstacleNode:
    """
    Dense Optical Flow Based Dynamic Obstacle Detector and Stop Sign Detector
    
    1. If paused (recent stop), publish flag to brake and return
    2. Detect stop sign, accumulate 3 consecutive hits, publish brake flag and stop robot for 3 seconds
    3. Compute Farneback flow  (prev→curr)
    4. Find Essential matrix, recover   R, t  triangulate points
    5. Residual = (observed - predicted) flow, use residual magnitude to determine "hazard" pixels
    6. If enough pixels simultaneously:
        • Marked as hazard
        • Are 10-40 cm in front of robot
        • Are pointing toward camera
       Then publish brake flag and stop robot for 3 seconds
    7. Publish debug images
    """

    def __init__(self):
        # ------------ ROS INTERFACES  -----------------------------------
        rospy.init_node('optical_flow_obstacles')

        # Subscribe to raw camera feed (BGR8), used for flow and stop sign detection
        cam       = rospy.get_param('~camera', '/csi_cam_0/image_raw')

        self.sub  = rospy.Subscriber(cam, 
                                     Image, 
                                     self.cb, 
                                     queue_size=1,
                                     buff_size=2**24)
        
        # Subscribe to odometry, used to scale t_cam from essential matrix to world distance
        self.odom_sub = rospy.Subscriber(
            '/odom',
            Odometry,
            self.odom_callback,
            queue_size=1)

        # Create Publisher for stop sign detection filtered image
        self.pub_stop  = rospy.Publisher(
                            'stop_sign',
                            Image, 
                            queue_size=1)
        
        # Create Publisher for flow image, residual heat-map (for rviz/image_view diagnostics)
        self.pub_img = rospy.Publisher(
                            'flow_image',
                            Image,
                            queue_size=1)
        
        # Create publisher for raw flow image
        self.pub_raw_img = rospy.Publisher(
                            'debug/flow_image',
                            Image,
                            queue_size=1)
        
        # Create publisher to command robot velocity
        self.cmd_vel_pub = rospy.Publisher(
                '/cmd_vel',
                Twist,              
                queue_size=1)
        
        # Create topic used in our PID Control to check whether robot is stopped
        self.brake_pub            = rospy.Publisher('/brake_flag', Bool, queue_size=1)
        # -------------------------------------------------------------------------

        self.pause_until          = rospy.Time(0)         # Time until robot can move again
        self.bridge               = CvBridge()            # cv_bridge object between ROS and OpenCV
        self.prev_frame_grey      = None                  # Previous frame in grayscale
        self.prev_pts             = None                  # Optical Flow points in previous frame
        self.prev_T               = np.eye(4)             # Previous odom frame Pose, used to determine robot motion                 
        self.T                    = np.eye(4)             # Current odom frame Pose, used to determine robot motion    
        self.farneback_params     = dict( winsize   = 9,  # Farneback Optical Flow Parameters
                                        pyr_scale  = 0.5,
                                        levels     = 2,
                                        iterations = 2,
                                        poly_n     = 3,
                                        poly_sigma = 1.1,
                                        flags      = 0)
        
        # Base Link to Camera Transformation
        T_base_cam        = np.eye(4)
        T_base_cam[0,3]   = 0.09
        T_base_cam[2,3]   = 0.09
        self.T_base_cam   = T_base_cam

        self.MAX_E_POINTS = 300          # Max optical flow points to use for essential matrix
        self.K_MIN_HAZARD = 70           # Minimum number of "hazard" points to trigger stop
        self.K_MIN_FRONT  = 40           # Minimum number of points in front of robot with velocity to trigger stop   
        self.K_MIN_TOWARD = 50           # Unused, but could be used for different scenario

        self.stop_sign_detected = False  # Flag to indicate if stop sign has been detected and robot has stopped after detection (False = Not Detected Yet, Not Stopped)     
        self.stop_detect_count  = 0      # Count of frames stop sign has been detected
        self.stopped            = False  # Flag to indicate if robot is stopped
        self.obstacle_detected  = False  # Flag to indicate if dynamic obstacle has been detected
        
    def _bookkeep(self, gray, xs, ys):
        # Bookkeep to store previous frame and points
        self.prev_frame_grey = gray
        self.prev_pts        = np.column_stack((xs, ys)).astype(np.float32)
        self.prev_T          = self.T.copy()


    def odom_callback(self, msg):
        # Extract robot (x, y, yaw) from odometry message
        
        # Get Position from Jetson's Odometry
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation            # quaternion x y z w
        R = tr.quaternion_matrix([q.x, q.y, q.z, q.w])  # 4×4
        T = R
        T[0:3, 3] = [p.x, p.y, p.z]

        self.T = T

    def stop_robot(self):
        # Stop Robot for 3 seconds and set Brake Flag to True
        self.cmd_vel_pub.publish(Twist())    
        self.brake_pub.publish(Bool(True))    
        self.pause_until = rospy.Time.now() + rospy.Duration(3.0)
        

    def detect_stop_sign(self, img_bgr):
        """
        Return bounding boxes (x, y, w, h) of candidate stop signs

        Detection pipeline:
            1.  Crop center 50 % vertically -- ignore sky/floor
            2.  Remove brown pixels (HSV)
            3.  Focus on red pixels (HSV)
            4.  Morphological open to remove noise
            5.  Find contours
            6.  Filter contours by area (≥ 1,000 pixels)
            7.  Approximate contours to polygon using Ramer–Douglas–Peucker algorithm
            8.  Filter by number of vertices (7-10) i.e. roughly octangonal and aspect ratio (roughly square)
            9.  Return bounding boxes of detected stop sign
            10. Publish debug image with result of steps 1-4

        """
        h, w, _ = img_bgr.shape

        # Scan middle 50% of image for stop sign
        roi     = img_bgr[h//4 : 3*h//4, :]
        hsv     = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Filter image to remove brown to focus on red 
        brown_lo   = np.array([10,  60,  40])
        brown_hi   = np.array([30, 255, 200])
        mask_brown = cv2.inRange(hsv, brown_lo, brown_hi)
        hsv        = cv2.bitwise_and(hsv, hsv, mask=cv2.bitwise_not(mask_brown))

        # Filter Image to focus on red
        LH, LS, LV = 168, 144,  75
        UH, US, UV = 179, 255, 115
        lower      = np.array([LH, LS, LV], dtype=np.uint8)
        upper      = np.array([UH, US, UV], dtype=np.uint8)
        mask_red   = cv2.inRange(hsv, lower, upper)

        # Remove noise, fill small holes
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, 1)

        # Find contours 
        cnts, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1000:    # If detected area is less than 1000 pixels, skip
                continue
            
            # Compute perimeter and approximate contour using 
            # Ramer–Douglas–Peucker algorithm
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03*peri, True)  # 3 % is looser

            # Check if approximated contour has 7-10 vertices i.e. roughly octagonal
            # and if width/height ratio is roughly square (==1)
            if 7 <= len(approx) <= 10:
                x, y, w_box, h_box = cv2.boundingRect(approx)
                y += h//4
                if 0.7 < float(w_box)/h_box < 1.3:   # Roughly Square Check    
                    detections.append((x, y, w_box, h_box))
                    # cv2.rectangle(roi, (x,y), (x+w_box, y+h_box), (0,255,0), 2)  # uncomment for debug

        # Debug visualisation 
        dbg = cv2.cvtColor(mask_red, cv2.COLOR_GRAY2BGR)
        self.pub_stop.publish(self.bridge.cv2_to_imgmsg(dbg, 'bgr8'))

        return detections


    def cb(self, msg):
        """ 
        Main Callback: Optical Flow and Stop Sign Detection
        Detect dynamic obstacles and stop sign using dense optical flow
        and stop robot if stop sign or dynamic obstacle is detected, 
        Only stop for stop sign once, while multiple stops for dynamic obstacles can be triggered
        see Class docstring
        """

        next_frame       = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        next_frame_grey  = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        h, w             = next_frame_grey.shape

        # Check if a stop sign is detected with 1,000 pxiels area, and it continues to be detected
        # for 3 frames, then stop robot (only once)
        if self.stop_sign_detected == False:
            stop_img = next_frame.copy()
            boxes    = self.detect_stop_sign(next_frame)

            for (x_box,y_box,w_box,h_box) in boxes:
                # cv2.rectangle(stop_img, (x_box,y_box), (x_box+w_box, y_box+h_box), (0,255,0), 2)
                
                # if any detected stop sign is larger than 1000 pixels and 
                # it is detected for 3 frames then stop robot
                if w_box*h_box > 1000:
                    self.stop_detect_count += 1
                    if self.stop_detect_count > 3:

                        rospy.logwarn("Stop Sign Detected, Stopping Robot")
                        self.stop_sign_detected = True
                        self.stop_robot()

        # Guard to check if robot is in stopped state (Recently Detected Stop Sign or Dynamic Obstacle)
        if self.pause_until > rospy.Time.now():
            self._bookkeep(next_frame_grey, None, None)
            self.brake_pub.publish(Bool(True))
            return
        
        # If we make it here, we are not stopped, so reset Brake Flag
        self.brake_pub.publish(Bool(False))

        # Initialization
        if self.prev_frame_grey  is None:
            self._bookkeep(next_frame_grey, None, None)
            return
        
        # Calculate Dense Optical Flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
                self.prev_frame_grey, 
                next_frame_grey, 
                None, 
                **self.farneback_params)
        
        # Calculate magnitude and angle of 2D vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 

        # Threshold flow magnitude to filter out small movements
        thr_flow   = 1.0 if mag.mean() > 0.1 else 0.2
        mask       = mag > thr_flow

        # Use mask to get "Good" points (really our potential dynamic obstacle points)
        ys, xs           = np.nonzero(mask)  # pixel row/col
        next_good_points = np.column_stack((xs, ys)).astype(np.float32)  

        # Check if we have enough points and have prior points 
        if self.prev_pts is None or len(self.prev_pts) < 5 or len(next_good_points) < 5:
            self._bookkeep(next_frame_grey, xs, ys)
            return
        
        # Make both arrays same length and ensure same type
        N        = min(len(self.prev_pts), len(next_good_points))
        p0       =    self.prev_pts[:N].astype(np.float32)
        p1       = next_good_points[:N].astype(np.float32)

        # Randomly sample points if we have too many        
        if N > self.MAX_E_POINTS:
            idx        = np.random.choice(N, self.MAX_E_POINTS, replace=False)
            p1_sample  = p1[idx]
            p0_sample  = p0[idx]
        else:
            p1_sample  = p1
            p0_sample  = p0

        # Draw raw dense optical flow with hsv colour map
        hsv         = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255   # full saturation
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)   # 0-179
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255,
                                    cv2.NORM_MINMAX).astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.pub_raw_img.publish(
                self.bridge.cv2_to_imgmsg(bgr, encoding='bgr8'))
        

        # Calculate Essential Matrix using RANSAC
        #   We already have our intrinsic camera properties (K) from project 3, 
        #   So we use the essential matrix to get extrinsic camera properties (R,t)
        #   Essential matrix is 3x3 matrix relating 
        #   previous step and current time step epipolar geometry so we can calculate 
        #   relative motion between the two frames
        E, inliers = cv2.findEssentialMat(p0_sample, 
                                          p1_sample,
                                          focal     = FX,
                                          pp        = (CX, CY), 
                                          method    = cv2.RANSAC,
                                          prob      = 0.999, 
                                          threshold = 1.0)
        
        if E is None or inliers is None:  # RANSAC failed, book-keep and continue
            self._bookkeep(next_frame_grey, xs, ys)
            return
        
        E_size = E.shape
        
        # if multiple solutions for E, select the E with the largest number of inliers indicating 
        # positive depth in front of camera
        if E_size[0] > 3:  
            best_inliers = -1
            best_E = None
            for i in range(0, E_size[0], 3):
                E_i = E[i:i+3, :]

                        # Get Rotation and Translation Matrix from previous step to current frame
                n_inliers, R_i, t_i, _ = cv2.recoverPose(E, p0_in, p1_in, 
                                                        cameraMatrix=K)
                
                if n_inliers > best_inliers:
                    best_inliers = n_inliers
                    best_E = E_i
                    best_R = R_i
                    best_t = t_i
            E = best_E
            R_cam = best_R
            t_cam = best_t
        else:  # E is 3x3
            # Get Rotation and Translation Matrix from previous step to current frame
            n_inliers, R_cam, t_cam, _ = cv2.recoverPose(E, p0_sample, p1_sample,
                                                        cameraMatrix=K)


        # Get inliers from RANSAC
        inliers = inliers.ravel().astype(bool)
        p0_in   = p0_sample[inliers] 
        p1_in   = p1_sample[inliers]

        if len(p0_in) < 5:  # Not enough points, book keep and continue
            self._bookkeep(next_frame_grey, xs, ys)
            return
        
        # Get how far robot has moved in world
        odom_T0_T1   = np.linalg.inv(self.prev_T).dot(self.T)

        # Get how far camera has moved in world
        T_odom_cam   = odom_T0_T1.dot(self.T_base_cam)    
        t_cam_world  = T_odom_cam[0:3, 3]                 
        delta_dist   = np.linalg.norm(t_cam_world)
        
        # If we haven't moved more than 5 cm, skip, noise at rest can lead to false positives
        if delta_dist < 0.05:   
            self._bookkeep(next_frame_grey, xs, ys)
            return
        
        # Scale translation vector to real world coordinates
        t_cam        = t_cam * delta_dist                      

        # Get Rotation Matrix from previous step to current frame
        I   = np.hstack( (np.eye(3), np.zeros((3,1)) ) )
        R_t = np.hstack((R_cam, t_cam.reshape(3,1)))  # [R|t]

        # Get Projection Matrix for previous and current frame
        P0  = np.dot(K, I)     # [K|0] 3x4 Projection Matrix for previous frame
        P1  = np.dot(K, R_t)   # [K|R] 3x4 Projection Matrix for current  frame

        # Triangulate inliers to get 3D points in camera frame
        triangulated_points_h = cv2.triangulatePoints(P0, P1, p0_in.T, p1_in.T)   # 4×N homogeneous
        w_pts4d_h             = triangulated_points_h[3]                          # Depth or homogenous scale factor
        pts3d                 = (triangulated_points_h[:3] / w_pts4d_h).T         # N×3, in Camera Frame


        X0 = pts3d.T     # 3×N

        # Transform inlier points to camera frame using same [R|t] from triangulation
        X1 = R_cam.dot(X0) + t_cam.reshape(3,1)   # 3×N

        # Project both sets back to image plane
        p0_hat = K.dot(X0 / X0[2]) # 3×N homogeneous
        p1_hat = K.dot(X1 / X1[2]) # 3×N homogeneous

        # Transpose and take only x,y coordinates
        p0_hat = p0_hat[:2].T # N×2
        p1_hat = p1_hat[:2].T # N×2

        # Expected Flow if Robot had not moved
        flow_pred = p1_hat - p0_hat

        # Residual flow = Velocity of Points - Robot Velocity
        residual  = (p1_in - p0_in) - flow_pred
        res_mag   = np.linalg.norm(residual, axis=1)

        # Get mad (median absolute deviation) and create residual magnitude threshold for mask to filter
        # Get "Hazard" points, i.e points with large residuals
        mad       = np.median(np.abs(res_mag - np.median(res_mag))) + 1e-6 
        thr       = max(3.0, 6.0*mad)      
        res_mask  = res_mag > thr
        hazard    = res_mask              # (N,) bool

        # Count how many detected flow points are in front of the robot
        z        = pts3d[:, 2]
        front    = (z > 0.1) & (z < 0.40) # 10 cm – 40 cm

        # Count how many detected flow points are moving toward the robot 
        u        = p1_in - np.array([CX, CY], np.float32)  # (N,2)
        u        = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-6)
        res_unit = residual / (res_mag[:,None] + 1e-6)
        toward   = (np.sum(u * res_unit, axis=1) >  0.2)    # i.e. coming towards robot

        # Convert to 1-D                                                          
        front  = front.ravel()
        toward = toward.ravel()
        hazard = hazard.ravel()

        # Get indices of hazard points
        hazard_idx = np.nonzero(res_mask)[0] 

        # Clip Hazard Points to image bounds and cast as int32
        x_h = p1_in[hazard_idx, 0].astype(np.int32)
        y_h = p1_in[hazard_idx, 1].astype(np.int32)
        x_h = np.clip(x_h, 0, w-1)
        y_h = np.clip(y_h, 0, h-1)
        res_mags    = res_mag[res_mask]

        # Count number of points in each category
        n_front   = np.count_nonzero(front)
        n_toward  = np.count_nonzero(toward)
        n_hazard  = np.count_nonzero(hazard)
        n_all     = np.count_nonzero(front & toward & hazard)

        # Log count of hazard, front, toward, and all 
        rospy.loginfo_throttle(
            1, "pts:%3d F:%2d T:%2d H:%2d ALL:%2d",
            len(res_mag), n_front, n_toward, n_hazard, n_all)


        # If we have enough points in front and toward the robot, stop the robot      
        if n_hazard  >=  self.K_MIN_HAZARD and \
            n_front  >=  self.K_MIN_FRONT and \
            n_toward >=  self.K_MIN_TOWARD :             
            rospy.logwarn("dynamic obstacle – braking "
                          "(front:%d toward:%d hazard:%d)",
                          np.count_nonzero(front),
                          np.count_nonzero(toward),
                          np.count_nonzero(hazard))
            self.stop_robot()
    
        # Publish Hazard Points to /flow_image topic
        res_img           = np.zeros((h, w), np.float32)
        res_img[y_h, x_h] = res_mags                 #
        
        # Normalize residual flows to 0-255
        if res_img.max() > 0:
            res_norm = (res_img / res_img.max() * 255).astype(np.uint8)
        else:
            res_norm = res_img.astype(np.uint8)

        vis = cv2.applyColorMap(res_norm, cv2.COLORMAP_JET)

        # Show hazard pixels as bright red
        vis[y_h, x_h] = (0, 0, 255)  

        # Update Flow Images
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(vis, 'bgr8'))
       
        # --- Book-Keep -----------------------------------------
        self.prev_frame_grey = next_frame_grey
        self.prev_pts        = np.column_stack((xs, ys)).astype(np.float32)  
        self.prev_T          = self.T.copy()

if __name__ == '__main__':
    FlowObstacleNode()
    rospy.spin()

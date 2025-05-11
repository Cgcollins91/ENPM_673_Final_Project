#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
-----------------------------------------------------------
PID Control to command robot to move towards detected aruco tag
-----------------------------------------------------------

Implements PID controller to command Jetson Racer to move towards detected aruco tag
Robot will move forward and turn to face furthest away detected aruco tag,
if no tag is detected, robot will search for a tag by moving forward slowly and oscillating
slightly left and right

See ArucoDualPID class docstring

Topics
------
• **Subscribes**
    ├─ /aruco_detections       PoseStamped   Pose of detected Aruco tag
    ├─ /brake_flag            Bool          Flag to stop robot
• **Publishes**
    ├─ /cmd_vel               Twist         Robot velocity command


Referenced: Tommy Chang and Samer Charifa Lecture Notes, openCV documentation

"""


import rospy, math
from geometry_msgs.msg import PoseStamped, Twist
from tf.transformations import euler_from_matrix
import numpy as np
from std_msgs.msg import Bool


# ─── Control Parameters ────────────────────────────────────────────────────────
TARGET_DIST      = 0.03               # m to aruco tag
KP_X, KI_X, KD_X = 0.7, 0.01, 0.10    # forward
KP_Y, KI_Y, KD_Y = 0.5, 0.01, 0.05    # heading
MAX_FWD          = 0.2      # m/s
MAX_YAW          = 0.3      # rad/s
TAG_TIMEOUT      = 1.0      # [s]  if no detection for this long → search
SEARCH_FWD       = 0.05     # [m/s] slow creep
SEARCH_YAW       = 0.05     # [rad/s] slow spin (sign chosen below)
# ──────────────────────────────────────────────────────────────────────────


class PID:
    """ Calculates PID control signal from error (position and heading) 
    
    Parameters
    ----------
    kp : float Proportional gain
    ki : float Integral gain
    kd : float Derivative gain
    """

    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.adaptive_kp = kp
        self.e_prev = 0.0
        self.i_sum  = 0.0
        self.t_prev = None # time of last step
        self.brake_active = False

    def step(self, error, curr_time):
        # Calculate PID Control Signal 

        if self.t_prev is None: # return 0.0 for control signal, but save error
            self.t_prev, self.e_prev = curr_time, error
            return 0.0
        
        dt = (curr_time - self.t_prev).to_sec()
        if dt <= 0.0: # return 0.0 if no time has passed
            return 0.0
        
        self.t_prev    = curr_time
        self.i_sum    += error*dt                  # Integral Term
        d              = (error - self.e_prev)/dt  # Derivative Term
        self.e_prev    = error 
        control_signal = self.kp*error + self.ki*self.i_sum + self.kd*d

        return control_signal


class ArucoDualPID:
    def __init__(self):
        # Brake Flag is set by optical_flow_dense.py and 
        # returns True for 3 seconds the first time a stop sign is detected
        # and returns True for 3 seconds every time a dynamic obstacle is detected
        # (e.g. a person walking in front of the robot)
        self.brake_active = False
        rospy.Subscriber('/brake_flag', Bool, self.cb_brake,  queue_size=1)

        # Initialize PID controller
        self.pid_x = PID(KP_X, KI_X, KD_X) # Forward Control
        self.pid_y = PID(KP_Y, KI_Y, KD_Y) # Heading Control

        # Publish PID Control Signal to /cmd_vel
        self.pub   = rospy.Publisher('/cmd_vel', Twist, queue_size=5)

        # /aruco detections topic is published by detect_aruco.py
        # and detected aruco tags and publishes the pose of tag relative to camera
        rospy.Subscriber('/aruco_detections', PoseStamped,
                         self.cb_pose, queue_size=5)
        self.last_seen = rospy.Time(0)            # Store last time a tag was detected
        self.search_dir = 1                       # ← NEW  (alternate spin dir)
        rospy.Timer(rospy.Duration(0.05), self.cb_search)    # 20 Hz


    def cb_brake(self, msg):
        """ Ge"""
        self.brake_active = msg.data          # if True stay braked

    def cb_pose(self, msg):
        """ Called while a aruco tag is detected. """
        # If brake is active, stop the robot and don't run PID
        if self.brake_active:
            self.pub.publish(Twist())         # keep robot stopped
            return                            # and don't run PID

        self.last_seen = msg.header.stamp    # Save Time Stamp of last detection of a aruco tag
        z = msg.pose.position.z              # Forward distance (m) to aruco tag
        x = msg.pose.position.x              # right (+) / left (−) offset (m) to tag

        # ── Forward Control ───────────────────────
        dist_error = z - TARGET_DIST
        u_fwd      = self.pid_x.step(dist_error, msg.header.stamp)
        u_fwd      = max(-MAX_FWD, min(MAX_FWD, u_fwd))

        # ── Heading Control ───────────────────────
        # (small-angle tan θ ≈ θ).   Gives radians.
        heading_err = math.atan2(x, z)       # sign: +right, −left
        u_yaw       = self.pid_y.step(heading_err, msg.header.stamp)
        u_yaw       = max(-MAX_YAW, min(MAX_YAW, u_yaw))

        # ── Publish Control Signal ──────────────────────────────────
        tw           = Twist()
        tw.linear.x  = u_fwd
        tw.angular.z = -u_yaw        # convention: +z yaw = left
        self.pub.publish(tw)

    def cb_search(self, _evt):
        """
        Called every 50ms to check if tag was detected recently
        If tag was not detected for TAG_TIMEOUT, search for it, 
        if tag was detected recently, do nothing
        """
        # If tag seen recently 
        if (rospy.Time.now() - self.last_seen).to_sec() < TAG_TIMEOUT:
            return                            # tag seen recently, do nothing
        if self.brake_active:
            self.pub.publish(Twist())         # keep robot stopped
            return                            # skip PID 

        # ------------ Search ------------------------------------------
        # If no tag seen for a while, search for it 
        # by moving forward with alternating small yaw
        tw = Twist()
        tw.linear.x  =  SEARCH_FWD
        tw.angular.z =  self.search_dir * SEARCH_YAW
        self.pub.publish(tw)

        # Flip steering direction every 5 seconds
        if (rospy.Time.now().secs // 5) % 2 == 0:
            self.search_dir = 1
        else:
            self.search_dir = -1


if __name__ == '__main__':
    rospy.init_node('aruco_pid')
    ArucoDualPID()
    rospy.loginfo('Distance + heading PID running')
    rospy.spin()

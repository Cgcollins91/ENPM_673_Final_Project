#!/usr/bin/env python
# -*- coding: utf-8 -*-

# vanishing_point_node.py

"""

-----------------------------------------------------------
Detect Vanishing Point in camera feed and overlay it on image
-----------------------------------------------------------

See VanishingPoiontNode class docstring

Topics
------
• **Subscribes**
  ├─ <camera>/image_raw    Image, BGR   Raw Camera Feed      
• **Publishes**
  ├─ /csi_cam_0/image_vp   Image, BGR   Camera Feed with VP Overlay

"""

import rospy, cv2, numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VanishingPointNode:
    """
    Detect Vanishing Point in camera feed and overlay it on image
    
    Pipeline:
    1. Convert to grayscale, apply CLAHE to balance brightness, apply Gaussian blur
    2. Canny edge detection
    3. Detect lines using Hough Transform
    4. Build Homogeneous Lines
    5. Build least-squares vanishing-point
    6. Apply Exponential Moving Average to smooth the VP
    7. Overlay the VP on the camera feed
    8. Publish the image with VP overlay

    Referenced: Tommy Chang and Samer Charifa Lecture Notes, openCV documentation

    """
    def __init__(self):
        rospy.init_node('vanishing_point_overlay')

        cam_topic   = rospy.get_param('~input',  '/csi_cam_0/image_raw')
        out_topic   = rospy.get_param('~output', '/csi_cam_0/image_vp')
        self.bridge = CvBridge()
        self.sub    = rospy.Subscriber(cam_topic, Image, self.cb, queue_size=1,
                                       buff_size=2**24)
        self.pub    = rospy.Publisher(out_topic, Image, queue_size=1)

        self.vp_filt      = None           # EMA of Vanishing Point
        self.last_seen_ts = rospy.Time(0)  # last time VP was detected
        self.alpha        = 0.3            # EMA Smoothing factor     

    # Main Callback:
    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w  = frame.shape[:2]


        # Convert to grayscale, apply CLAHE to balance brightness, and apply Gaussian blur
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray   = clahe.apply(gray)
        gray   = cv2.GaussianBlur(gray, (9, 9), 0)

        # Canny edge detection
        median = np.median(gray)
        edges  = cv2.Canny(gray, 
                           median*0.6,  # low threshold --   60% of Median
                           median*1.3,  # high threshold -- 130% of Median
                           apertureSize=3)

        # Detect lines using Hough Transform
        segs = cv2.HoughLinesP(edges,
                       1,                   # Resolution in pixels
                       np.pi/180,           # Angle resolution in radians (1 degree)
                       threshold     = 50,  # Minimum number of votes
                       minLineLength = 20,  # Minimum length of line (pixels)
                       maxLineGap    = 10)  # Maximum gap between line segments (pixels)
        
        if segs is None: # No Lines Detected
            segs = []                     
        else:
            segs = segs.reshape(-1, 4)    # (N,1,4) --> (N,4)

       # Get the length of each detected lines 
        if segs.size:
            lengths = np.hypot(segs[:, 2] - segs[:, 0],
                            segs[:, 3] - segs[:, 1])

            # Convert each detected segment into a homogeneous line l = p1 × p2
            # where l = (a, b, c) and  a·x + b·y + c = 0  for every point (x,y) on l
            lines   = []
            weights = []                      
            for (x1, y1, x2, y2), L in zip(segs, lengths):
                p1 = np.array([x1, y1, 1.0])
                p2 = np.array([x2, y2, 1.0])
                a, b, c = np.cross(p1, p2).astype(float)

                # Discard almost-horizontal lines
                if abs(a) / np.hypot(a, b) < 0.3:     #  Theta < 17 degrees
                    continue

                lines.append([a, b, c])
                weights.append(L)                     

            # ---------- Least Squares Vanishing Point -------------
            # Calculate vanishing point using least-squares to solve for vp
            if len(lines) >= 2:
                L  = np.asarray(lines)               # (N,3)
                A  =  L[:, :2]  # 
                b  = -L[:, 2]

                # weight each equation by sqrt(length) so longer lines count more
                w  = np.sqrt(np.asarray(weights, dtype=np.float64))
                A_w = A * w[:, None]
                b_w = b * w

                vp, _, _, _ = np.linalg.lstsq(A_w, b_w, rcond=-1)
                

        # ---------- Exponential Moving Average ------------------
        if vp is not None: # VP detected this frame?
            if self.vp_filt is None:
                self.vp_filt = np.array(vp, int)   # Initialize

            else: # Vanishing Point Detected, Update EMA
                self.vp_filt = (self.alpha * np.array(vp, int) +
                                (1.0 - self.alpha) * self.vp_filt)
            self.last_seen_ts = msg.header.stamp

        else: # Time out
            if (msg.header.stamp - self.last_seen_ts).to_sec() > 1.0:
                self.vp_filt = None

        # Vanishing Point Overlay on top of camera feed
        vp  = self.vp_filt
        vis = frame.copy()

        if self.vp_filt is not None and len(vp) == 2:
            x, y = map(int, self.vp_filt)  # Ensure integers for plotting
            s    = 5                       # half-size of cross to be plotted (pixels)

            # Draw Cross at Vanishing Point
            cv2.line(vis, (x-s, y-s), (x+s, y+s),
                    (0, 0, 255), 2)
            cv2.line(vis, (x-s, y+s), (x+s, y-s),
                    (0, 0, 255), 2)
            
            # Add Label with coordinates 
            cv2.putText(vis, "VP (%d,%d)" % (x,y), (x+5, y+20),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (0,0,255), 2)
        # publish
        self.pub.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))

if __name__ == '__main__':
    try:
        VanishingPointNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

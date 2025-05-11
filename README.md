Chris Collins, UID 110697305

ENPM 673 Perception for Autonomous Robots Final Project, A Waveshare JetRacer Mobile Robot was used to:
------------
1. Autonomously navigate a course defined by pieces of paper with aruco tags by
  - Detecting Aruco Tags with camera feed and determine their pose relative to Jetracer's camera frame
  - Send PID control using distance/heading error between robot pose and aruco pose
2. Detect Stop Sign with camera feed and stop once detected
3. Detect dynamic obstacle in front of the camera and stop if detected
4. Calculate the Vanishing Point of the camera image and overlay it

![Screenshot 2025-05-11 at 15-25-55 JetRacer AI Kit AI Racing Robot Powered by Jetson Nano](https://github.com/user-attachments/assets/aa5daddd-c872-45ec-ae7b-1fcc65728f55)

Video of my console for test run (On a Linux VM):
https://www.dropbox.com/scl/fi/pbdnabxqzh8mnblatljcx/Chris_Collins_ENPM_673_Final_Project_Console_View.mp4?rlkey=bu22osijhsr5021aig0rnqxge&st=lxdtldzn&dl=0

Video of Jetracer performing run (without dynamic obstacle):
https://www.dropbox.com/scl/fi/c7b4ur48rfxhcs7nv7c3s/Chris_Collins_ENPM_673_Final_Project_View_of_Robot.mp4?rlkey=lhdcmpmqtyejqa6xkip9jhgd2&st=ph2ddqag&dl=0


For test run the following steps were performed:
1. Turn on Jetracer, place robot in front of course then start ros core, chassis, and camera node  with these commands on the Jetson Nano:
  - roscore                              # Start the robot master node
  - roslaunch jetracer jetracer.launch   # Start the robot chassis node
  - roslaunch jetracer csi_camera.launch # Start Camera
  - On Linux VM to see camera and optical_flow/stop_sign/vanishing_point debug images:      rosrun rqt_image_view rqt_image_view
2. From my Linux VM, first open multiple rqt_image_viewers (Command Above), then run the below commands in seperate terminals:
- ./run_detector.bash  
- ./run_optical_flow_dense.bash
- ./run_vanishing.bash
3.Finally Run  ./run_course.bash    (also in seperate terminal, this will command robot movement)

The code base consists of 4 major files:
- src/detector/detector/aruco_pid.py             --> PID Control
- src/detector/detector/detect_aruco.py          --> Detect Aruco Tags and publish their pose
- src/detector/detector/optical_flow_dense.py    --> Detect Dynamic Obstacle and Stop Sign and publish command to brake robot for 3 seconds if detected
- src/detector/detector/vanishing_point_node.py  --> Detect Vanishing Points and Overlay on top of camera feed


optical_flow_dense.py
---------------
     **Subscribes**
        ├─ <camera>/image_raw    (sensor_msgs/Image, BGR8) 
        ├─ /odom                 (nav_msgs/Odometry)
      • **Publishes**
        ├─ /flow_image           (sensor_msgs/Image, BGR8)  residual-flow heatmap
        ├─ /debug/flow_image     (sensor_msgs/Image, BGR8)  raw dense-flow HSV
        ├─ /stop_sign            (sensor_msgs/Image, BGR8)  Not Red filtered camera feed 
        ├─ /brake_flag           (std_msgs/Bool)            True if robot should remain stopped
        └─ /cmd_vel              (geometry_msgs/Twist)      Used to Stop Robot
      
      Algorithm steps
      ---------------
      1. Stop-sign detection:     HSV masking, then contour octagon fit (≥3 frames)
      2. Dense Farneback flow:    Between consecutive greyscale frames
      3. E matrix:                cv2.findEssentialMat  + recoverPose
      4. Residual flow:           (observed - expected robot motion)
      5. Hazard points:           Median absolute deviation threshold
      6. Brake:                   If   |hazard| ≥ 70,  |front| ≥ 40, |toward| ≥ 50
      
      Coordinate frames
      -----------------
      Camera Frame  : Camera intrinsics (``FX,FY,CX,CY``) from project-3 calib, translated by [0.09m, 0, 0.09m] from base_link (odom_frame)
      Odom Frame    : Taken from /odom; delta pose used to scale translation

      Main Callback
      -----------------
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
        -----------------
        
        detect_stop_sign:
         -----------------
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

    Referenced: Tommy Chang and Samer Charifa Lecture Notes, opencv documentation (i.e. https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0)
                https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
      


vanishing_point_node.py
---------------
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


aruco_pid.py
------------
PID Control to command robot to move towards detected aruco tag

  Implements PID control to command Jetson Racer to move towards detected aruco tag
  Jetson will move forward and turn to face furthest away detected aruco tag,
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

detect_aruco.py
---------------
 Detect Aruco tags with camera and publish their pose relative to camera 

 Assumes intrinsic camera calibration is known and can be provided in CALIB_FILE (For this project leveraged ENPM 673 Project 3 camera calibration)

  See ArucoDetector class docstring

    Topics
    ------
    • **Subscribes**
      ├─ /csi_cam_0/image_raw    Image, BGR   Raw Camera Feed
    • **Publishes**
        ├─ /aruco_detections       PoseStamped   Pose of detected Aruco tag

    Referenced: Tommy Chang and Samer Charifa Lecture Notes, openCV documentation



Notes:
- Aruco tags from 6x6 Dictionary with 100mm size markers from (https://chev.me/arucogen/) used
- Linux Virtual image that was provided by waveshare to write ROS code
- ROS Version:  Melodic
- Sensor Processing was done by High Performance desktop computer (Not by Jetson Nano)

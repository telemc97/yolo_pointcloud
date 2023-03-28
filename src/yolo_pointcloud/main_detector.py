import os
import sys
import time
import math
import cv2
import message_filters
import numpy as np
import rospkg
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from yolo_pointcloud.msg import BoundingBox, BoundingBoxes, YoloStereoDebug, PointConfidenceStamped

import yolo_pointcloud.tools.general as tools
from yolo_pointcloud.tools.pcl import PointCloud_gen

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ultralytics"))

from yolo_pointcloud.ultralytics.ultralytics.yolo.engine.model import YOLO
from yolo_pointcloud.ultralytics.ultralytics.yolo.utils.plotting import Annotator
from yolo_pointcloud.ultralytics.ultralytics.yolo.utils.torch_utils import select_device

pkg = rospkg.RosPack()
package_path = pkg.get_path('yolo_pointcloud')

class mainDetector:
    def __init__(self):

        #Initialize OpenCV Bridge
        self.br = CvBridge()
        self.matcher = cv2.BFMatcher()
        self.descriptor = cv2.ORB_create()


        #YOLO SECTION--------------------------------------------------------------------------------------------------------------------------------------------
        self.weights = rospy.get_param('~weights')
        self.matching_dist_coef = rospy.get_param('~matching_distance_coefficient', 0.25)
        self.conf_thres = rospy.get_param('~confidence', 0.7)
        self.iou_thres = rospy.get_param('~iou_thres', 0.45)
        self.agnostic = rospy.get_param('~agnostic_nms', True)
        self.max_det = rospy.get_param('~max_detections', 10)
        self.classes = rospy.get_param('~classes', None)
        self.deviceName = rospy.get_param('~device', '0')
        self.device = select_device(self.deviceName)
        self.network_img_size = rospy.get_param('~img_size', 480)
        self.half = rospy.get_param('~half', False)
        self.font_scale = rospy.get_param('~font_scale', 0.8)
        self.thickness = rospy.get_param('~thickness', 2)


        #CAMERA SECTION-----------------------------------------------------------------------------------------------------------------------------------------
        #ROS Parametres Camera0
        self.camera0_topic = rospy.get_param('~camera0_topic', '/stereo/left/image_raw') #Camera0 Topic
        self.camera0_info_topic = rospy.get_param('~camera0_info_topic', '/stereo/left/camera_info')
        self.detected_obj_topic0 = rospy.get_param('~detected_objects_topic0', '/yolo_pointcloud/left/detected_objects')
        self.publish_img = rospy.get_param('~publish_image', True) #Publish or not images
        #Subscriber Camera0
        self.imageRight = message_filters.Subscriber(self.camera0_topic, Image)
        self.cameraInfoRight = message_filters.Subscriber(self.camera0_info_topic, CameraInfo)

        #ROS Parametres Camera1
        self.camera1_topic = rospy.get_param('~camera1_topic', '/stereo/right/image_raw') #Camera1 Topic
        self.camera1_info_topic = rospy.get_param('~camera1_info_topic', '/stereo/right/camera_info')
        #Subscriber Camera1
        self.imageLeft = message_filters.Subscriber(self.camera1_topic, Image)
        self.cameraInfoLeft = message_filters.Subscriber(self.camera1_info_topic, CameraInfo)
        rospy.loginfo("Subscribed to image topics: %s, %s, %s, %s" % (self.camera0_topic, self.camera0_info_topic, self.camera1_topic, self.camera1_info_topic))

        
        #Define Publisher
        self.publisher_obj0 = rospy.Publisher('/yolo_pointcloud/detected_objects', BoundingBoxes, queue_size=10)
        self.publisher_img0 = rospy.Publisher('/yolo_pointcloud/detection_image', Image, queue_size=10)
        self.publisher_point_with_conf = rospy.Publisher('/yolov5/point_with_confidence', PointConfidenceStamped, queue_size=10)

        self.publish_debug = rospy.Publisher('yolo_pointcloud_stereo_debug', YoloStereoDebug, queue_size=1)
        
        rospy.loginfo("Nodes Launched")

        #Load model
        self.model = YOLO(self.weights)

        self.synch = message_filters.ApproximateTimeSynchronizer([self.imageRight, self.cameraInfoRight, self.imageLeft, self.cameraInfoLeft], queue_size=10, slop=0.5)
        self.synch.registerCallback(self.callback)



    def callback(self, imageR, camInfoR, imageL, camInfoL):
        callback_start_time = time.time()

        bboxL = np.zeros(shape=(0,6), dtype=float)
        bboxR = np.zeros(shape=(0,6), dtype=float)

        self.cv_imageR = self.br.imgmsg_to_cv2(imageR, 'bgr8')
        self.cv_imageL = self.br.imgmsg_to_cv2(imageL, 'bgr8')

        #Left Camera Infos
        self.cam_height = camInfoL.height
        self.cam_width = camInfoL.width
        self.cam_camMatrix_Left = np.array(camInfoL.K).reshape((3,3))
        self.cam_R_Left = np.array(camInfoL.R).reshape((3,3))
        self.cam_P_Left = np.array(camInfoL.P).reshape((3,4))
        #Left Camera Infos
        self.cam_camMatrix_Right = np.array(camInfoR.K).reshape((3,3))
        self.cam_R_Right = np.array(camInfoR.R).reshape((3,3))
        self.cam_P_Right = np.array(camInfoR.P).reshape((3,4))

        im_viz = self.cv_imageL.copy()

        predictions = self.model.predict(source=[self.cv_imageL,self.cv_imageR])

        predictionsL = predictions[0].cpu().numpy()
        predictionsR = predictions[1].cpu().numpy()

        bboxL = predictionsL.boxes.xyxy
        bboxR = predictionsR.boxes.xyxy

        confL = predictionsL.boxes.conf
        confR = predictionsR.boxes.conf
        clsL = predictionsL.boxes.cls
        clsR = predictionsR.boxes.cls

        annotator = Annotator(im_viz, line_width=self.thickness, font_size=self.font_scale)

        det_bounds = BoundingBoxes()
        det_bounds.header = imageL.header
        det_bounds.image_header = imageL.header

        if (len(bboxL)>0 and len(bboxR)>0):

            if (len(bboxL)<len(bboxR)):
                bboxR = bboxR[:len(bboxL),:]
                confR = confR[:len(confL)]
            else:
                bboxL = bboxL[:len(bboxR),:]
                confL = confL[:len(confR)]

            matched = self.matches_gen(bboxL, bboxR)
            points = np.zeros(shape=(3,0), dtype=np.float32)

            for i in range(bboxL.shape[0]):
                center_x = int(((bboxL[i, 2]-bboxL[i, 0])/2) + bboxL[i, 0])
                center_y = int(((bboxL[i, 3]-bboxL[i, 1])/2) + bboxL[i, 1])

                for j in range(matched.shape[0]):
                    if ((center_x== matched[j,0]) and (center_y == matched[j,1])):
                        pointL = matched[j,:2]
                        pointR = matched[j,2:]
                        conf = (confL[j] + confR[j])/2 #Confidence of two detection (Average of left and right)
                        point = self.dist_calc(pointL, pointR)

                        if ((np.isnan(point).any()==False) and (np.isinf(point).any()==False)):

                            pointConfidenceStamped = PointConfidenceStamped()
                            pointConfidenceStamped.header = imageL.header
                            pointConfidenceStamped.point.x = point[0]
                            pointConfidenceStamped.point.y = point[1]
                            pointConfidenceStamped.point.z = point[2]
                            pointConfidenceStamped.confidence = conf
                            self.publisher_point_with_conf.publish(pointConfidenceStamped)

                        points = np.column_stack((points, point))

            pointcloud2 = PointCloud_gen(points, imageL.header.stamp)
            pointcloud2.pub_pcl()

            for i in range(bboxL.shape[0]):
                bboxmsg = BoundingBox()
                bboxmsg.Class = self.model.names[int(clsL[i])]
                bboxmsg.probability = confL[i]
                bboxmsg.xmin = int(bboxL[i,0])
                bboxmsg.ymin = int(bboxL[i,1])
                bboxmsg.xmax = int(bboxL[i,2])
                bboxmsg.ymax = int(bboxL[i,3])
                if self.publish_img:
                    label = f"{self.model.names[int(clsL[i])]} {confL[i]:.2f}"
                    annotator.box_label(box=bboxL[i,:], label=label)                
                det_bounds.bounding_boxes.append(bboxmsg)
       
        #Publish Messages
        self.publisher_obj0.publish(det_bounds)


        #Publish Images
        if self.publish_img:
            image_msg0 = self.br.cv2_to_imgmsg(im_viz, 'rgb8')
            image_msg0.header.frame_id = 'left'
            image_msg0.header.stamp = rospy.Time.now()
            self.publisher_img0.publish(image_msg0)     
       
        debug_msg = YoloStereoDebug()
        callback_end_time = time.time()
        total_exec_time = (callback_end_time-callback_start_time)*1000
        debug_msg.header.stamp = rospy.Time.now()
        debug_msg.proc_time = total_exec_time
        self.publish_debug.publish(debug_msg)


    def dist_calc (self, pointL_, pointR_):
        ptL = cv2.undistortPoints(src=pointL_, cameraMatrix=self.cam_camMatrix_Left, distCoeffs=None, R=self.cam_R_Left, P=self.cam_P_Left)
        ptR = cv2.undistortPoints(src=pointR_, cameraMatrix=self.cam_camMatrix_Right, distCoeffs=None, R=self.cam_R_Right, P=self.cam_P_Right)
        dist = cv2.triangulatePoints(self.cam_P_Left, self.cam_P_Right, ptL, ptR)
        distance_xyz = tools.homogenous_to_euclidian(dist)
        distance_xyz_pcl = np.array([abs(distance_xyz[2]), -distance_xyz[0], -distance_xyz[1]])
        return distance_xyz_pcl


    def matches_gen (self, boxL, boxR):
        imageLeftMono = cv2.cvtColor(self.cv_imageL, cv2.COLOR_BGR2GRAY)
        imageRightMono = cv2.cvtColor(self.cv_imageR, cv2.COLOR_BGR2GRAY)

        keypointsL = []
        descriptorsL = np.zeros(shape=(0,32), dtype=np.uint8)
        keypointsR = []
        descriptorsR = np.zeros(shape=(0,32), dtype=np.uint8)

        for i in range(boxL.shape[0]):
            center_left = int(((boxL[i, 2]-boxL[i, 0])/2) + boxL[i, 0]), int(((boxL[i, 3]-boxL[i, 1])/2) + boxL[i, 1])
            center_right = int(((boxR[i, 2]-boxR[i, 0])/2) + boxR[i, 0]), int(((boxR[i, 3]-boxR[i, 1])/2) + boxR[i, 1])
            
            kp_left = [cv2.KeyPoint(x=center_left[0], y=center_left[1], size=1)]
            kp_right = [cv2.KeyPoint(x=center_right[0], y=center_right[1], size=1)]

            # Left Descriptors
            kp_left, ds_left = self.descriptor.compute(imageLeftMono, kp_left)
            keypointsL.extend(kp_left)
            if type(ds_left)!= type(None):
                descriptorsL = np.vstack((descriptorsL, ds_left))
                
            #Right Descriptors
            kp_right, ds_right = self.descriptor.compute(imageRightMono, kp_right)
            keypointsR.extend(kp_right)
            if type(ds_right)!= type(None):
                descriptorsR = np.vstack((descriptorsR, ds_right))

        if type(descriptorsL)!= type(None) and type(descriptorsR)!= type(None):

            knnMatches = 2
            matches = self.matcher.knnMatch(descriptorsL, descriptorsR, k=knnMatches)
            good_matches = []
            tresholdDist = self.matching_dist_coef * math.sqrt((math.pow(imageLeftMono.shape[0],2) + math.pow(imageLeftMono.shape[1],2)))
            for i in range(len(matches)):
                for j in range(len(matches[0])):
                    match0 = keypointsL[matches[i][j].queryIdx].pt
                    match1 = keypointsR[matches[i][j].trainIdx].pt
                    dist = math.sqrt((match0[0] - match1[0]) * (match0[0] - match1[0]) + (match0[1] - match1[1]) * (match0[1] - match1[1]))
                    if (dist < tresholdDist and abs(match0[1]-match1[1])<1):
                        good_matches.append(matches[i][j])

        matched_kp = np.zeros(shape=(0,4), dtype=float)

        for mat in good_matches:
            imgL_idx = mat.queryIdx
            imgR_idx = mat.trainIdx

            ptL = keypointsL[imgL_idx].pt
            ptR = keypointsR[imgR_idx].pt

            matched_kp = np.vstack((matched_kp, np.array([ptL[0], ptL[1], ptR[0], ptR[1]], dtype=float)))
        
        return matched_kp
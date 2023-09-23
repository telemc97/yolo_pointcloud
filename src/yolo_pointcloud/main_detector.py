import os
import sys
import time
import math
import cv2
import message_filters
import numpy as np
import rospkg
import rospy
import tf
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PointStamped

from yolo_pointcloud.msg import BoundingBox, BoundingBoxes, YoloStereoDebug, PointConfidenceStamped

import yolo_pointcloud.tools.general as tools
from yolo_pointcloud.tools.det_collector import Collector

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
        self.matching_dist_coef = rospy.get_param('~matching_distance_coefficient', 0.15)
        self.deviceName = rospy.get_param('~device', '0')
        self.device = select_device(self.deviceName)
        
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
        self.publisher_point_with_conf = rospy.Publisher('/yolo_pointcloud/point_with_confidence', PointConfidenceStamped, queue_size=10)

        self.publish_debug = rospy.Publisher('yolo_pointcloud_stereo_debug', YoloStereoDebug, queue_size=1)
        
        rospy.loginfo("Nodes Launched")

        #Create Collector
        self.collector = Collector(20)

        #Load model
        self.model = YOLO(self.weights)
        self.listener = tf.TransformListener()
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

        predictions = self.model.track(source=[self.cv_imageL,self.cv_imageR])

        predictionsL = predictions[0].cpu().numpy()
        predictionsR = predictions[1].cpu().numpy()

        bboxL = predictionsL.boxes.data
        bboxR = predictionsR.boxes.data

        annotator = Annotator(im_viz, line_width=self.thickness, font_size=self.font_scale)

        det_bounds = BoundingBoxes()
        det_bounds.header = imageL.header
        det_bounds.image_header = imageL.header

        if ((bboxL.shape[0]>0 and bboxL.shape[1]==7) and (bboxR.shape[0]>0 and bboxR.shape[1]==7)):

            if (len(bboxL)<len(bboxR)): bboxR = bboxR[:len(bboxL),:]
            else: bboxL = bboxL[:len(bboxR),:]

            matched = self.matches_gen(bboxL, bboxR)

            for i in range(len(matched)):
                pointL = (int(((matched[i, 2]-matched[i, 0])/2) + matched[i, 0]), int(((matched[i, 3]-matched[i, 1])/2) + matched[i, 1]))
                pointR = (int(((matched[i, 6]-matched[i, 4])/2) + matched[i, 4]), int(((matched[i, 7]-matched[i, 5])/2) + matched[i, 5]))
                point = self.dist_calc(pointL, pointR)
                
                transform_ok=False
                while not transform_ok and not rospy.is_shutdown():
                    try:
                        tr_matrix = self.listener.asMatrix('map', imageL.header)
                        xyz = tuple(np.dot(tr_matrix, np.array([point[0], point[1], point[2], 1.0])))[:3]
                        det = np.array([xyz[0], xyz[1], xyz[2], matched[i, 8], matched[i, 9], matched[i, 10]], dtype=np.float32)
                        self.collector.insertPoint(det, imageL.header.stamp)
                        transform_ok = True
                    except tf.ExtrapolationException as e:
                        rospy.logwarn("Exception on transforming pose... trying again \n(" + str(e) + ")")
                        rospy.sleep(0.1)
                        imageL.header.stamp = self.listener.getLatestCommonTime('stereo_cam', 'map')



                if ((np.isnan(point).any()==False) and (np.isinf(point).any()==False)):

                    pointConfidenceStamped = PointConfidenceStamped()
                    pointConfidenceStamped.header = imageL.header
                    pointConfidenceStamped.point.x = point[0]
                    pointConfidenceStamped.point.y = point[1]
                    pointConfidenceStamped.point.z = point[2]
                    pointConfidenceStamped.id = matched[i,8]
                    pointConfidenceStamped.confidence =  matched[i,9]
                    pointConfidenceStamped.class_name = self.model.names[int(matched[i,10])]
                    self.publisher_point_with_conf.publish(pointConfidenceStamped)

                if self.publish_img:
                    label = f"{self.model.names[int(matched[i,10])]} {matched[i,9]:.2f}"
                    annotator.box_label(box=bboxL[i,:], label=label)                
       
        # #Publish Messages
        # self.publisher_obj0.publish(det_bounds)


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
            matched_kp = np.zeros(shape=(0,11), dtype=np.float32)
            knnMatches = 2
            matches = self.matcher.knnMatch(descriptorsL, descriptorsR, k=knnMatches)
            tresholdDist = self.matching_dist_coef * math.sqrt((math.pow(imageLeftMono.shape[0],2) + math.pow(imageLeftMono.shape[1],2)))
            for i in range(len(matches)):
                for j in range(len(matches[0])):
                    if (boxL[(matches[i][j].queryIdx), 6]==boxR[(matches[i][j].trainIdx), 6]):
                        match0 = keypointsL[matches[i][j].queryIdx].pt
                        match1 = keypointsR[matches[i][j].trainIdx].pt
                        dist = math.sqrt((match0[0] - match1[0]) * (match0[0] - match1[0]) + (match0[1] - match1[1]) * (match0[1] - match1[1]))
                    if (dist < tresholdDist and abs(match0[1]-match1[1])<1):
                        matched_kp = np.vstack((matched_kp, 
                                                [boxL[(matches[i][j].queryIdx),0],#x min Left
                                                 boxL[(matches[i][j].queryIdx),1],#y min Left
                                                 boxL[(matches[i][j].queryIdx),2],#x max Left
                                                 boxL[(matches[i][j].queryIdx),3],#y max Left
                                                 boxR[(matches[i][j].trainIdx),0],#x min Right
                                                 boxR[(matches[i][j].trainIdx),1],#y min Right  
                                                 boxR[(matches[i][j].trainIdx),2],#x max Right
                                                 boxR[(matches[i][j].trainIdx),3],#y max Right
                                                 boxL[(matches[i][j].queryIdx), 4], #ID from left image
                                                 ((boxL[(matches[i][j].queryIdx),5]+boxL[(matches[i][j].trainIdx),5])/2), #avg confidence
                                                 boxL[(matches[i][j].queryIdx), 6]])) #class from left image
                        #matched_kp data is in the following format [xL,yL,xL,yL,xR,yR,xR,yR,ID,conf,cls]
        return matched_kp
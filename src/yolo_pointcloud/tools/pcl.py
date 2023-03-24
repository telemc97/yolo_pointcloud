#!/usr/bin/env python3

import rospy
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
import numpy as np
from sensor_msgs.msg import PointCloud2

class PointCloud_gen:
    def __init__(self, xyz, stamp_):
        self.xyz = xyz
        self.stamp_ = stamp_
        #Args
        self.PointCloud_topic = rospy.get_param('~PointCloud_Topic', 'PointCloud')
        self.PointCloud_Frame = rospy.get_param('~PointCloud_Frame', 'stereo_cam')
        #Define Publisher
        self.publisher_pointCloud = rospy.Publisher(self.PointCloud_topic, PointCloud2, queue_size=10)

    def pub_pcl(self):
        
        header = Header()
        header.frame_id = self.PointCloud_Frame
        header.stamp = self.stamp_
        x = self.xyz[0,:]
        y = self.xyz[1,:]
        z = self.xyz[2,:]
        points = np.array([x,y,z]).reshape(3,-1).T
        # rospy.loginfo(points.shape)
        self.cloud = pc2.create_cloud_xyz32(header, points)
        self.cloud.is_dense = False
        self.cloud.is_bigendian = False


        self.publisher_pointCloud.publish(self.cloud)        
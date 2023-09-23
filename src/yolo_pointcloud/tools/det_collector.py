import numpy as np
from geometry_msgs.msg import PointStamped
import rospy

#TODO: this module will recieve the points and will store them according to their ID ADD timestamp and confidence.
class Collector:
    def __init__(self, num):
        self.data={} #Key is tuple(id, cls)
        self.id = 0
        self.cls = 0
        self.max_conf = 0
        self.num = num # This is the number of consequtive points taken to output one.
        self.cur_idx = 0

        self.point_publisher = rospy.Publisher('/det_tracker/point', PointStamped, queue_size=10)

    def insertPoint(self, point, stamp):
        #point format: [x, y, z, ID, conf,cls]
        #              [0, 1, 2, 3,  4,   5  ]
        #point in Collector format: [x, y, z, conf, stamp]

        if tuple((point[3],point[5])) not in self.data:
            self.data[tuple((point[3],point[5]))]=np.zeros(shape=(self.num,3),dtype=np.float32)
            self.cls = point[5]
            self.id = point[3]
        
        if point[4]>self.max_conf:
            self.max_conf = point[4]
        
        for i in range(self.num):
            datum = self.data[tuple((point[3],point[5]))][i]
            if (datum[0]==0 and datum[1]==0 and datum[2]==0):
                datum[0] = point[0]
                datum[1] = point[1]
                datum[2] = point[2]
                if (i==(self.num-1)):
                    datum = self.data[tuple((point[3],point[5]))][0]
                    datum[0] = 0
                    datum[1] = 0
                    datum[2] = 0
                    self.calcPosition(self.data[tuple((point[3],point[5]))])
                else:
                    datum = self.data[tuple((point[3],point[5]))][i+1]
                    datum[0] = 0
                    datum[1] = 0
                    datum[2] = 0
                break
                





    def calcPosition(self, points, outlier_dist=2):
        #Output is the odometry ()
        median_x = np.median(points[:,0])
        median_y = np.median(points[:,1])
        median_z = np.median(points[:,2])
        median_stamp = np.median(points[:,4])

        new_pt = np.zeros(shape=3, dtype=np.float32)
        norm_conf_sum = 0
        for point in np.nditer(points):
            if abs(point[0]-median_x)<outlier_dist or abs(point[1]-median_y)<outlier_dist or abs(point[2]-median_z)<outlier_dist:
                # checks if point is an outlier
                norm_conf = point[3]/(self.max_conf)
                new_pt[0] = (norm_conf_sum*point[0])/(norm_conf_sum+norm_conf)
                new_pt[1] = (norm_conf_sum*point[1])/(norm_conf_sum+norm_conf)
                new_pt[2] = (norm_conf_sum*point[2])/(norm_conf_sum+norm_conf)
                norm_conf_sum += norm_conf
            
            point_stamped = PointStamped()
            point_stamped.point.x = new_pt[0]
            point_stamped.point.y = new_pt[1]
            point_stamped.point.z = new_pt[2]
            point_stamped.header.stamp = median_stamp
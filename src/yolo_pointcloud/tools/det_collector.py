import numpy as np
from geometry_msgs.msg import PointStamped
import rospy

class Collector:
    def __init__(self, num):
        #Key is tuple(id, cls)
        self.data={}
        self.id = 0
        self.cls = 0
        self.max_conf = 0
        # This is the number of consequtive points taken to output one.
        self.num = num
        self.cur_idx = 0

        self.point_publisher = rospy.Publisher('/det_tracker/point', PointStamped, queue_size=10)

    def insertPoint(self, point):
        #point format: [x, y, z, ID, conf, cls, stamp]
        #              [0, 1, 2, 3,  4,    5,   6  ]
        #point in Collector format: [x, y, z, conf, stamp]

        if tuple((point[3],point[5])) not in self.data:
            self.data[tuple((point[3],point[5]))]=np.zeros(shape=(self.num,5),dtype=np.float32)
            self.cls = point[5]
            self.id = point[3]
        
        for i in range(self.num):
            datum = self.data[tuple((point[3],point[5]))][i]
            if (datum[0]==0 and datum[1]==0 and datum[2]==0):
                datum[0] = point[0]
                datum[1] = point[1]
                datum[2] = point[2]
                datum[3] = point[4]
                datum[4] = point[6]
                if (i==(self.num-1)):
                    datum = self.data[tuple((point[3],point[5]))][0]
                    point_stamped = self.calcPosition(self.data[tuple((point[3],point[5]))], point[3])
                    self.point_publisher.publish(point_stamped)
                    datum[0] = 0
                    datum[1] = 0
                    datum[2] = 0
                    break
                else:
                    datum = self.data[tuple((point[3],point[5]))][i+1]
                    datum[0] = 0
                    datum[1] = 0
                    datum[2] = 0
                    break
                
    def calcPosition(self, points, id, outlier_dist=2) -> PointStamped:
        #Output is the odometry ()
        median_x = np.median(points[:,0])
        median_y = np.median(points[:,1])
        median_z = np.median(points[:,2])
        median_stamp = np.median(points[:,4])
        new_pt = np.zeros(shape=3, dtype=np.float32)
        avg_conf = 0
        for i in range(self.num):
            point = points[i]
            # checks if point is an outlier
            if abs(point[0]-median_x)<outlier_dist or abs(point[1]-median_y)<outlier_dist or abs(point[2]-median_z)<outlier_dist:
                new_pt[0] = (i*new_pt[0]+point[0])/(i+1)
                new_pt[1] = (i*new_pt[1]+point[1])/(i+1)
                new_pt[2] = (i*new_pt[2]+point[2])/(i+1)
                avg_conf = (i*new_pt[2]+point[3])/(i+1)
            
        point_stamped_ = DetectionStamped()
        point_stamped_.point.x = new_pt[0]
        point_stamped_.point.y = new_pt[1]
        point_stamped_.point.z = new_pt[2]
        point_stamped_.confidence = avg_conf
        point_stamped_.id = id
        point_stamped_.header.stamp = rospy.Time.from_sec(median_stamp)
        return point_stamped_
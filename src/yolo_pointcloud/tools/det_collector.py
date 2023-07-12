import numpy as np

#TODO: this module will recieve the points and will store them according to their ID.
class Collector:
    def __init__(self, num):
        self.data={} #Key is tuple(id, cls)
        self.id = 0
        self.cls = 0
        self.max_conf = 0
        self.num = num
        self.cur_idx = 0

    def insertPoint(self, point):
        #point format: [x, y, z, ID, conf,cls]
        #              [0, 1, 2, 3,  4,   5  ]
        #point in Collector format: [x, y, z, conf]

        if self.data[tuple(point[3],point[5])] is None:
            self.data[tuple(point[3],point[5])]=np.zeros(shape=(self.num,4),dtype=np.float32)
            self.cls = point[5]
            self.id = point[3]
        
        if point[4]>self.max_conf:
            self.max_conf = point[4]
        
        if self.cur_idx==self.num:
            self.cur_idx=0
            self.calcPosition(self, self.data[tuple(point[3],point[5])])
        else:
            self.data[tuple(point[3],point[5])][self.cur_idx,:3] = point[:3]
            self.data[tuple(point[3],point[5])][self.cur_idx,3] = point[4]
            self.cur_idx+=1



    def calcPosition(self, point, outlier_dist):
        median_x = np.median(point[0,])
        median_y = np.median(point[2,])
        median_z = np.median(point[3,])
        new_pt = np.zeros(shape=3, dtype=np.float32)
        norm_conf_sum = 0
        for x in np.nditer(point):
            if abs(point[0]-median_x)<outlier_dist or abs(point[1]-median_y)<outlier_dist or abs(point[2]-median_z)<outlier_dist:
                norm_conf = point[3]/(self.max_conf)
                new_pt[0] = (norm_conf_sum*point[0])/(norm_conf_sum+norm_conf)
                new_pt[1] = (norm_conf_sum*point[1])/(norm_conf_sum+norm_conf)
                new_pt[2] = (norm_conf_sum*point[2])/(norm_conf_sum+norm_conf)
                norm_conf_sum += norm_conf
                



    def rejectOutliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]
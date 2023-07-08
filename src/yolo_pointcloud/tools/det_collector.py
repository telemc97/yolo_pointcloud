import numpy as np

class Detection():

    def __init__(self):
        self.position = np.zeros(shape=(3), dtype=np.float32)

        self.time = 0.0
        self.id = 0
        self.cls = 0
        self.conf = 0

    def insertData(self, point, id, conf, cls):
        #point format: [x, y, z ,ID,conf,cls]
        #              [0, 1, 2, 3, 4,   5]
        self.left_det_box = point

        self.id = id
        self.conf = conf
        self.cls = cls

#TODO: this module will recieve the points and will store them according to their ID.
class Collector:
    def __init__(self, num):
        self.ids={"id":[]}
        self.num = num

    def insertPoint(self, point: Detection):
        #point format: [x, y, z, ID,conf,cls]
        #              [0, 1, 2, 3, 4,   5  ]

        self.ids[point.id].append(point)
        if len(self.ids[point.id])>20:
            self.ids[point.id].popleft()

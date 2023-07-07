import numpy as np
#TODO: this module will recieve the points and will store them according to their ID.
class Collector:
    def __init__(self, num):
        self.ids={}
        self.num = num

    def insertPoint(self, point):
        #point format: [xL,yL,xL,yL,xR,yR,xR,yR,ID,conf,cls]
        #              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,   10 ]

        self.ids[point[8]] ==

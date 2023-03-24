import numpy as np
import torch
import math
import cv2


def get_list_size (matrix, name):
    nrow = len(matrix)
    if len(matrix[0]) is not None:
        ncol = len(matrix[0])
    else:
        ncol = 0
    # rospy.loginfo('%s size %i X %i' %(name, nrow, ncol))

def get_2d_list_slice(matrix, start_row, end_row, start_col, end_col):
    return [row[start_col:end_col] for row in matrix[start_row:end_row]]

def homogenous_to_euclidian(dist):
    x = dist[0,0]/dist[3,0]
    y = dist[1,0]/dist[3,0]
    z = dist[2,0]/dist[3,0]
    points_3d = np.array([x,y,z], dtype=np.float16)
    return points_3d

def euler_from_quaternion(quaternion):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        
        quaternion[0] = x
        quaternion[1] = y
        quaternion[2] = z
        quaternion[3] = w

        """
        t0 = +2.0 * (quaternion[3] * quaternion[0] + quaternion[1] * quaternion[2])
        t1 = +1.0 - 2.0 * (quaternion[0] * quaternion[0] + quaternion[1] * quaternion[1])
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (quaternion[3] * quaternion[1] - quaternion[2] * quaternion[0])
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (quaternion[3] * quaternion[2] + quaternion[0] * quaternion[1])
        t4 = +1.0 - 2.0 * (quaternion[1] * quaternion[1] + quaternion[2] * quaternion[2])
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def get_single_det(box):
    newbox = np.zeros(shape=(0,6), dtype=object)
    i = 1
    xmin, ymin, xmax, ymax = 0
    for i in range(len(box)):
        if ( box[i-1,0] < box[i,0] ):
            xmin = box[i,0]
        if ( box[i-1,1] < box[i,1] ):
            ymin = box[i,1]
        if ( box[i-1,2] > box[i,2] ):
            xmax = box[i,2]
        if ( box[i-1,3] > box[i,3] ):
            ymax = box[i,3]
        newbox = np.vstack((xmin, ymin, xmax, ymax, box[i,4], box[i,5]))
    return newbox

def get_center(xmin, ymin, xmax, ymax):
    center_x = int( ((xmax-xmin)/2) + xmin )
    center_y = int( ((ymax-ymin)/2) + ymin ) 
    return center_x, center_y

## From https://github.com/telemc97/yolov7/blob/main/utils/general.py Forked from https://github.com/WongKinYiu/yolov7.git

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)
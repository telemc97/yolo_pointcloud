import rospy
from yolo_pointcloud.main_detector import mainDetector

if __name__ == "__main__":    
    rospy.init_node("yolo_pointcloud", anonymous=True)
    mainDetector()
    rospy.spin()
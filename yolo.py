from object_distance_detection.realsense_camera import *
import cv2
from ultralytics import YOLO
from object_distance_detection.object_detection import *



rs = RealsenseCamera()
object_detection = ObjectDetection()

while True:
    ret, color_frame, depth_frame = rs.get_frame_stream() 
    height, width, _ = color_frame.shape

    bboxes, class_ids, scores = object_detection.detect(color_frame, imgsz=640)

    
    
    # cv2.imshow("depth_frame", depth_frame)
    cv2.imshow("color frame",color_frame)

    key = cv2.waitKey(1)

    if key == 27:
        break


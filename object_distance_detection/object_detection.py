#http://www.pysource.com
import numpy as np
from ultralytics import YOLO
import random
import colorsys
import torch
import cv2
from realsense_camera import RealsenseCamera

# Set random seed
random.seed(2)


class ObjectDetection:
    def __init__(self, weights_path="runs/detect/train2/weights/best.pt"):
        # Load Network
        self.weights_path = weights_path

        self.colors = self.random_colors(800)

        # Load Yolo
        self.model = YOLO(self.weights_path)
        self.classes = self.model.names

        # Load Default device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device(0)
        else:
            self.device = torch.device("cpu")

    def get_id_by_class_name(self, class_name):
        for i, name in enumerate(self.classes.values()):
            if name.lower() == class_name.lower():
                return i
        return -1

    def random_colors(self, N, bright=False):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        brightness = 255 if bright else 180
        hsv = [(i / N + 1, 1, brightness) for i in range(N + 1)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        random.shuffle(colors)
        return colors

    def detect(self, frame, imgsz=1280, conf=0.25, nms=True, classes=None, device=None):
        # Filter classes
        filter_classes = classes if classes else None
        device = device if device else self.device
        # Detect objects
        results = self.model.predict(source=frame, save=False, save_txt=False,
                                     imgsz=imgsz,
                                     conf=conf,
                                     nms=nms,
                                     classes=filter_classes,
                                     half=False,
                                     device=device)  # save predictions as labels

        # Get the first result from the array as we are only using one image
        result = results[0]
        # Get bboxes
        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        # Get class ids
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        # Get scores
        # round score to 2 decimal places
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, scores
    def draw_object_info(self,realsense_camera,color_image,depth_image):
        # Get the object detection
        bboxes, class_ids, score = self.detect(color_image)
        for bbox, class_id, score in zip(bboxes, class_ids, score):
            x, y, x2, y2 = bbox
            color = self.colors[class_id]
            cv2.rectangle(color_image, (x, y), (x2, y2), color, 2)

            # display name
            class_name = self.classes[class_id]
            cv2.putText(color_image, f"{class_name}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Get center of the bbox
            cx, cy = (x + x2) // 2, (y + y2) // 2
            distance = realsense_camera.get_distance_point(depth_image, cx, cy)

            # Draw circle
            cv2.circle(color_image, (cx, cy), 5, color, -1)
            cv2.putText(color_image, f"Distance: {distance} cm", (cx, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            
    def Xac_dinh_vi_tri(self, color_image):
        #get height, width of screen
        height, width = color_image.shape[:2]
        #get center of the screen
        height, width = height//2, width//2

        color = (255,255,255)
        bboxes, class_ids, score = self.detect(color_image)

        for bbox, class_id, score in zip(bboxes, class_ids, score):
            x, y, x2, y2 = bbox
            cx, cy = (x + x2) // 2, (y + y2) // 2
            cv2.putText(color_image, f"({cx},{cy})" ,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            #duoi phai
            if(cx>width and cy>height):
                cv2.putText(color_image, "back right", (cx, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            #tren phai
            elif(cx>width and cy<height):
                cv2.putText(color_image, "front right", (cx, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            #duoi trai
            elif(cx<width and cy>height):
                cv2.putText(color_image, "back left", (cx, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            # tren trai
            else:
                cv2.putText(color_image, "front left", (cx, cy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
                
                


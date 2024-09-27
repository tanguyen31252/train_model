from roboflow import Roboflow
import cv2
# from ultralytics import YOLO

# Khởi tạo Roboflow
rf = Roboflow(api_key="O3hPlIXGJtFHDSHaNfMC")
project = rf.workspace("tanguyn").project("backboard-dl3lc")
model = project.version(2).model



# visualize your prediction
model.predict("pexels-samuel-reis-355265419-14262633.jpg", confidence=40, overlap=30).save("prediction.jpg")

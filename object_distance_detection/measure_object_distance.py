#http://www.pysource.com

from realsense_camera import RealsenseCamera
from object_detection import ObjectDetection
import cv2

# Create the Camera object
camera = RealsenseCamera()

# Create the Object Detection object
object_detection = ObjectDetection()


while True:
    # Get frame from realsense camera
    ret, color_image, depth_image = camera.get_frame_stream()
    height, width = color_image.shape[:2]
    
    center_x, center_y = width//2, height//2
    
    object_detection.draw_object_info(camera,color_image,depth_image)

    color_image = cv2.line(color_image, (0, center_y), (width, center_y), (0, 255, 0), 2)

    # Trục dọc: từ (center_x, 0) đến (center_x, height)
    color_image = cv2.line(color_image, (center_x, 0), (center_x, height), (0, 255, 0), 2)

    object_detection.Xac_dinh_vi_tri(color_image)

    # show color image
    cv2.imshow("Color Image", color_image)
    # cv2.imshow("depth Image", depth_image)
    key = cv2.waitKey(1)
    if key == 27:
        break

# release the camera
camera.release()
cv2.destroyAllWindows()
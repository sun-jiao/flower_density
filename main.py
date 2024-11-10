from ultralytics import YOLO
from PIL import Image

from flower_detection import yolo_detection

if __name__ == '__main__':
    yolo_detection("test_images")
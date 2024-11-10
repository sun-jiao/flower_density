from PIL import Image
from ultralytics import YOLO

model = YOLO("yolo11x.pt")

def yolo_detection(image):
    results = model(image)
    # it returns a list of results, if the given `image` is a folder, the returned list would contain multiple results,
    # while we keep the param as a single image so `results` contains only one result.
    result = results[0]
    result.save(filename=f"result-{image}")



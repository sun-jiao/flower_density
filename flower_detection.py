import os
import re

import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

model = YOLO("yolo11x.pt")

flower1_cls = 47  # actually it's apple's class
flower2_cls = 46  # banana

def yolo_detection(image):
    results = model(image)
    # it returns a list of results, if the given `image` is a folder, the returned list would contain multiple results
    # for every image in this folder, while we keep the param as a single image so `results` contains only one result.
    result = results[0]
    result.save(filename=f"results/{os.path.splitext(os.path.basename(image))[0]}-result.JPG")

    return len(list(filter(lambda r: r.boxes.cls.item() == flower1_cls, list(result))))



pattern = re.compile(r"^\d{4}-\d{2}-\d{2}\.JPG$")


def batch_detection(directory):
    data = []

    for filename in os.listdir(directory):
        if pattern.match(filename):
            file_path = os.path.join(directory, filename)
            result = yolo_detection(file_path)

            date_str = filename[:10]  # Get datetime string (2024-06-01) from filename
            data.append((date_str, result))

    df = pd.DataFrame(data, columns=["Date", "Density"])
    df["Date"] = pd.to_datetime(df["Date"])  # convert string to datetime
    df = df.sort_values("Date")

    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Density"])
    plt.xlabel("Date")
    plt.ylabel("Flower density")
    plt.title("Flower density Over Time")
    plt.grid(True)
    plt.savefig("results/results_chart-density.png")

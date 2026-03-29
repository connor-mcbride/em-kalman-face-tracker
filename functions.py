import numpy as np
import os
import matplotlib.pyplot as plt

def get_bounding_box(filename):
    with open(filename) as f:
        data = f.readlines()

    combined = ' '.join(data)
    start = combined.index("{") + 3
    end = combined.index("}")
    points = [coord.split(" ") for coord in combined[start:end].strip().split("\n ")]
    points = np.array(points).astype(float)

    max_coords = np.max(points, axis=0)
    min_coords = np.min(points, axis=0)
    x, y = min_coords
    w, h = max_coords - min_coords
    
    return x, y, w, h

def train_video_test(video_id):
    # Return x, y, w, h instead of print
    filename = f"data/{video_id}/annot"
    frames = sorted(os.listdir(filename))
    xywh_list = []
    for frame in frames:
        if frame.endswith(".pts"):
            file = filename + "/" + frame
            x, y, w, h = get_bounding_box(file)
            xywh_list.append([x, y, w, h])
    return np.array(xywh_list)
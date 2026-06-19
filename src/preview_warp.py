import json
import cv2
import numpy as np

H = np.load("mydata/metadata/homography.npy")

with open("mydata/metadata/warp_config.json", "r") as f:
    config = json.load(f)

lane_width = config["lane_width"]
lane_height = config["lane_height"]

frame = cv2.imread("mydata/raw/chassis1/chassis1_0123.png")

warped = cv2.warpPerspective(
    frame,
    H,
    (lane_width, lane_height)
)

cv2.imwrite("mydata/processed/warped_0123.jpg", warped)
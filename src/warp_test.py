import json
import cv2
import numpy as np

JSON_PATH = "mydata/metadata/project_export.json"


def fit_line(points):
    pts = np.asarray(points, dtype=np.float32)

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    return np.array([
        float(vx),
        float(vy),
        float(x0),
        float(y0)
    ])


def line_boundary_points(line, img_h):

    vx, vy, x0, y0 = line

    y_top = 0
    y_bot = img_h - 1

    x_top = x0 + (y_top - y0) * vx / vy
    x_bot = x0 + (y_bot - y0) * vx / vy

    return (
        (float(x_top), float(y_top)),
        (float(x_bot), float(y_bot))
    )


with open(JSON_PATH, "r", encoding="utf-8") as f:
    tasks = json.load(f)

left_lines = []
right_lines = []

img_w = None
img_h = None

for task in tasks:

    if not task["annotations"]:
        continue

    ann = task["annotations"][0]

    left_pts = []
    right_pts = []

    for r in ann["result"]:

        if r["type"] != "keypointlabels":
            continue

        img_w = r["original_width"]
        img_h = r["original_height"]

        x = r["value"]["x"] / 100.0 * img_w
        y = r["value"]["y"] / 100.0 * img_h

        label = r["value"]["keypointlabels"][0]

        if label == "left_edge":
            left_pts.append([x, y])

        elif label == "right_edge":
            right_pts.append([x, y])

    if len(left_pts) >= 2:
        left_lines.append(fit_line(left_pts))

    if len(right_pts) >= 2:
        right_lines.append(fit_line(right_pts))


if len(left_lines) == 0 or len(right_lines) == 0:
    raise RuntimeError("No valid annotations found")


left_line = np.mean(left_lines, axis=0)
right_line = np.mean(right_lines, axis=0)

lt, lb = line_boundary_points(left_line, img_h)
rt, rb = line_boundary_points(right_line, img_h)

src_pts = np.float32([
    lt,
    rt,
    rb,
    lb
])

lane_width = int(
    (
        np.linalg.norm(np.array(rt) - np.array(lt))
        +
        np.linalg.norm(np.array(rb) - np.array(lb))
    ) / 2
)

lane_height = img_h

dst_pts = np.float32([
    [0, 0],
    [lane_width, 0],
    [lane_width, lane_height],
    [0, lane_height]
])

H = cv2.getPerspectiveTransform(
    src_pts,
    dst_pts
)

import os
os.makedirs("mydata/metadata", exist_ok=True)
np.save("mydata/metadata/homography.npy", H)

config = {
    "lane_width": lane_width,
    "lane_height": lane_height
}
with open("mydata/metadata/warp_config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

print("Saved mydata/metadata/homography.npy")
print("Saved mydata/metadata/warp_config.json")
print("Lane width:", lane_width)
print("Lane height:", lane_height)
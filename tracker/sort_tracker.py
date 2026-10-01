from ultralytics import YOLO
from sort.sort import Sort

import cv2
import csv
import os
import time
import numpy as np


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = "models/yolo11m.pt"
VIDEO_PATH = "video/PNNL_Parking_LOT(1).avi"

OUTPUT_DIR = "results/sort"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "sort_tracking.mp4"
)

TRACKING_CSV = os.path.join(
    OUTPUT_DIR,
    "tracking_results.csv"
)

METRICS_CSV = os.path.join(
    OUTPUT_DIR,
    "metrics.csv"
)


# Create output directory
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# --------------------------------------------------
# LOAD YOLO MODEL
# --------------------------------------------------

print("Loading YOLO11m model...")

model = YOLO(MODEL_PATH)


# --------------------------------------------------
# CREATE SORT TRACKER
# --------------------------------------------------

tracker = Sort(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)


# --------------------------------------------------
# OPEN VIDEO
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        "Could not open video."
    )


fps_video = cap.get(
    cv2.CAP_PROP_FPS
)

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_video_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)


print()
print("YOLO11m + SORT TRACKING")
print("=" * 50)

print(
    f"Video frames: {total_video_frames}"
)

print(
    f"Video FPS: {fps_video:.2f}"
)


# --------------------------------------------------
# CREATE OUTPUT VIDEO
# --------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps_video,
    (
        frame_width,
        frame_height
    )
)


# --------------------------------------------------
# VARIABLES
# --------------------------------------------------

frame_number = 0

total_inference_time = 0.0

total_processing_time = 0.0

unique_track_ids = set()

tracking_rows = []


# --------------------------------------------------
# PROCESS VIDEO
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    start_time = time.perf_counter()


    # --------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------

    inference_start = time.perf_counter()

    results = model.predict(
        frame,
        conf=0.25,
        verbose=False
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    total_inference_time += (
        inference_time
    )


    # Get first result
    result = results[0]


    # --------------------------------------------------
    # GET DETECTIONS
    # --------------------------------------------------

    detections = []


    if result.boxes is not None:

        boxes = (
            result.boxes.xyxy
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
        )


        for box, confidence, class_id in zip(
            boxes,
            confidences,
            classes
        ):

            x1, y1, x2, y2 = box

            detections.append(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence,
                    class_id
                ]
            )


    # --------------------------------------------------
    # CONVERT TO NUMPY ARRAY
    # --------------------------------------------------

    if detections:

        detections = np.array(
            detections,
            dtype=float
        )

    else:

        detections = np.empty(
            (0, 6),
            dtype=float
        )


    # --------------------------------------------------
    # SORT TRACKING
    # --------------------------------------------------

    tracked_objects = tracker.update(
        detections
    )


    # --------------------------------------------------
    # DRAW TRACKING RESULTS
    # --------------------------------------------------

    for tracked in tracked_objects:

        x1, y1, x2, y2, track_id, class_id = (
            tracked
        )


        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        track_id = int(track_id)
        class_id = int(class_id)


        # Get class name
        class_name = model.names[
            class_id
        ]


        # Store unique track ID
        unique_track_ids.add(
            track_id
        )


        # --------------------------------------------------
        # DRAW BOUNDING BOX
        # --------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # --------------------------------------------------
        # DRAW TRACK ID
        # --------------------------------------------------

        label = (
            f"{class_name} "
            f"ID:{track_id}"
        )


        cv2.putText(
            frame,
            label,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


        # --------------------------------------------------
        # SAVE TRACKING DATA
        # --------------------------------------------------

        tracking_rows.append(
            [
                frame_number,
                track_id,
                class_name,
                x1,
                y1,
                x2,
                y2
            ]
        )


    # --------------------------------------------------
    # SAVE OUTPUT FRAME
    # --------------------------------------------------

    writer.write(frame)


    # --------------------------------------------------
    # PROCESSING TIME
    # --------------------------------------------------

    processing_time = (
        time.perf_counter()
        - start_time
    )

    total_processing_time += (
        processing_time
    )


    # --------------------------------------------------
    # SHOW PROGRESS
    # --------------------------------------------------

    if frame_number % 50 == 0:

        print(
            f"Processed "
            f"{frame_number}/"
            f"{total_video_frames} frames"
        )


# --------------------------------------------------
# RELEASE VIDEO
# --------------------------------------------------

cap.release()

writer.release()


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

if frame_number > 0:

    average_inference_time_ms = (
        total_inference_time
        / frame_number
        * 1000
    )

    average_processing_time_ms = (
        total_processing_time
        / frame_number
        * 1000
    )

else:

    average_inference_time_ms = 0

    average_processing_time_ms = 0


if total_processing_time > 0:

    tracking_fps = (
        frame_number
        / total_processing_time
    )

else:

    tracking_fps = 0


# --------------------------------------------------
# SAVE TRACKING CSV
# --------------------------------------------------

with open(
    TRACKING_CSV,
    "w",
    newline=""
) as file:

    csv_writer = csv.writer(file)

    csv_writer.writerow(
        [
            "frame",
            "track_id",
            "class",
            "x1",
            "y1",
            "x2",
            "y2"
        ]
    )

    csv_writer.writerows(
        tracking_rows
    )


# --------------------------------------------------
# SAVE METRICS CSV
# --------------------------------------------------

metrics = [

    ["metric", "value"],

    [
        "model",
        "YOLO11m + SORT"
    ],

    [
        "total_frames",
        frame_number
    ],

    [
        "unique_track_ids",
        len(unique_track_ids)
    ],

    [
        "total_inference_time_seconds",
        round(
            total_inference_time,
            4
        )
    ],

    [
        "average_inference_time_ms",
        round(
            average_inference_time_ms,
            4
        )
    ],

    [
        "average_processing_time_ms",
        round(
            average_processing_time_ms,
            4
        )
    ],

    [
        "tracking_fps",
        round(
            tracking_fps,
            4
        )
    ]
]


with open(
    METRICS_CSV,
    "w",
    newline=""
) as file:

    csv_writer = csv.writer(file)

    csv_writer.writerows(
        metrics
    )


# --------------------------------------------------
# FINAL OUTPUT
# --------------------------------------------------

print()
print("=" * 50)
print("SORT TRACKING COMPLETED")
print("=" * 50)

print(
    f"Total frames          : "
    f"{frame_number}"
)

print(
    f"Unique track IDs      : "
    f"{len(unique_track_ids)}"
)

print(
    f"Avg inference time    : "
    f"{average_inference_time_ms:.2f} ms"
)

print(
    f"Avg processing time   : "
    f"{average_processing_time_ms:.2f} ms"
)

print(
    f"Tracking FPS          : "
    f"{tracking_fps:.2f}"
)

print()
print("Results saved to:")

print(
    OUTPUT_VIDEO
)

print(
    TRACKING_CSV
)

print(
    METRICS_CSV
)
from ultralytics import YOLO
from sort.sort import Sort

import cv2
import csv
import json
import logging
import os
import time
import numpy as np
import psutil


# ============================================================
# PATHS
# ============================================================

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

METRICS_JSON = os.path.join(
    OUTPUT_DIR,
    "metrics.json"
)

LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "tracking.log"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("SORT tracking started")


# ============================================================
# SYSTEM MONITORING
# ============================================================

process = psutil.Process(os.getpid())

cpu_samples = []
ram_samples_mb = []


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO11m model...")

model = YOLO(MODEL_PATH)

logging.info("YOLO11m model loaded")


# ============================================================
# CREATE SORT TRACKER
# ============================================================

tracker = Sort(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

logging.info("SORT tracker created")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    logging.error("Could not open video.")

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
print("=" * 60)

print(
    f"Video frames : {total_video_frames}"
)

print(
    f"Video FPS    : {fps_video:.2f}"
)

print(
    "GPU          : Not Available"
)

print()


# ============================================================
# OUTPUT VIDEO
# ============================================================

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


# ============================================================
# VARIABLES
# ============================================================

frame_number = 0

total_detections = 0

total_processing_time = 0.0

total_inference_time = 0.0

unique_track_ids = set()

tracking_rows = []

latencies_ms = []

active_track_counts = []

track_start_frames = {}

track_end_frames = {}

potential_track_losses = 0

previous_active_ids = set()


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    frame_start = time.perf_counter()


    # --------------------------------------------------------
    # YOLO INFERENCE
    # --------------------------------------------------------

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


    result = results[0]


    # --------------------------------------------------------
    # GET DETECTIONS
    # --------------------------------------------------------

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

        total_detections += len(boxes)


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


    # --------------------------------------------------------
    # CONVERT DETECTIONS TO NUMPY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # SORT TRACKING
    # --------------------------------------------------------

    tracked_objects = tracker.update(
        detections
    )


    # --------------------------------------------------------
    # ACTIVE TRACKS
    # --------------------------------------------------------

    current_active_ids = {
        int(obj[4])
        for obj in tracked_objects
    }

    active_track_counts.append(
        len(current_active_ids)
    )


    # --------------------------------------------------------
    # TRACK LIFETIME
    # --------------------------------------------------------

    for track_id in current_active_ids:

        unique_track_ids.add(
            track_id
        )

        if track_id not in track_start_frames:

            track_start_frames[
                track_id
            ] = frame_number

        track_end_frames[
            track_id
        ] = frame_number


    # --------------------------------------------------------
    # TRACK LOSS STATISTIC
    # --------------------------------------------------------

    disappeared_ids = (
        previous_active_ids
        - current_active_ids
    )

    for track_id in disappeared_ids:

        if track_id in unique_track_ids:

            potential_track_losses += 1


    previous_active_ids = (
        current_active_ids
    )


    # --------------------------------------------------------
    # DRAW TRACKING RESULTS
    # --------------------------------------------------------

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

        class_name = model.names[
            class_id
        ]


        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # Track label
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


        # Save tracking information
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


    # --------------------------------------------------------
    # SAVE OUTPUT FRAME
    # --------------------------------------------------------

    writer.write(frame)


    # --------------------------------------------------------
    # FRAME PROCESSING TIME / LATENCY
    # --------------------------------------------------------

    processing_time = (
        time.perf_counter()
        - frame_start
    )

    total_processing_time += (
        processing_time
    )

    latency_ms = (
        processing_time * 1000
    )

    latencies_ms.append(
        latency_ms
    )


    # --------------------------------------------------------
    # CPU / RAM
    # --------------------------------------------------------

    cpu_usage = psutil.cpu_percent(
        interval=None
    )

    memory_info = process.memory_info()

    ram_mb = (
        memory_info.rss
        / (1024 * 1024)
    )

    cpu_samples.append(
        cpu_usage
    )

    ram_samples_mb.append(
        ram_mb
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 50 == 0:

        print(
            f"Processed "
            f"{frame_number}/"
            f"{total_video_frames} frames"
        )

        logging.info(
            f"Processed "
            f"{frame_number}/"
            f"{total_video_frames} frames"
        )


# ============================================================
# RELEASE RESOURCES
# ============================================================

cap.release()

writer.release()


# ============================================================
# CALCULATE PERFORMANCE METRICS
# ============================================================

if frame_number > 0:

    average_fps = (
        frame_number
        / total_processing_time
    )

    average_latency_ms = (
        sum(latencies_ms)
        / len(latencies_ms)
    )

else:

    average_fps = 0

    average_latency_ms = 0


# P95 latency
if latencies_ms:

    p95_latency_ms = float(
        np.percentile(
            latencies_ms,
            95
        )
    )

else:

    p95_latency_ms = 0


# ============================================================
# ACTIVE TRACK METRICS
# ============================================================

if active_track_counts:

    average_active_tracks = (
        sum(active_track_counts)
        / len(active_track_counts)
    )

    maximum_active_tracks = max(
        active_track_counts
    )

else:

    average_active_tracks = 0

    maximum_active_tracks = 0


# ============================================================
# TRACK LIFETIME
# ============================================================

track_lifetimes = []

for track_id in unique_track_ids:

    start_frame = track_start_frames.get(
        track_id
    )

    end_frame = track_end_frames.get(
        track_id
    )

    if (
        start_frame is not None
        and end_frame is not None
    ):

        lifetime = (
            end_frame
            - start_frame
            + 1
        )

        track_lifetimes.append(
            lifetime
        )


if track_lifetimes:

    average_track_lifetime = (
        sum(track_lifetimes)
        / len(track_lifetimes)
    )

    longest_track_lifetime = max(
        track_lifetimes
    )

else:

    average_track_lifetime = 0

    longest_track_lifetime = 0


# ============================================================
# CPU METRICS
# ============================================================

if cpu_samples:

    average_cpu = (
        sum(cpu_samples)
        / len(cpu_samples)
    )

    peak_cpu = max(
        cpu_samples
    )

else:

    average_cpu = 0

    peak_cpu = 0


# ============================================================
# RAM METRICS
# ============================================================

if ram_samples_mb:

    average_ram_mb = (
        sum(ram_samples_mb)
        / len(ram_samples_mb)
    )

    peak_ram_mb = max(
        ram_samples_mb
    )

else:

    average_ram_mb = 0

    peak_ram_mb = 0


# ============================================================
# OUTPUT VIDEO SIZE
# ============================================================

if os.path.exists(
    OUTPUT_VIDEO
):

    output_video_size_mb = (
        os.path.getsize(
            OUTPUT_VIDEO
        )
        / (1024 * 1024)
    )

else:

    output_video_size_mb = 0


# ============================================================
# GPU INFORMATION
# ============================================================

gpu_available = False

gpu_name = "Not Available"

peak_gpu_memory_mb = None

average_gpu_percent = None

peak_gpu_percent = None


try:

    import torch

    gpu_available = (
        torch.cuda.is_available()
    )

    if gpu_available:

        gpu_name = (
            torch.cuda.get_device_name(0)
        )

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(0)
            / (1024 * 1024)
        )

except Exception:

    gpu_available = False


# ============================================================
# ID SWITCHES
# ============================================================

id_switches = None

id_switches_note = (
    "Not calculated because ground-truth "
    "identity annotations are not available."
)


# ============================================================
# METRICS
# ============================================================

metrics = {

    "model":
        "YOLO11m + SORT",

    "video":
        VIDEO_PATH,

    "total_processing_time_seconds":
        round(
            total_processing_time,
            4
        ),

    "average_fps":
        round(
            average_fps,
            4
        ),

    "average_latency_ms":
        round(
            average_latency_ms,
            4
        ),

    "p95_latency_ms":
        round(
            p95_latency_ms,
            4
        ),

    "total_frames":
        frame_number,

    "total_detections":
        total_detections,

    "unique_track_ids":
        len(unique_track_ids),

    "average_active_tracks":
        round(
            average_active_tracks,
            4
        ),

    "maximum_active_tracks":
        maximum_active_tracks,

    "average_track_lifetime_frames":
        round(
            average_track_lifetime,
            4
        ),

    "longest_track_lifetime_frames":
        longest_track_lifetime,

    "id_switches":
        id_switches,

    "id_switches_note":
        id_switches_note,

    "potential_track_losses":
        potential_track_losses,

    "average_cpu_percent":
        round(
            average_cpu,
            4
        ),

    "peak_cpu_percent":
        round(
            peak_cpu,
            4
        ),

    "average_ram_mb":
        round(
            average_ram_mb,
            4
        ),

    "peak_ram_mb":
        round(
            peak_ram_mb,
            4
        ),

    "gpu_available":
        gpu_available,

    "gpu_name":
        gpu_name,

    "average_gpu_percent":
        average_gpu_percent,

    "peak_gpu_percent":
        peak_gpu_percent,

    "peak_gpu_memory_mb":
        peak_gpu_memory_mb,

    "output_video_size_mb":
        round(
            output_video_size_mb,
            4
        ),

    "output_video":
        OUTPUT_VIDEO,

    "tracking_csv":
        TRACKING_CSV,

    "log_file":
        LOG_FILE
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    METRICS_JSON,
    "w"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    METRICS_CSV,
    "w",
    newline=""
) as file:

    csv_writer = csv.writer(
        file
    )

    csv_writer.writerow(
        [
            "metric",
            "value"
        ]
    )

    for key, value in metrics.items():

        csv_writer.writerow(
            [
                key,
                value
            ]
        )


# ============================================================
# FINAL LOG
# ============================================================

logging.info(
    "SORT tracking completed"
)

logging.info(
    f"Total frames: {frame_number}"
)

logging.info(
    f"Total detections: {total_detections}"
)

logging.info(
    f"Unique track IDs: "
    f"{len(unique_track_ids)}"
)

logging.info(
    f"Average FPS: "
    f"{average_fps:.4f}"
)

logging.info(
    f"Average latency: "
    f"{average_latency_ms:.4f} ms"
)

logging.info(
    f"P95 latency: "
    f"{p95_latency_ms:.4f} ms"
)


# ============================================================
# FINAL TERMINAL OUTPUT
# ============================================================

print()
print("=" * 60)
print("SORT TRACKING COMPLETED")
print("=" * 60)

print(
    f"Total processing time : "
    f"{total_processing_time:.2f} sec"
)

print(
    f"Average FPS           : "
    f"{average_fps:.2f}"
)

print(
    f"Average latency       : "
    f"{average_latency_ms:.2f} ms"
)

print(
    f"P95 latency           : "
    f"{p95_latency_ms:.2f} ms"
)

print(
    f"Total frames          : "
    f"{frame_number}"
)

print(
    f"Total detections      : "
    f"{total_detections}"
)

print(
    f"Unique track IDs      : "
    f"{len(unique_track_ids)}"
)

print(
    f"Average active tracks : "
    f"{average_active_tracks:.2f}"
)

print(
    f"Maximum active tracks : "
    f"{maximum_active_tracks}"
)

print(
    f"Average track lifetime: "
    f"{average_track_lifetime:.2f} frames"
)

print(
    f"Longest track lifetime: "
    f"{longest_track_lifetime} frames"
)

print(
    f"Potential track losses: "
    f"{potential_track_losses}"
)

print(
    f"Average CPU           : "
    f"{average_cpu:.2f}%"
)

print(
    f"Peak CPU              : "
    f"{peak_cpu:.2f}%"
)

print(
    f"Average RAM           : "
    f"{average_ram_mb:.2f} MB"
)

print(
    f"Peak RAM              : "
    f"{peak_ram_mb:.2f} MB"
)

print(
    f"GPU                   : "
    f"{gpu_name}"
)

print(
    f"Peak GPU memory       : "
    f"{peak_gpu_memory_mb}"
)

print(
    f"Output video size     : "
    f"{output_video_size_mb:.2f} MB"
)

print()
print("Files created:")

print(
    f"Video  : {OUTPUT_VIDEO}"
)

print(
    f"CSV    : {METRICS_CSV}"
)

print(
    f"JSON   : {METRICS_JSON}"
)

print(
    f"Log    : {LOG_FILE}"
)

print(
    f"Tracks : {TRACKING_CSV}"
)

print("=" * 60)
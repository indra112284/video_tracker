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


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE_THRESHOLD = 0.25

SORT_MAX_AGE = 30
SORT_MIN_HITS = 3
SORT_IOU_THRESHOLD = 0.3


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

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

logging.info("YOLO11m + SORT tracking started")


# ============================================================
# SYSTEM MONITORING
# ============================================================

process = psutil.Process(os.getpid())

process_cpu_samples = []
system_cpu_samples = []
ram_samples_mb = []


# ============================================================
# PIPELINE START
# ============================================================

pipeline_start_time = time.perf_counter()


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("Loading YOLO11m model...")

model = YOLO(MODEL_PATH)

logging.info(
    f"YOLO model loaded: {MODEL_PATH}"
)


# ============================================================
# DEVICE
# ============================================================

try:

    import torch

    if torch.cuda.is_available():

        device = "CUDA"

        gpu_available = True

        gpu_name = torch.cuda.get_device_name(0)

    else:

        device = "CPU"

        gpu_available = False

        gpu_name = "Not Available"

except Exception:

    device = "CPU"

    gpu_available = False

    gpu_name = "Not Available"


# ============================================================
# CREATE SORT TRACKER
# ============================================================

tracker = Sort(
    max_age=SORT_MAX_AGE,
    min_hits=SORT_MIN_HITS,
    iou_threshold=SORT_IOU_THRESHOLD
)

logging.info(
    "SORT tracker created"
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    logging.error(
        "Could not open input video."
    )

    raise RuntimeError(
        "Could not open input video."
    )


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps_video = cap.get(
    cv2.CAP_PROP_FPS
)

frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_video_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print()
print("=" * 60)
print("YOLO11m + SORT OBJECT TRACKING")
print("=" * 60)

print(
    f"Model        : YOLO11m"
)

print(
    f"Tracker      : SORT"
)

print(
    f"Device       : {device}"
)

print(
    f"Video frames : {total_video_frames}"
)

print(
    f"Video FPS    : {fps_video:.2f}"
)

print(
    f"Resolution   : "
    f"{frame_width}x{frame_height}"
)

print(
    f"GPU          : {gpu_name}"
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

if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create output video."
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


    # ========================================================
    # YOLO INFERENCE
    # ========================================================

    inference_start = time.perf_counter()

    results = model.predict(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False,
        device=device
    )

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    total_inference_time += (
        inference_time
    )

    result = results[0]


    # ========================================================
    # GET DETECTIONS
    # ========================================================

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

        total_detections += len(
            boxes
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


    # ========================================================
    # CONVERT DETECTIONS TO NUMPY
    # ========================================================

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


    # ========================================================
    # SORT TRACKING
    # ========================================================

    tracked_objects = tracker.update(
        detections
    )


    # ========================================================
    # ACTIVE TRACKS
    # ========================================================

    current_active_ids = {
        int(obj[4])
        for obj in tracked_objects
    }

    active_track_counts.append(
        len(current_active_ids)
    )


    # ========================================================
    # TRACK LIFETIME
    # ========================================================

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


    # ========================================================
    # TRACK LOSS STATISTIC
    # ========================================================

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


    # ========================================================
    # DRAW TRACKING RESULTS
    # ========================================================

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


        # ----------------------------------------------------
        # DRAW BOUNDING BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ----------------------------------------------------
        # TRACK LABEL
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SAVE TRACKING INFORMATION
        # ----------------------------------------------------

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


    # ========================================================
    # SAVE OUTPUT FRAME
    # ========================================================

    writer.write(frame)


    # ========================================================
    # FRAME PROCESSING TIME
    # ========================================================

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


    # ========================================================
    # PROCESS CPU
    # ========================================================

    process_cpu_usage = (
        process.cpu_percent(
            interval=None
        )
    )


    # ========================================================
    # SYSTEM CPU
    # ========================================================

    system_cpu_usage = (
        psutil.cpu_percent(
            interval=None
        )
    )


    # ========================================================
    # RAM
    # ========================================================

    memory_info = process.memory_info()

    ram_mb = (
        memory_info.rss
        / (1024 * 1024)
    )


    process_cpu_samples.append(
        process_cpu_usage
    )

    system_cpu_samples.append(
        system_cpu_usage
    )

    ram_samples_mb.append(
        ram_mb
    )


    # ========================================================
    # PROGRESS
    # ========================================================

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
# PIPELINE WALL TIME
# ============================================================

pipeline_wall_time_seconds = (
    time.perf_counter()
    - pipeline_start_time
)


# ============================================================
# PERFORMANCE METRICS
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

    average_fps = 0.0

    average_latency_ms = 0.0


# ============================================================
# P95 LATENCY
# ============================================================

if latencies_ms:

    p95_latency_ms = float(
        np.percentile(
            latencies_ms,
            95
        )
    )

else:

    p95_latency_ms = 0.0


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

    average_active_tracks = 0.0

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

    average_track_lifetime = 0.0

    longest_track_lifetime = 0


# ============================================================
# PROCESS CPU METRICS
# ============================================================

if process_cpu_samples:

    average_process_cpu = (
        sum(process_cpu_samples)
        / len(process_cpu_samples)
    )

    peak_process_cpu = max(
        process_cpu_samples
    )

else:

    average_process_cpu = 0.0

    peak_process_cpu = 0.0


# ============================================================
# SYSTEM CPU METRICS
# ============================================================

if system_cpu_samples:

    average_system_cpu = (
        sum(system_cpu_samples)
        / len(system_cpu_samples)
    )

    peak_system_cpu = max(
        system_cpu_samples
    )

else:

    average_system_cpu = 0.0

    peak_system_cpu = 0.0


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

    average_ram_mb = 0.0

    peak_ram_mb = 0.0


# ============================================================
# GPU MEMORY
# ============================================================

peak_gpu_memory_mb = None

average_gpu_percent = None
peak_gpu_percent = None

if gpu_available:

    try:

        import torch

        peak_gpu_memory_mb = (
            torch.cuda.max_memory_allocated(0)
            / (1024 * 1024)
        )

    except Exception:

        peak_gpu_memory_mb = None


# ============================================================
# FINAL JSON
# ============================================================

metrics = {

    "project": {

        "name":
            "YOLO11m + SORT Object Tracking",

        "tracker":
            "SORT"
    },


    "model": {

        "name":
            "YOLO11m",

        "format":
            "pt",

        "path":
            os.path.abspath(
                MODEL_PATH
            )
    },


    "configuration": {

        "confidence_threshold":
            CONFIDENCE_THRESHOLD,

        "iou_threshold":
            SORT_IOU_THRESHOLD,

        "tracker_config":
            "SORT",

        "device":
            device
    },


    "video": {

        "input":
            os.path.abspath(
                VIDEO_PATH
            ),

        "width":
            frame_width,

        "height":
            frame_height,

        "fps":
            round(
                float(fps_video),
                4
            ),

        "total_frames":
            frame_number
    },


    "tracking_statistics": {

        "frame_count":
            frame_number,

        "total_detections":
            total_detections,

        "unique_track_ids":
            len(unique_track_ids),

        "average_active_tracks":
            round(
                average_active_tracks,
                10
            ),

        "maximum_active_tracks":
            maximum_active_tracks,

        "average_track_lifetime":
            round(
                average_track_lifetime,
                10
            ),

        "longest_track_lifetime":
            longest_track_lifetime,

        "average_fps":
            round(
                average_fps,
                10
            ),

        "average_latency_ms":
            round(
                average_latency_ms,
                10
            ),

        "p95_latency_ms":
            round(
                p95_latency_ms,
                10
            ),

        "total_processing_time_seconds":
            round(
                total_processing_time,
                10
            )
    },


    "resources": {

        "process": {

            "average_cpu_percent":
                round(
                    average_process_cpu,
                    10
                ),

            "peak_cpu_percent":
                round(
                    peak_process_cpu,
                    10
                ),

            "average_ram_mb":
                round(
                    average_ram_mb,
                    10
                ),

            "peak_ram_mb":
                round(
                    peak_ram_mb,
                    10
                )
        },


        "system": {

            "average_cpu_percent":
                round(
                    average_system_cpu,
                    10
                ),

            "peak_cpu_percent":
                round(
                    peak_system_cpu,
                    10
                )
        }
    },


    "outputs": {

        "video":
            os.path.abspath(
                OUTPUT_VIDEO
            ),

        "json":
            os.path.abspath(
                METRICS_JSON
            ),

        "log":
            os.path.abspath(
                LOG_FILE
            )
    }
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    METRICS_JSON,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# SAVE TRACKING CSV
# ============================================================

with open(
    TRACKING_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    csv_writer = csv.writer(
        file
    )

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


# ============================================================
# SAVE METRICS CSV
# ============================================================

with open(
    METRICS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    csv_writer = csv.writer(
        file
    )

    csv_writer.writerow(
        [
            "category",
            "metric",
            "value"
        ]
    )


    # --------------------------------------------------------
    # PROJECT
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "project",
            "name",
            "YOLO11m + SORT Object Tracking"
        ]
    )

    csv_writer.writerow(
        [
            "project",
            "tracker",
            "SORT"
        ]
    )


    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "model",
            "name",
            "YOLO11m"
        ]
    )

    csv_writer.writerow(
        [
            "model",
            "format",
            "pt"
        ]
    )

    csv_writer.writerow(
        [
            "model",
            "path",
            os.path.abspath(
                MODEL_PATH
            )
        ]
    )


    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "configuration",
            "confidence_threshold",
            CONFIDENCE_THRESHOLD
        ]
    )

    csv_writer.writerow(
        [
            "configuration",
            "iou_threshold",
            SORT_IOU_THRESHOLD
        ]
    )

    csv_writer.writerow(
        [
            "configuration",
            "tracker_config",
            "SORT"
        ]
    )

    csv_writer.writerow(
        [
            "configuration",
            "device",
            device
        ]
    )


    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "video",
            "input",
            os.path.abspath(
                VIDEO_PATH
            )
        ]
    )

    csv_writer.writerow(
        [
            "video",
            "width",
            frame_width
        ]
    )

    csv_writer.writerow(
        [
            "video",
            "height",
            frame_height
        ]
    )

    csv_writer.writerow(
        [
            "video",
            "fps",
            fps_video
        ]
    )

    csv_writer.writerow(
        [
            "video",
            "total_frames",
            frame_number
        ]
    )


    # --------------------------------------------------------
    # TRACKING STATISTICS
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "tracking_statistics",
            "frame_count",
            frame_number
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "total_detections",
            total_detections
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "unique_track_ids",
            len(unique_track_ids)
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "average_active_tracks",
            average_active_tracks
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "maximum_active_tracks",
            maximum_active_tracks
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "average_track_lifetime",
            average_track_lifetime
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "longest_track_lifetime",
            longest_track_lifetime
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "average_fps",
            average_fps
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "average_latency_ms",
            average_latency_ms
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "p95_latency_ms",
            p95_latency_ms
        ]
    )

    csv_writer.writerow(
        [
            "tracking_statistics",
            "total_processing_time_seconds",
            total_processing_time
        ]
    )


    # --------------------------------------------------------
    # PROCESS RESOURCES
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "resources.process",
            "average_cpu_percent",
            average_process_cpu
        ]
    )

    csv_writer.writerow(
        [
            "resources.process",
            "peak_cpu_percent",
            peak_process_cpu
        ]
    )

    csv_writer.writerow(
        [
            "resources.process",
            "average_ram_mb",
            average_ram_mb
        ]
    )

    csv_writer.writerow(
        [
            "resources.process",
            "peak_ram_mb",
            peak_ram_mb
        ]
    )


    # --------------------------------------------------------
    # SYSTEM RESOURCES
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "resources.system",
            "average_cpu_percent",
            average_system_cpu
        ]
    )

    csv_writer.writerow(
        [
            "resources.system",
            "peak_cpu_percent",
            peak_system_cpu
        ]
    )


    # --------------------------------------------------------
    # OUTPUTS
    # --------------------------------------------------------

    csv_writer.writerow(
        [
            "outputs",
            "video",
            os.path.abspath(
                OUTPUT_VIDEO
            )
        ]
    )

    csv_writer.writerow(
        [
            "outputs",
            "json",
            os.path.abspath(
                METRICS_JSON
            )
        ]
    )

    csv_writer.writerow(
        [
            "outputs",
            "log",
            os.path.abspath(
                LOG_FILE
            )
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
    f"Unique track IDs: {len(unique_track_ids)}"
)

logging.info(
    f"Average FPS: {average_fps:.4f}"
)

logging.info(
    f"Average latency: {average_latency_ms:.4f} ms"
)

logging.info(
    f"P95 latency: {p95_latency_ms:.4f} ms"
)

logging.info(
    f"Total processing time: "
    f"{total_processing_time:.4f} seconds"
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
    f"Process average CPU   : "
    f"{average_process_cpu:.2f}%"
)

print(
    f"Process peak CPU      : "
    f"{peak_process_cpu:.2f}%"
)

print(
    f"System average CPU    : "
    f"{average_system_cpu:.2f}%"
)

print(
    f"System peak CPU       : "
    f"{peak_system_cpu:.2f}%"
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
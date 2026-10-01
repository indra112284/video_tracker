import csv
import os
import time

from ultralytics import YOLO

MODEL_PATH = "models/yolo11m.pt"
VIDEO_PATH = "video/PNNL_Parking_LOT(1).avi"

RESULTS_DIR = "results/pt"
DETECTIONS_FILE = os.path.join(RESULTS_DIR, "detections.csv")
METRICS_FILE = os.path.join(RESULTS_DIR, "metrics.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load model
model = YOLO(MODEL_PATH)

# Variables for benchmarking
total_frames = 0
total_inference_time = 0.0
confidence_scores = []

# Open CSV files
with open(DETECTIONS_FILE, "w", newline="") as detection_file:

    detection_writer = csv.writer(detection_file)

    detection_writer.writerow([
        "frame",
        "class",
        "confidence"
    ])

    # stream=True processes one frame at a time
    results = model.predict(
        source=VIDEO_PATH,
        conf=0.25,
        stream=True,
        verbose=True
    )

    for frame_number, result in enumerate(results, start=1):

        # Measure inference time
        start_time = time.perf_counter()

        # Result is already processed by YOLO
        inference_time = time.perf_counter() - start_time

        total_frames += 1

        # Store detections
        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = model.names[class_id]

                confidence_scores.append(confidence)

                detection_writer.writerow([
                    frame_number,
                    class_name,
                    confidence
                ])

        # Ultralytics provides actual inference timing
        if result.speed:
            inference_ms = result.speed.get("inference", 0.0)
            total_inference_time += inference_ms / 1000.0


# Calculate metrics
if total_frames > 0:
    average_inference_time = total_inference_time / total_frames
    fps = total_frames / total_inference_time if total_inference_time > 0 else 0
else:
    average_inference_time = 0
    fps = 0

if confidence_scores:
    average_confidence = sum(confidence_scores) / len(confidence_scores)
    minimum_confidence = min(confidence_scores)
    maximum_confidence = max(confidence_scores)
else:
    average_confidence = 0
    minimum_confidence = 0
    maximum_confidence = 0


# Save metrics
with open(METRICS_FILE, "w", newline="") as metrics_file:

    metrics_writer = csv.writer(metrics_file)

    metrics_writer.writerow([
        "metric",
        "value"
    ])

    metrics_writer.writerow([
        "model",
        "YOLO11m PT"
    ])

    metrics_writer.writerow([
        "total_frames",
        total_frames
    ])

    metrics_writer.writerow([
        "total_inference_time_seconds",
        round(total_inference_time, 4)
    ])

    metrics_writer.writerow([
        "average_inference_time_ms",
        round(average_inference_time * 1000, 4)
    ])

    metrics_writer.writerow([
        "fps",
        round(fps, 4)
    ])

    metrics_writer.writerow([
        "average_confidence",
        round(average_confidence, 4)
    ])

    metrics_writer.writerow([
        "minimum_confidence",
        round(minimum_confidence, 4)
    ])

    metrics_writer.writerow([
        "maximum_confidence",
        round(maximum_confidence, 4)
    ])


print("\nPT benchmark completed successfully.")
print(f"Total frames: {total_frames}")
print(f"Total inference time: {total_inference_time:.2f} seconds")
print(f"Average inference time: {average_inference_time * 1000:.2f} ms")
print(f"FPS: {fps:.2f}")
print(f"Average confidence: {average_confidence:.4f}")
print(f"Results saved in: {RESULTS_DIR}")
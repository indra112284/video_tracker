from ultralytics import YOLO
import csv
import os

MODEL_PATH = "models/yolo11m.onnx"
VIDEO_PATH = "video/PNNL_Parking_LOT(1).avi"
RESULTS_DIR = "results/onnx"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load ONNX model
model = YOLO(MODEL_PATH)

total_frames = 0
total_inference_time_ms = 0.0
confidence_values = []
detection_counts = {}

# Run ONNX inference
results = model.predict(
    source=VIDEO_PATH,
    conf=0.25,
    stream=True,
    save=False,
    show=False
)

for result in results:

    total_frames += 1

    # Inference time
    inference_time = result.speed.get("inference", 0.0)
    total_inference_time_ms += inference_time

    # Confidence values
    if result.boxes is not None and len(result.boxes) > 0:

        confidence_values.extend(
            result.boxes.conf.cpu().tolist()
        )

        # Detection count by class
        for class_id in result.boxes.cls.cpu().tolist():

            class_name = model.names[int(class_id)]

            detection_counts[class_name] = (
                detection_counts.get(class_name, 0) + 1
            )


# Calculate metrics
total_inference_time_seconds = total_inference_time_ms / 1000

average_inference_time_ms = (
    total_inference_time_ms / total_frames
    if total_frames > 0
    else 0
)

fps = (
    1000 / average_inference_time_ms
    if average_inference_time_ms > 0
    else 0
)

average_confidence = (
    sum(confidence_values) / len(confidence_values)
    if confidence_values
    else 0
)

minimum_confidence = (
    min(confidence_values)
    if confidence_values
    else 0
)

maximum_confidence = (
    max(confidence_values)
    if confidence_values
    else 0
)


# Save metrics
metrics = [
    ["metric", "value"],
    ["model", "YOLO11m ONNX"],
    ["total_frames", total_frames],
    [
        "total_inference_time_seconds",
        round(total_inference_time_seconds, 4)
    ],
    [
        "average_inference_time_ms",
        round(average_inference_time_ms, 4)
    ],
    ["fps", round(fps, 4)],
    ["average_confidence", round(average_confidence, 4)],
    ["minimum_confidence", round(minimum_confidence, 4)],
    ["maximum_confidence", round(maximum_confidence, 4)]
]

with open(
    os.path.join(RESULTS_DIR, "metrics.csv"),
    "w",
    newline=""
) as file:

    writer = csv.writer(file)
    writer.writerows(metrics)


# Save detection counts
with open(
    os.path.join(RESULTS_DIR, "detection_counts.csv"),
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(["class", "total_detections"])

    for class_name, count in sorted(detection_counts.items()):
        writer.writerow([class_name, count])


print("\nONNX inference completed successfully.")
print("----------------------------------------")
print(f"Total frames              : {total_frames}")
print(
    f"Total inference time      : "
    f"{total_inference_time_seconds:.2f} seconds"
)
print(
    f"Average inference time    : "
    f"{average_inference_time_ms:.2f} ms/frame"
)
print(f"FPS                       : {fps:.2f}")
print(f"Average confidence        : {average_confidence:.4f}")
print(f"Minimum confidence        : {minimum_confidence:.4f}")
print(f"Maximum confidence        : {maximum_confidence:.4f}")

print("\nDetection counts:")
for class_name, count in sorted(detection_counts.items()):
    print(f"{class_name}: {count}")

print("\nResults saved to:")
print("results/onnx/metrics.csv")
print("results/onnx/detection_counts.csv")
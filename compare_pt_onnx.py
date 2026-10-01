import csv
import os

PT_METRICS = "results/pt/metrics.csv"
ONNX_METRICS = "results/onnx/metrics.csv"

ONNX_COUNTS = "results/onnx/detection_counts.csv"

OUTPUT_DIR = "results/comparison"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pt_vs_onnx.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_metrics(file_path):
    metrics = {}

    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            metrics[row["metric"]] = row["value"]

    return metrics


def read_onnx_counts(file_path):
    counts = {}

    with open(file_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            counts[row["class"]] = int(row["total_detections"])

    return counts


# Read metrics
pt_metrics = read_metrics(PT_METRICS)
onnx_metrics = read_metrics(ONNX_METRICS)

# Metrics to compare
metrics_to_compare = [
    "total_frames",
    "total_inference_time_seconds",
    "average_inference_time_ms",
    "fps",
    "average_confidence",
    "minimum_confidence",
    "maximum_confidence"
]

comparison = []

for metric in metrics_to_compare:

    pt_value = float(pt_metrics[metric])
    onnx_value = float(onnx_metrics[metric])

    difference = onnx_value - pt_value

    if pt_value != 0:
        percentage_difference = (
            difference / pt_value
        ) * 100
    else:
        percentage_difference = 0

    comparison.append([
        metric,
        pt_value,
        onnx_value,
        difference,
        percentage_difference
    ])


# Save PT vs ONNX metric comparison
with open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "metric",
        "PT",
        "ONNX",
        "difference",
        "percentage_difference"
    ])

    writer.writerows(comparison)


# Read ONNX detection counts
onnx_counts = read_onnx_counts(ONNX_COUNTS)

COUNT_FILE = os.path.join(
    OUTPUT_DIR,
    "onnx_detection_counts.csv"
)

with open(COUNT_FILE, "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "class",
        "total_detections"
    ])

    for class_name, count in sorted(onnx_counts.items()):
        writer.writerow([
            class_name,
            count
        ])


# Display comparison
print("\nPT vs ONNX COMPARISON")
print("=" * 75)

print(
    f"{'Metric':35}"
    f"{'PT':>12}"
    f"{'ONNX':>12}"
    f"{'Difference':>12}"
)

print("-" * 75)

for row in comparison:

    metric = row[0]
    pt_value = row[1]
    onnx_value = row[2]
    difference = row[3]

    print(
        f"{metric:35}"
        f"{pt_value:12.4f}"
        f"{onnx_value:12.4f}"
        f"{difference:12.4f}"
    )


print("\nONNX Detection Counts")
print("=" * 50)

for class_name, count in sorted(onnx_counts.items()):
    print(f"{class_name:20} {count}")


print("\nComparison completed successfully.")

print(f"\nSaved:")
print(OUTPUT_FILE)
print(COUNT_FILE)
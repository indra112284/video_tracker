import cv2
import os

VIDEO_PATH = "video/PNNL_Parking_LOT(1).avi"
OUTPUT_DIR = "ground_truth/frames"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Extract 20 evenly spaced frames
frame_numbers = [
    int(i * (total_frames - 1) / 19)
    for i in range(20)
]

for frame_number in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if success:
        filename = os.path.join(
            OUTPUT_DIR,
            f"frame_{frame_number + 1:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

cap.release()

print(f"Total video frames: {total_frames}")
print(f"Extracted frames: {len(frame_numbers)}")
print(f"Saved to: {OUTPUT_DIR}")
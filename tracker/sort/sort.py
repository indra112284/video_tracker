import numpy as np
from collections import deque


def iou_batch(bb_test, bb_gt):
    """
    Calculate IoU between two sets of bounding boxes.
    """

    bb_test = np.asarray(bb_test)
    bb_gt = np.asarray(bb_gt)

    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)))

    xx1 = np.maximum(
        bb_test[:, 0][:, None],
        bb_gt[:, 0][None, :]
    )

    yy1 = np.maximum(
        bb_test[:, 1][:, None],
        bb_gt[:, 1][None, :]
    )

    xx2 = np.minimum(
        bb_test[:, 2][:, None],
        bb_gt[:, 2][None, :]
    )

    yy2 = np.minimum(
        bb_test[:, 3][:, None],
        bb_gt[:, 3][None, :]
    )

    width = np.maximum(0.0, xx2 - xx1)
    height = np.maximum(0.0, yy2 - yy1)

    intersection = width * height

    area_test = (
        (bb_test[:, 2] - bb_test[:, 0]) *
        (bb_test[:, 3] - bb_test[:, 1])
    )

    area_gt = (
        (bb_gt[:, 2] - bb_gt[:, 0]) *
        (bb_gt[:, 3] - bb_gt[:, 1])
    )

    union = (
        area_test[:, None] +
        area_gt[None, :] -
        intersection
    )

    return intersection / np.maximum(union, 1e-6)


class KalmanBoxTracker:

    count = 0

    def __init__(self, bbox, class_id):

        self.bbox = np.asarray(bbox, dtype=float)

        self.class_id = class_id

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        self.history = deque(maxlen=10)

    def predict(self):

        self.age += 1
        self.time_since_update += 1

        self.history.append(self.bbox.copy())

        return self.bbox.copy()

    def update(self, bbox):

        self.bbox = np.asarray(bbox, dtype=float)

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self):

        return self.bbox.copy()


class Sort:

    def __init__(
        self,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3
    ):

        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers = []

        self.frame_count = 0

    def update(self, detections):

        self.frame_count += 1

        detections = np.asarray(detections)

        if detections.size == 0:

            for tracker in self.trackers:
                tracker.predict()

            self.trackers = [
                tracker
                for tracker in self.trackers
                if tracker.time_since_update <= self.max_age
            ]

            return np.empty((0, 6))

        # Detection format:
        # [x1, y1, x2, y2, confidence, class_id]

        predictions = []

        for tracker in self.trackers:

            prediction = tracker.predict()

            predictions.append(prediction)

        if len(predictions) > 0:

            predictions = np.asarray(predictions)

            ious = iou_batch(
                detections[:, :4],
                predictions
            )

        else:

            ious = np.empty(
                (len(detections), 0)
            )

        matched = []

        unmatched_detections = list(
            range(len(detections))
        )

        unmatched_trackers = list(
            range(len(self.trackers))
        )

        if ious.size > 0:

            while True:

                detection_index, tracker_index = np.unravel_index(
                    np.argmax(ious),
                    ious.shape
                )

                best_iou = ious[
                    detection_index,
                    tracker_index
                ]

                if best_iou < self.iou_threshold:
                    break

                matched.append(
                    (
                        detection_index,
                        tracker_index
                    )
                )

                ious[detection_index, :] = -1
                ious[:, tracker_index] = -1

                if np.max(ious) < self.iou_threshold:
                    break

            matched_detection_indices = {
                match[0] for match in matched
            }

            matched_tracker_indices = {
                match[1] for match in matched
            }

            unmatched_detections = [
                i
                for i in range(len(detections))
                if i not in matched_detection_indices
            ]

            unmatched_trackers = [
                i
                for i in range(len(self.trackers))
                if i not in matched_tracker_indices
            ]

        # Update matched trackers

        for detection_index, tracker_index in matched:

            detection = detections[detection_index]

            self.trackers[tracker_index].update(
                detection[:4]
            )

            self.trackers[tracker_index].class_id = int(
                detection[5]
            )

        # Create new trackers

        for detection_index in unmatched_detections:

            detection = detections[detection_index]

            tracker = KalmanBoxTracker(
                detection[:4],
                int(detection[5])
            )

            self.trackers.append(tracker)

        # Prepare output

        outputs = []

        for tracker in self.trackers:

            if (
                tracker.time_since_update == 0
                and (
                    tracker.hits >= self.min_hits
                    or self.frame_count <= self.min_hits
                )
            ):

                bbox = tracker.get_state()

                outputs.append(
                    [
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                        tracker.id,
                        tracker.class_id
                    ]
                )

        # Remove old trackers

        self.trackers = [
            tracker
            for tracker in self.trackers
            if tracker.time_since_update <= self.max_age
        ]

        return np.asarray(outputs)
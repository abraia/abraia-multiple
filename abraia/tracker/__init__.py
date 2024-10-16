import numpy as np

try:
    import scipy
except ImportError:
    print('Install the scipy package to work with byte tracker')

from .byte_tracker import ByteTracker
from . import matching


def results_to_arrays(results):
        bboxes, scores = [], []
        for result in results:
            x, y, w, h = result['box']
            bboxes.append([x, y, x + w, y + h])
            scores.append(result['confidence'])
        return np.array(bboxes), np.array(scores)


class Tracker():
    def __init__(self, track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=30):
        self.tracker = ByteTracker(track_thresh, track_buffer, match_thresh, frame_rate)

    def update(self, results):
        if results:
            bboxes, scores = results_to_arrays(results)
            tracks = self.tracker.update(bboxes, scores)
            if len(tracks) > 0:
                track_bounding_boxes = np.asarray([track.tlbr for track in tracks])
                ious = matching.box_iou_batch(bboxes, track_bounding_boxes)
                matches, _, _ = matching.linear_assignment(1 - ious, 0.5)
                for i_detection, i_track in matches:
                    results[i_detection]['tracker_id'] = tracks[i_track].track_id
        return results


__all__ = ['Tracker']
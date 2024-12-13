import numpy as np
from scipy.optimize import linear_sum_assignment
from simple_tracker.track import Track

class Tracker:
    def __init__(self):
        self.tracks_ = {}
        self.id_ = 0
        self.kMaxCoastCycles = 10  # Adjust this value as needed

    @staticmethod
    def calculate_iou(det, track):
        trk = track.get_state_as_bbox()
        
        # Get min/max points
        xx1 = max(det[0], trk[0])
        yy1 = max(det[1], trk[1])
        xx2 = min(det[0] + det[2], trk[0] + trk[2])
        yy2 = min(det[1] + det[3], trk[1] + trk[3])
        
        w = max(0, xx2 - xx1)
        h = max(0, yy2 - yy1)

        # Calculate area of intersection and union
        det_area = det[2] * det[3]
        trk_area = trk[2] * trk[3]
        intersection_area = w * h
        union_area = det_area + trk_area - intersection_area
        
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou

    @staticmethod
    def hungarian_matching(iou_matrix):
        # Convert to cost matrix
        cost_matrix = -np.array(iou_matrix)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def associate_detections_to_tracks(self, detections, tracks, iou_threshold=0.1):
        if not tracks:
            return {}, detections

        iou_matrix = np.zeros((len(detections), len(tracks)))
        for i, det in enumerate(detections):
            for j, (_, trk) in enumerate(tracks.items()):
                iou_matrix[i, j] = self.calculate_iou(det, trk)

        row_ind, col_ind = self.hungarian_matching(iou_matrix)

        matched = {}
        unmatched_det = []

        for i, det in enumerate(detections):
            if i in row_ind:
                j = col_ind[list(row_ind).index(i)]
                if iou_matrix[i, j] >= iou_threshold:
                    track_id = list(tracks.keys())[j]
                    matched[track_id] = det
                else:
                    unmatched_det.append(det)
            else:
                unmatched_det.append(det)

        return matched, unmatched_det

    def run(self, detections):
        # Predict internal tracks from previous frame
        for track in self.tracks_.values():
            track.predict()

        matched, unmatched_det = self.associate_detections_to_tracks(detections, self.tracks_)

        # Update tracks with associated bbox
        for track_id, det in matched.items():
            self.tracks_[track_id].update(det)

        # Create new tracks for unmatched detections
        for det in unmatched_det:
            track = Track()
            track.init(det)
            self.tracks_[self.id_] = track
            self.id_ += 1

        # Delete lost tracked tracks
        self.tracks_ = {track_id: track for track_id, track in self.tracks_.items() 
                        if track.coast_cycles_ <= self.kMaxCoastCycles}
    
    def update_dt(self, dt):
        for track_id, track in self.tracks_.items():
            track.set_dt(dt)

    def get_tracks(self):
        return self.tracks_
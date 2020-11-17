import torch
import numpy as np
from Predictive_Tracker.iou3d import iou3d
from scipy.optimize import linear_sum_assignment
from Predictive_Tracker.utils import label_to_str, label_to_box

class NaiveTracker:
    def __init__(self):
        self.id_count = 0
        self.trackers = {}

    def update(self, detections):

        matched, unmatched_detections, unmatched_trackers = assign_detections_to_trackers(detections, self.trackers)

        print(f"Matched {len(matched)} of {len(self.trackers)} trackers and {len(detections)} detections.")

        for t, trk_id in enumerate(self.trackers.keys()):
            if trk_id not in unmatched_trackers:
                d = matched[np.where(matched[:,1] == t)[0],0][0]
                self.trackers[trk_id] = detections[d]

        print(f"Creating {len(unmatched_detections)} trackers for unmatched detections.")

        print("Unmatched: ", end='')
        for i in unmatched_detections:
            print(f"{i} ({label_to_str[detections[i][7]]})", end=', ')
            self.trackers[self.id_count] = detections[i]
            self.id_count += 1
        print()

        print(f"Removing {len(unmatched_trackers)} unmatched trackers.")
        
        for trk_id in unmatched_trackers:
            del self.trackers[trk_id]

        hypothesis = {}

        for id, box in self.trackers.items():
            hypothesis[id] = box

        return hypothesis

def assign_detections_to_trackers(detections, trackers, iou_threshold=0.1):
    """
    Assigns detections to tracked object (both represented as bounding boxes)
    detections:  N x 8 x 3
    trackers:    M x 8 x 3
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), range(len(detections)), [] 

    tracker_state = np.array(list(trackers.values()))

    cost_matrix = iou3d(
        torch.Tensor(detections)[:,:7],
        torch.Tensor(tracker_state)[:,:7]
    ).numpy()

    # set overlap of boxes with different types to 0
    for d in range(cost_matrix.shape[0]):
        for p in range(cost_matrix.shape[1]):
            if detections[d,7] != tracker_state[p,7]:
                cost_matrix[d,p] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    matched_indices = np.stack((row_ind, col_ind), axis=1)

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]): 
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t, trk_id in enumerate(trackers.keys()):
        if (t not in matched_indices[:, 1]): unmatched_trackers.append(trk_id)

    matches = []
    for m in matched_indices:
        if cost_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(list(trackers.keys())[m[1]])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, unmatched_detections, unmatched_trackers

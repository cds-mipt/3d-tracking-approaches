# cds-3d-tracking-approaches
This repository contains several tracking approaches based on 3D-object detection.

The approach is heavily influenced by [Weng's Baseline for 3D Multi-Object Tracking](https://github.com/xinshuoweng/AB3DMOT) and [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275).

We assumend that you have already solved the 3D object detection task and the 3D bounding boxes will be as inputs to the tracker.

## Velocity Tracker:

The best implemetation of such a tracker can be found in [Center-based 3D Object Detection and Tracking](https://arxiv.org/abs/2006.11275) and it is based on linear assignment (Hungarian Algorithm). This approach uses bounding box position xyz and velocity vector (vx,vy) and returns the input data with additional ID-field.

### How to implement: 
 
 ```python
 from Speed_Tracker import PubTracker as Tracker
 # initialize
 tracker = Tracker(max_age=31, hungarian=True)
 tracker.reset()
 
 # prepare detections:
   pred = {
                  "sample_token": i,
                  "translation": box3d[i, :3].tolist(),
                  "size": box3d[i, 3:6].tolist(),
                  "rotation": [quat[3], quat[0], quat[1], quat[2]],
                  "velocity": velocity,
                  "detection_name": int(types[i].tolist()),
                  "detection_score": scores[i].tolist(),
                  "attribute_name": '',
              }
   preds.append(pred)
   
 # update:
 tracks = self.tracker.step_centertrack(preds, time_lag)
 
 # getting IDs
 for item in tracks:                      
             if item['active'] == 0:
                 continue
             ID.append(item["tracking_id"])
 ```


## AB3DMOT Tracker:

AB3DMOT approach is applied in [Weng's Baseline for 3D Multi-Object Tracking](https://github.com/xinshuoweng/AB3DMOT) and it aggregates hungarian alogorithm with kalman filter to predict the unmatched detections with unmatched tracks.
The input to this approach is the 3D bounding-box information ( np.array of  h, w, l, x, y, z, yaw ) and the output is the input data with additional ID-field.

### How to implement:

```python
 from AB3DMOT_Tracker.model import AB3DMOT
 
 # initialize
 mot_tracker = AB3DMOT()
 
 # prepare detections:
   det_mot_tracker = [box3d[i, 5].tolist(),box3d[i, 3].tolist(),box3d[i, 4].tolist(),box3d[i, 0].tolist(),box3d[i, 1].tolist(),box3d[i, 2].tolist(), yaw]
   dets.append(det_mot_tracker)
 dets = np.array(dets)
 
 # update:
 tracks = mot_tracker.update(dets) 
 
 # getting IDs:
 for item in tracks:                     
            ID.append(int(item[7]))  
 ```
 
 
 ## Predictive Tracker:
 
 Predictive Tracker is an improved version of the previous tracker (AB3DMOT) as it aggregates hungarian alogorithm with kalman filter. here you can choose between the naive approach and the predictive one.
The input to this approach is the 3D bounding-box inforamtion and its label ( np.array of x, y, z, w, l, h, yaw, label ) and the output is the input data with additional ID-field.

### How to implement:

```python
 from Predictive_Tracker.PredictiveTracker import PredictiveTracker
 from Predictive_Tracker.NaiveTracker import NaiveTracker
 
 # initialize
 pred_tracker = PredictiveTracker()
 naiv_tracker = NaiveTracker()
 
 # prepare detections:
       det = [box3d[i, 0].tolist(),box3d[i, 1].tolist(),box3d[i, 2].tolist(),box3d[i, 3].tolist(),box3d[i, 4].tolist(),box3d[i, 5].tolist(), yaw, types[i].tolist() ]
       dets.append(det)
 dets = np.array(dets)
 
 # update:
 tracks1 = pred_tracker.update(dets)                                         # use hungarian + kalman with filter speed    
 tracks2 = naiv_tracker.update(dets)   
 
 # getting IDs:
 for key in tracks.keys():                
            ID.append(key)
 ```

 ## Probabilistic Tracker: 
Probabilistic Tracker is an on-line tracking method. Its code is based on [Probabilistic 3D Multi-Object Tracking for Autonomous Driving](https://arxiv.org/abs/2001.05673) report.

### How to implement:
-- soon

 

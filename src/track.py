import numpy as np

from python.object_tracker.src.kalmanFilter import KalmanFilter

class Track:
    def __init__(self):
        # Initialize Kalman Filter with 8 states and 4 observations
        self.kf_ = KalmanFilter(8, 4)

        # Define constant velocity model for the Kalman Filter
        dt = 1.0  # Time step (you may adjust this)
        
        # State transition matrix F
        self.kf_.F[0:4, 0:4] = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])
        
        # Velocity components in the state transition matrix
        for i in range(4):
            self.kf_.F[i][i + 4] = dt
        
        # Initial error covariance matrix P with high uncertainty for velocities
        self.kf_.P[0:4, 0:4] = np.eye(4) * 10
        for i in range(4, 8):
            self.kf_.P[i][i] = 10000
        
        # Observation matrix H
        self.kf_.H[0:4, 0:4] = np.eye(4)

        # Process noise covariance Q
        self.kf_.Q[0:4, 0:4] = np.eye(4)
        
        for i in range(4):
            self.kf_.Q[i + 4][i + 4] = 0.1
            
        # Measurement noise covariance R
        self.kf_.R[0:2, 0:2] = np.eye(2) * 10
        for i in range(2):
            self.kf_.R[i + 2][i + 2] = 100

    def init(self, bbox):
        """ Initialize tracker with initial bounding box. """
        
        observation = self.convert_bbox_to_observation(bbox)
        
        # Set initial state estimate based on bounding box observation
        self.kf_.x[0:4] = observation
        
    def predict(self):
        """ Predict the next state of the tracker. """
        
        # Call the Kalman filter's predict method
        self.kf_.predict()
        
    def update(self, bbox):
        """ Update the tracker with a new bounding box. """
        
        observation = self.convert_bbox_to_observation(bbox)
        
        # Update coast cycle count and hit streak count
        if hasattr(self, 'coast_cycles_'):
            if getattr(self, 'coast_cycles_', 0) > 0:
                getattr(self, 'hit_streak_', 0)
            getattr(self, 'coast_cycles_', 0) += 1
            
            if not hasattr(self, 'hit_streak_'):
                setattr(self, 'hit_streak_', 1)
            else:
                setattr(self, 'hit_streak_', getattr(self, 'hit_streak_') + 1)

            # Update Kalman filter with new observation
            self.kf_.update(observation)

    def get_state_as_bbox(self):
        """ Returns the current bounding box estimate. """
        
        return self.convert_state_to_bbox(self.kf_.x)

    def convert_bbox_to_observation(self, bbox):
      """ Convert a bounding box to an observation vector. """
      
      center_x = bbox[0] + bbox[2] / 2.0
      center_y = bbox[1] + bbox[3] / 2.0
      
      return np.array([center_x, center_y, bbox[2], bbox[3]])

    def convert_state_to_bbox(self, state):
      """ Convert a state vector to a bounding box. """
      
      width = int(state[2])
      height = int(state[3])
      tl_x = int(state[0] - width / 2.0)
      tl_y = int(state[1] - height / 2.0)
      
      return (tl_x, tl_y, width, height) 

# Example usage:
# tracker_instance = Track()
# tracker_instance.init((x_min,y_min,width,height))
# tracker_instance.predict()
# tracker_instance.update((x_min,y_min,width,height))
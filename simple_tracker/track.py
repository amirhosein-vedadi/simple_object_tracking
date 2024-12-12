import numpy as np
from simple_tracker.kalmanFilter import KalmanFilter

class Track:
    def __init__(self):
        # Initialize Kalman Filter with 8 states and 4 observations
        self.kf_ = KalmanFilter(8, 4)

        self.dt = 1.0 / 200  # Assuming 25 fps
        
        self.kf_.F[0:4, 0:4] = np.array([[1, 0, 0, 0],
                                          [0, 1, 0, 0],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]])
        
        # Velocity components in the state transition matrix
        for i in range(4):
            self.kf_.F[i][i + 4] = self.dt
        
        # Initial error covariance matrix P with high uncertainty for velocities
        self.kf_.P[0:4, 0:4] = np.eye(4) * 10
        for i in range(4, 8):
            self.kf_.P[i][i] = 10000
        
        # Observation matrix H
        self.kf_.H[0:4, 0:4] = np.eye(4)

        # Process noise covariance Q - Increase these values to trust measurements more
        self.kf_.Q[0:4, 0:4] = np.eye(4) * 5.0   # Lower process noise for position states
        for i in range(4):
            self.kf_.Q[i + 4][i + 4] = 1.0         # Lower process noise for velocity states
            
        # Measurement noise covariance R - Decrease these values to trust measurements more
        self.kf_.R[0:2, 0:2] = np.eye(2) * 1.0   # Lower measurement noise for position
        for i in range(2):
            self.kf_.R[i + 2][i + 2] = 10          # Higher measurement noise for size (width/height)

        # Initialize coast cycles and hit streak
        self.coast_cycles_ = 0
        self.hit_streak_ = 0

    def set_dt(self, dt):
        self.dt = dt
        self.kf_.F[0:4, 4:8] = np.array([[dt if i == j else 0 for j in range(4)] for i in range(4)])

    def init(self, bbox):
        """ Initialize tracker with initial bounding box. """
        
        observation = self.convert_bbox_to_observation(bbox)
        
        # Set initial state estimate based on bounding box observation
        self.kf_.x[0:4] = observation
        
    def predict(self):
        """ Predict the next state of the tracker. """
        
        self.kf_.predict()
        if self.coast_cycles_ > 0:
            self.hit_streak_ = 0
        self.coast_cycles_+= 1
        
    def update(self, bbox):
        """ Update the tracker with a new bounding box. """
        
        observation = self.convert_bbox_to_observation(bbox)
        
        self.coast_cycles_ = 0
        self.hit_streak_ += 1
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

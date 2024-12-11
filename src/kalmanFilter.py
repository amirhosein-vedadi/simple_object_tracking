import numpy as np

class KalmanFilter:
    def __init__(self, num_states, num_obs):
        self.num_states = num_states
        self.num_obs = num_obs
        
        # State vector
        self.x = np.zeros(num_states)
        # Predicted state vector
        self.x_predict = np.zeros(num_states)

        # State transition matrix
        self.F = np.zeros((num_states, num_states))
        
        # Error covariance matrix
        self.P = np.zeros((num_states, num_states))
        # Predicted error covariance matrix
        self.P_predict = np.zeros((num_states, num_states))

        # Covariance matrix of process noise
        self.Q = np.zeros((num_states, num_states))

        # Observation matrix
        self.H = np.zeros((num_obs, num_states))

        # Covariance matrix of observation noise
        self.R = np.zeros((num_obs, num_obs))

    def coast(self):
        self.x_predict = self.F @ self.x
        self.P_predict = self.F @ self.P @ self.F.T + self.Q

    def predict(self):
        self.coast()
        self.x = self.x_predict
        self.P = self.P_predict

    def prediction_to_observation(self, state):
        return self.H @ state

    def update(self, z):
        z_predict = self.prediction_to_observation(self.x_predict)

        # Innovation
        y = z - z_predict
        
        Ht = self.H.T

        # Innovation covariance
        S = self.H @ self.P_predict @ Ht + self.R
        
        # Normalized Innovation Squared (NIS)
        self.NIS = y.T @ np.linalg.inv(S) @ y

        # Kalman gain
        K = self.P_predict @ Ht @ np.linalg.inv(S)

        # Updated state estimation
        self.x += K @ y

        I = np.eye(self.num_states)
        
        # Updated error covariance
        self.P = (I - K @ self.H) @ self.P_predict
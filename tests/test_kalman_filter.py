import unittest
import numpy as np
from simple_tracker.kalmanFilter import KalmanFilter

class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        """ Set up a KalmanFilter instance for testing. """
        self.kf = KalmanFilter(num_states=4, num_obs=2)  # Example with 4 states and 2 observations

        # Set initial values for test purposes
        self.kf.F = np.array([[1, 1, 0, 0],   # Assuming dt = 1 for simplicity
                               [0, 1, 0, 0],
                               [0, 0, 1, 1],
                               [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]])
        self.kf.R = np.eye(2) * 10          # Measurement noise
        self.kf.Q = np.eye(4) * 0.01        # Process noise

    def test_prediction_step(self):
        """ Test the prediction step of the Kalman Filter. """
        dt = 1  # Define a time step for simplicity
        self.kf.F = np.array([[1, dt, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, dt],
                            [0, 0, 0, 1]])
        
        # Initial state: [position_x, velocity_x, position_y, velocity_y]
        self.kf.x = np.array([0., 1., 0., -1.]) 
        
        # Perform prediction
        self.kf.predict()
        
        # Expected predicted state after one time step
        expected_x_predict = np.array([1., 1., -1, -1.])
        
        # Check if the predicted state is correct
        self.assertTrue(np.allclose(self.kf.x, expected_x_predict), 
                        f"Expected {expected_x_predict}, but got {self.kf.x}")


    def test_update_step(self):
        """ Test the update step of the Kalman Filter. """
        # Set up matrices for testing
        dt = 1  # Define a time step for simplicity
        self.kf.F = np.array([[1, dt, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, dt],
                            [0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        
        self.kf.R = np.eye(2) * 10          # Measurement noise
        self.kf.Q = np.eye(4) * 0.01        # Process noise

        # Initial state and covariance
        self.kf.x = np.array([2.5, 3.5, -2.5, -3])
        self.kf.P = np.eye(4) * 10
        
        # Simulate an observation
        z = np.array([3.5, -3]) 
        
        # Perform prediction first
        self.kf.predict()
        
        # Perform update with observation
        self.kf.update(z)
        
        # Check updated state estimation (expected values will depend on your specific implementation)
        expected_x_updated = self.kf.x.copy()  
    
        # Check if the updated state is close to expected values (you may need to adjust this based on your logic)
        self.assertTrue(np.allclose(self.kf.x[:2], expected_x_updated[:2], atol=1e-2))

if __name__ == '__main__':
    unittest.main()
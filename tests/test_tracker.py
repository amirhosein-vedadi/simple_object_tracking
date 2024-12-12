import unittest
from simple_tracker import Tracker, Track

class TestTrack(unittest.TestCase):
    def setUp(self):
        """ Set up a Track instance for testing. """
        self.track = Track()

    def test_initialization(self):
        """ Test if the track is initialized correctly. """
        self.assertEqual(self.track.coast_cycles_, 0)
        self.assertEqual(self.track.hit_streak_, 0)

    def test_init_with_bbox(self):
        """ Test the initialization with a bounding box. """
        bbox = [100, 100, 50, 50]  # Example bounding box: [x_min, y_min, width, height]
        self.track.init(bbox)
        state = self.track.kf_.x[:4]  # Get the position state from Kalman Filter
        expected_center_x = bbox[0] + bbox[2] / 2.0
        expected_center_y = bbox[1] + bbox[3] / 2.0
        
        self.assertAlmostEqual(state[0], expected_center_x)
        self.assertAlmostEqual(state[1], expected_center_y)

    def test_update(self):
        """ Test updating the track with a new bounding box. """
        bbox = [100, 100, 50, 50]
        self.track.init(bbox)
        
        new_bbox = [110, 110, 50, 50]
        self.track.update(new_bbox)
        
        # Check if coast cycles are reset after a valid update
        self.assertEqual(self.track.coast_cycles_, 0)

    def test_coast_cycles_increment(self):
        """ Test coast cycles increment when no updates are received. """
        self.track.update([100, 100, 50, 50])  # Valid update
        self.track.coast_cycles_ += 1          # Simulate no update
        
        # Increment coast cycles without valid detection
        self.track.update(None)                  # Simulate no detection
        
        self.assertEqual(self.track.coast_cycles_, 1)

class TestTracker(unittest.TestCase):
    def setUp(self):
        """ Set up a Tracker instance for testing. """
        self.tracker = Tracker()

    def test_initialization(self):
        """ Test if the tracker initializes correctly. """
        self.assertIsNotNone(self.tracker.tracks_)
    
    def test_run_with_detections(self):
        """ Test running the tracker with detections. """
        detections = [[100, 100, 50, 50], [200, 200, 60, 60]]  # Example detections
        
        # Run tracker with initial detections
        self.tracker.run(detections)
        
        # Check if tracks are created
        current_tracks = self.tracker.get_tracks()
        
        self.assertEqual(len(current_tracks), len(detections))

if __name__ == '__main__':
    unittest.main()

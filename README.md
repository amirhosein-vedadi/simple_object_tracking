# Simple Object Tracker in Python
The **Simple Object Tracker** is a Python project that utilizes Kalman Filter for tracking detected objects in video streams.
<p align="center">
  <a href="https://github.com/amirhosein-vedadi/simple_object_tracking">
    <img width="70%" src="data/videos/output.gif" alt="logo">
  </a>
</p>

## Installation

It's better to create a separate python environment:
  ```bash
  conda create -n tracker python=3.12
  conda activate tracker
  ```

Then to install the Simple Object Tracker, follow these steps:

   ```bash
   git clone https://github.com/amirhosein-vedadi/simple_object_tracking.git
   
   cd simple_object_tracking

   pip install -e .
   ```
## Usage

To demonstrate how to use the Simple Object Tracker, an example script is provided in the examples directory. The example script initializes the YOLO model and the tracker, processes each frame of the video, displays the tracking results and saves the output video.

```bash
cd examples
python yolo_tracker.py
```
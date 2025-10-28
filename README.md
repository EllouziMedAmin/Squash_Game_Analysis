# 3D Pose Estimation & Joint Angle Calculator

A lightweight Python tool for extracting 3D human pose skeletons and calculating joint angles from video files using Google's MediaPipe framework.

---

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Output Format](#output-format)
- [Joint Angles Calculated](#joint-angles-calculated)
- [Technical Details](#technical-details)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## ✨ Features

- **Real 3D Pose Detection**: Extracts 33 body landmarks with world coordinates (x, y, z)
- **Joint Angle Calculation**: Computes 9 key joint angles using 3D vector mathematics
- **Video Processing**: Batch process entire videos with skeleton overlay
- **Temporal Smoothing**: Built-in smoothing for stable tracking across frames
- **Optimized Performance**: FPS decimation for faster processing without quality loss
- **Multiple Outputs**: 
  - Annotated video with skeleton visualization
  - JSON file with all pose data and angles
  - Statistical summary report
- **Sports Analysis Ready**: Designed for biomechanical analysis of dynamic movements

---

## 🔧 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
# Clone or download the repository
git clone https://github.com/EllouziMedAmin/Squash_Game_Analysis.git
cd squash analysis

# Install required packages
pip install mediapipe opencv-python numpy

# Or use requirements.txt
pip install -r requirements.txt
```

### Requirements File

Create a `requirements.txt`:
```
mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.21.0
```

---

## 🚀 Quick Start

### Basic Usage

```bash
# Process a video with default settings
python pose-estimation.py input_video.mp4
```

This will generate:
- `output_pose_input_video.mp4` - Video with skeleton overlay
- `pose_data_input_video.json` - Detailed pose data

### Example Output

```
============================================================
🎯 3D Pose Estimation with MediaPipe
============================================================

📹 Video Properties:
   Resolution: 1920x1080
   FPS: 30
   Total Frames: 900
   Processing every 3 frames (~10 FPS)

🔄 Processing video...
   Progress: 100.0% (300 frames processed)

✅ Processing complete!
   Total frames processed: 300
   Poses detected: 295

💾 Results saved to: pose_data_input_video.json

📊 Joint Angle Summary (degrees):
------------------------------------------------------------
Left Elbow          | Mean:  145.3° | Min:   89.2° | Max:  178.4° | Std:   18.7°
Right Elbow         | Mean:  142.8° | Min:   85.6° | Max:  179.1° | Std:   19.3°
Left Knee           | Mean:  168.5° | Min:  120.3° | Max:  179.8° | Std:   12.4°
...

🎬 Output video: output_pose_input_video.mp4
📄 Output JSON: pose_data_input_video.json
```

---

## 📖 Usage

### Command Line Arguments

```bash
python pose-estimation.py <video_path> [options]
```

#### Required Arguments

- `video_path` - Path to input video file (MP4, AVI, MOV, etc.)

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-video` | `output_pose_<name>.mp4` | Path to save annotated output video |
| `--output-json` | `pose_data_<name>.json` | Path to save pose data JSON |
| `--fps-decimation` | `3` | Process every Nth frame (higher = faster, lower = more data) |

### Advanced Examples

#### Process at different frame rates:

```bash
# High quality (process every frame, slow)
python pose-estimation.py video.mp4 --fps-decimation 1

# Balanced (default, ~10 FPS for 30fps video)
python pose-estimation.py video.mp4 --fps-decimation 3

# Fast processing (~5 FPS for 30fps video)
python pose-estimation.py video.mp4 --fps-decimation 6
```

#### Custom output paths:

```bash
python pose-estimation.py athlete_match.mp4 \
    --output-video analysis/match_skeleton.mp4 \
    --output-json analysis/match_data.json \
    --fps-decimation 3
```

#### Process multiple videos:

```bash
# Bash script for batch processing
for video in videos/*.mp4; do
    python pose-estimation.py "$video" --fps-decimation 3
done
```

---

## 📊 Output Format

### 1. Annotated Video

The output video includes:
- **Skeleton overlay**: 33 keypoints connected by lines
- **Joint angles**: Real-time display of calculated angles
- **Same resolution**: Maintains original video quality

### 2. JSON Data Structure

```json
[
  {
    "frame_number": 30,
    "timestamp": 1.0,
    "joint_angles": {
      "right_elbow": 145.23,
      "left_elbow": 152.87,
      "right_shoulder": 89.45,
      "left_shoulder": 91.23,
      "right_knee": 168.34,
      "left_knee": 170.12,
      "right_hip": 178.90,
      "left_hip": 179.45,
      "torso_lean": 85.67
    },
    "landmarks_3d": [
      {
        "id": 0,
        "name": "NOSE",
        "x": 0.5123,
        "y": 0.3456,
        "z": -0.2345,
        "visibility": 0.9876
      },
      ...
    ]
  },
  ...
]
```

### 3. Console Summary

Statistical summary printed to console:
- Mean, Min, Max, Standard Deviation for each joint angle
- Total frames processed
- Detection success rate

---

## 🦴 Joint Angles Calculated

The tool calculates **9 biomechanically important joint angles** using 3D world coordinates:

| Joint Angle | Landmarks Used | Description |
|-------------|----------------|-------------|
| **Right Elbow** | Shoulder → Elbow → Wrist | Arm flexion/extension |
| **Left Elbow** | Shoulder → Elbow → Wrist | Arm flexion/extension |
| **Right Shoulder** | Elbow → Shoulder → Hip | Shoulder abduction/adduction |
| **Left Shoulder** | Elbow → Shoulder → Hip | Shoulder abduction/adduction |
| **Right Knee** | Hip → Knee → Ankle | Leg flexion/extension |
| **Left Knee** | Hip → Knee → Ankle | Leg flexion/extension |
| **Right Hip** | Shoulder → Hip → Knee | Hip flexion/extension |
| **Left Hip** | Shoulder → Hip → Knee | Hip flexion/extension |
| **Torso Lean** | Shoulder Mid → Hip Mid → Vertical | Body lean angle |

### Angle Calculation Method

Uses the **3D dot product method** for accurate angle calculation:

```python
# Vector from b to a
ba = [a.x - b.x, a.y - b.y, a.z - b.z]

# Vector from b to c
bc = [c.x - b.x, c.y - b.y, c.z - b.z]

# Calculate angle
cos(θ) = (ba · bc) / (|ba| × |bc|)
angle = arccos(cos(θ)) × 180/π
```

---

## 🔬 Technical Details

### MediaPipe Configuration

```python
mp.solutions.pose.Pose(
    static_image_mode=False,        # Video mode
    model_complexity=2,             # 0=Lite, 1=Full, 2=Heavy (most accurate)
    smooth_landmarks=True,          # Temporal smoothing enabled
    enable_segmentation=False,      # Not needed for pose only
    min_detection_confidence=0.5,   # Detection threshold
    min_tracking_confidence=0.5     # Tracking threshold
)
```

### Keypoint Structure

MediaPipe provides **33 landmarks** per frame:

```
0: nose                    17: left_pinky
1: left_eye_inner         18: right_pinky
2: left_eye               19: left_index
3: left_eye_outer         20: right_index
4: right_eye_inner        21: left_thumb
5: right_eye              22: right_thumb
6: right_eye_outer        23: left_hip
7: left_ear               24: right_hip
8: right_ear              25: left_knee
9: mouth_left             26: right_knee
10: mouth_right           27: left_ankle
11: left_shoulder         28: right_ankle
12: right_shoulder        29: left_heel
13: left_elbow            30: right_heel
14: right_elbow           31: left_foot_index
15: left_wrist            32: right_foot_index
16: right_wrist
```

### Coordinate System

- **2D Landmarks**: Normalized [0, 1] relative to frame dimensions
- **3D World Landmarks**: Real-world coordinates in meters
  - `x`: Horizontal (left/right)
  - `y`: Vertical (up/down)
  - `z`: Depth (forward/backward, relative to hip center)

---

## ⚡ Performance Optimization

### Recommended Settings for Different Use Cases

#### Real-time Analysis (Fastest)
```bash
python pose-estimation.py video.mp4 --fps-decimation 10
# Processes ~3 FPS for 30fps video
# Processing time: ~0.5x video length
```

#### Biomechanical Analysis (Balanced)
```bash
python pose-estimation.py video.mp4 --fps-decimation 3
# Processes ~10 FPS for 30fps video
# Processing time: ~1.5x video length
# ✅ Recommended for sports analysis
```

#### High-Precision Research (Slowest)
```bash
python pose-estimation.py video.mp4 --fps-decimation 1
# Processes all frames
# Processing time: ~3-5x video length
```

### Performance Benchmarks

Tested on Intel i7 CPU (no GPU):

| Video Length | FPS Decimation | Processing Time | Frames Analyzed |
|--------------|----------------|-----------------|-----------------|
| 60 seconds   | 1              | ~4 minutes      | 1800 frames     |
| 60 seconds   | 3              | ~1.5 minutes    | 600 frames      |
| 60 seconds   | 6              | ~45 seconds     | 300 frames      |

### Memory Requirements

- **Typical video**: ~500 MB RAM
- **Large video (4K)**: ~2 GB RAM
- **Model loading**: ~200 MB RAM

---

## 🐛 Troubleshooting

### Common Issues

#### 1. "Cannot open video" error

```bash
# Check video file exists
ls -lh your_video.mp4

# Test with OpenCV
python -c "import cv2; cap = cv2.VideoCapture('your_video.mp4'); print('OK' if cap.isOpened() else 'FAILED')"
```

**Solution**: Ensure video codec is supported. Convert to MP4 if needed:
```bash
ffmpeg -i input.mov -c:v libx264 output.mp4
```

#### 2. Low detection rate

**Symptoms**: Few poses detected (< 50% of frames)

**Solutions**:
- Ensure subject is clearly visible
- Check lighting conditions
- Reduce `min_detection_confidence` in code (line 25):
  ```python
  min_detection_confidence=0.3  # Lower threshold
  ```

#### 3. Slow processing

**Solutions**:
- Increase `--fps-decimation` value
- Reduce video resolution before processing:
  ```bash
  ffmpeg -i input.mp4 -vf scale=640:-1 input_small.mp4
  ```
- Use `model_complexity=1` or `0` in code (line 24)

#### 4. Missing joint angles

**Symptoms**: Some angles not calculated

**Cause**: Required keypoints not detected (low visibility)

**Solution**: Check `visibility` values in JSON output. Values < 0.5 indicate poor detection.

#### 5. Installation issues

```bash
# If mediapipe installation fails
pip install --upgrade pip
pip install mediapipe --no-cache-dir

# If OpenCV issues
pip install opencv-python-headless  # Use headless version
```

---

## 📚 Use Cases

### Sports Performance Analysis
- **Squash/Tennis**: Analyze swing mechanics, racket angles
- **Running**: Gait analysis, knee angles, stride length
- **Weightlifting**: Form analysis, joint angles during lifts
- **Yoga/Fitness**: Pose assessment, movement quality

### Medical & Rehabilitation
- Physical therapy progress tracking
- Gait analysis for injury recovery
- Range of motion assessment
- Posture analysis

### Animation & VFX
- Motion capture reference
- Character animation reference
- Movement studies

---

## 🔮 Future Enhancements

Planned features for future releases:

- [ ] GPU acceleration support (CUDA/OpenVINO)
- [ ] Real-time webcam processing
- [ ] Multi-person tracking
- [ ] Custom joint angle definitions
- [ ] CSV export option
- [ ] Visualization dashboard
- [ ] Integration with VideoPose3D for temporal lifting
- [ ] GCN refinement block for occlusion handling

---

## 📖 References

### Based on Research Paper
This implementation follows best practices from:
*"Deployment of Lightweight, Open-Source 3D Human Pose Estimation Pipelines for Dynamic Squash Game Analysis: A Robustness-Focused Technical Assessment"*

### Key Technologies
- **MediaPipe Pose**: [Google MediaPipe Documentation](https://google.github.io/mediapipe/solutions/pose.html)
- **OpenCV**: [OpenCV Documentation](https://docs.opencv.org/)
- **NumPy**: [NumPy Documentation](https://numpy.org/doc/)

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

---

## 💬 Support

For issues and questions:
- Open an issue on GitHub
- Check [Troubleshooting](#troubleshooting) section
- Review MediaPipe documentation

---

## ✍️ Authors

Created as a practical implementation of lightweight 3D pose estimation for sports analysis.

---

## 🙏 Acknowledgments

- Google MediaPipe team for the pose detection framework
- OpenCV community for computer vision tools
- Sports biomechanics research community

---

**Happy Pose Estimation! 🎯**
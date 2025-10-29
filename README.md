3D Pose Estimation & Joint Angle Calculator

A lightweight Python tool for extracting 3D human pose skeletons and calculating joint angles from video files using Google's MediaPipe framework.

📋 Table of Contents

Features

Installation

Quick Start

Usage

⚙️ Model Selection

Output Format

Joint Angles Calculated

Technical Details

Performance Optimization

Troubleshooting

Citation

✨ Features

Real 3D Pose Detection: Extracts 33 body landmarks with world coordinates (x, y, z)

Joint Angle Calculation: Computes 9 key joint angles using 3D vector mathematics

Video Processing: Batch process entire videos with skeleton overlay

Temporal Smoothing: Built-in smoothing for stable tracking across frames

Optimized Performance: FPS decimation for faster processing without quality loss

Multiple Outputs:
  - Annotated video with skeleton visualization
  - JSON file with all pose data and angles
  - Statistical summary report

Sports Analysis Ready: Designed for biomechanical analysis of dynamic movements

🔧 Installation

Prerequisites

Python 3.7 or higher

pip package manager

Install Dependencies

# Clone or download the repository
git clone [https://github.com/EllouziMedAmin/Squash_Game_Analysis.git](https://github.com/EllouziMedAmin/Squash_Game_Analysis.git)
cd squash analysis

# Install required packages
pip install mediapipe opencv-python numpy

# Or use requirements.txt
pip install -r requirements.txt


Requirements File

Create a requirements.txt:

mediapipe>=0.10.0
opencv-python>=4.8.0
numpy>=1.21.0


🚀 Quick Start

Basic Usage

# Process a video with default settings (Heavy model, 3 FPS decimation)
python pose-estimation.py input_video.mp4


This will generate:

output_pose_input_video.mp4 - Video with skeleton overlay

pose_data_input_video.json - Detailed pose data

Example Output

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
Left Elbow          | Mean:  145.3° | Min:   89.2° | Max:  178.4° | Std:   18.7°
Right Elbow         | Mean:  142.8° | Min:   85.6° | Max:  179.1° | Std:   19.3°
Left Knee           | Mean:  168.5° | Min:  120.3° | Max:  179.8° | Std:   12.4°
...

🎬 Output video: output_pose_input_video.mp4
📄 Output JSON: pose_data_input_video.json


📖 Usage

Command Line Arguments

python pose-estimation.py <video_path> [options]


Required Arguments

video_path - Path to input video file (MP4, AVI, MOV, etc.)

Optional Arguments

Argument

Default

Description

--output-video

output_pose_<name>.mp4

Path to save annotated output video

--output-json

pose_data_<name>.json

Path to save pose data JSON

--fps-decimation

3

Process every Nth frame (higher = faster, lower = more data)

--model-complexity

2

0=Lite (Fastest), 1=Full (Balanced), 2=Heavy (Most Accurate)

Advanced Examples

Adjusting Model Complexity for Speed:

# Use the fastest model for mobile-like performance (less accurate 3D)
python pose-estimation.py video.mp4 --model-complexity 0 --fps-decimation 6

# Use the Heavy model for highest accuracy (slower processing)
python pose-estimation.py video.mp4 --model-complexity 2 --fps-decimation 1 


Custom output paths:

python pose-estimation.py athlete_match.mp4 \
    --output-video analysis/match_skeleton.mp4 \
    --output-json analysis/match_data.json \
    --model-complexity 1


Process multiple videos:

# Bash script for batch processing
for video in videos/*.mp4; do
    python pose-estimation.py "$video" --fps-decimation 3
done


⚙️ Model Selection

MediaPipe provides three models for pose estimation, trading off speed and accuracy. The complexity level is set via the --model-complexity argument.

Complexity

Model File

Speed

Accuracy

Best For

0 (Lite)

pose_landmarker_lite.task

Fastest

Lowest

Real-time mobile applications or low-power devices where speed is critical.

1 (Full)

pose_landmarker_full.task

Balanced

Good

General use, quick analysis, and moderate-speed tasks.

2 (Heavy)

pose_landmarker_heavy.task

Slowest

Highest

Biomechanical research where precise 3D keypoints and stable joint angles are required.

Recommendation: For accurate joint angle calculation, especially in 3D, the Heavy (2) model is strongly recommended to ensure stability and precision.

📊 Output Format

1. Annotated Video

The output video includes:

Skeleton overlay: 33 keypoints connected by lines

Joint angles: Real-time display of calculated angles

Same resolution: Maintains original video quality

2. JSON Data Structure

[
  {
    "frame_number": 30,
    "timestamp": 1.0,
    "joint_angles": {
      "right_elbow": 145.23,
      "left_elbow": 152.87,
      ...
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


3. Console Summary

Statistical summary printed to console:

Mean, Min, Max, Standard Deviation for each joint angle

Total frames processed

Detection success rate

🦴 Joint Angles Calculated

The tool calculates 9 biomechanically important joint angles using 3D world coordinates:

Joint Angle

Landmarks Used

Description

Right Elbow

Shoulder → Elbow → Wrist

Arm flexion/extension

Left Elbow

Shoulder → Elbow → Wrist

Arm flexion/extension

Right Shoulder

Elbow → Shoulder → Hip

Shoulder abduction/adduction

Left Shoulder

Elbow → Shoulder → Hip

Shoulder abduction/adduction

Right Knee

Hip → Knee → Ankle

Leg flexion/extension

Left Knee

Hip → Knee → Ankle

Leg flexion/extension

Right Hip

Shoulder → Hip → Knee

Hip flexion/extension

Left Hip

Shoulder → Hip → Knee

Hip flexion/extension

Torso Lean

Shoulder Mid → Hip Mid → Vertical

Body lean angle

Angle Calculation Method

Uses the 3D dot product method for accurate angle calculation:

# Vector from b to a
ba = [a.x - b.x, a.y - b.y, a.z - b.z]

# Vector from b to c
bc = [c.x - b.x, c.y - b.y, c.z - b.z]

# Calculate angle
cos(θ) = (ba · bc) / (|ba| × |bc|)
angle = arccos(cos(θ)) × 180/π


🔬 Technical Details

MediaPipe Configuration

The script uses the following configuration, controlled by the --model-complexity flag:

mp.solutions.pose.Pose(
    static_image_mode=False,        # Video mode
    model_complexity=2,             # 0=Lite, 1=Full, 2=Heavy (default: most accurate)
    smooth_landmarks=True,          # Temporal smoothing enabled for stability
    enable_segmentation=False,      # Not needed for pose only
    min_detection_confidence=0.5,   # Detection threshold
    min_tracking_confidence=0.5     # Tracking threshold
)


Keypoint Structure

MediaPipe provides 33 landmarks per frame:

0: nose                    17: left_pinky
1: left_eye_inner         18: right_pinky
2: left_eye               19: left_index
...
11: left_shoulder         24: right_hip
12: right_shoulder        25: left_knee
...
32: right_foot_index


Coordinate System

2D Landmarks: Normalized [0, 1] relative to frame dimensions

3D World Landmarks: Real-world coordinates in meters   - x: Horizontal (left/right)   - y: Vertical (up/down)   - z: Depth (forward/backward, relative to hip center)

⚡ Performance Optimization

Performance is a combination of the Model Complexity and FPS Decimation.

Recommended Settings for Different Use Cases

Use Case

Model Complexity

FPS Decimation

Priority

Real-time / Mobile Speed

0 (Lite)

6 or higher

Maximum speed

Biomechanical Analysis

2 (Heavy)

1 to 3

Maximum accuracy

General Analysis

1 (Full)

3

Balanced speed and quality

Example: Biomechanical Analysis (Balanced)

python pose-estimation.py video.mp4 --model-complexity 2 --fps-decimation 3
# ✅ Recommended for sports analysis (good accuracy, reasonable speed)


Performance Benchmarks

Tested on Intel i7 CPU (no GPU), assuming a 30 FPS input video:

Model

FPS Decimation

Frames Analyzed (60s video)

Est. Processing Time

Heavy (2)

1

1800

~4 minutes

Heavy (2)

3

600

~1.5 minutes

Lite (0)

6

300

~25 seconds

Lite (0)

10

180

~15 seconds

Memory Requirements

Typical video: ~500 MB RAM

Large video (4K): ~2 GB RAM

Model loading: ~200 MB RAM (smaller for Lite model)

🐛 Troubleshooting

Common Issues

1. "Cannot open video" error

# Check video file exists
ls -lh your_video.mp4


Solution: Ensure video codec is supported. Convert to MP4 if needed:

ffmpeg -i input.mov -c:v libx264 output.mp4


2. Low detection rate

Symptoms: Few poses detected (< 50% of frames)

Solutions:

Ensure subject is clearly visible

Check lighting conditions

Reduce min_detection_confidence in code (line 25):
  python   min_detection_confidence=0.3  # Lower threshold   

3. Jittery or Inaccurate Angles

Cause: The Lite model (complexity 0) often produces less stable 3D world coordinates, causing angles to jump.

Solution:

Increase Model Complexity: Use --model-complexity 1 (Full) or --model-complexity 2 (Heavy).

4. Slow processing

Solutions:

Increase --fps-decimation value.

Decrease Model Complexity: Use --model-complexity 1 or --model-complexity 0.

Reduce video resolution before processing:
  bash   ffmpeg -i input.mp4 -vf scale=640:-1 input_small.mp4   

5. Missing joint angles

Symptoms: Some angles not calculated

Cause: Required keypoints not detected (low visibility)

Solution: Check visibility values in JSON output. Values < 0.5 indicate poor detection.

📚 Use Cases

Sports Performance Analysis

Squash/Tennis: Analyze swing mechanics, racket angles

Running: Gait analysis, knee angles, stride length

Weightlifting: Form analysis, joint angles during lifts

Yoga/Fitness: Pose assessment, movement quality

Medical & Rehabilitation

Physical therapy progress tracking

Gait analysis for injury recovery

Range of motion assessment

Posture analysis

Animation & VFX

Motion capture reference

Character animation reference

Movement studies

🔮 Future Enhancements

Planned features for future releases:

[ ] GPU acceleration support (CUDA/OpenVINO)

[ ] Real-time webcam processing

[ ] Multi-person tracking

[ ] Custom joint angle definitions

[ ] CSV export option

[ ] Visualization dashboard

[ ] Integration with VideoPose3D for temporal lifting

[ ] GCN refinement block for occlusion handling

📖 References

Based on Research Paper

This implementation follows best practices from:
"Deployment of Lightweight, Open-Source 3D Human Pose Estimation Pipelines for Dynamic Squash Game Analysis: A Robustness-Focused Technical Assessment"

Key Technologies

MediaPipe Pose: Google MediaPipe Documentation

OpenCV: OpenCV Documentation

NumPy: NumPy Documentation

📄 License

This project is provided as-is for educational and research purposes.

🤝 Contributing

Contributions welcome! Please:

Fork the repository

Create a feature branch

Submit a pull request with detailed description

💬 Support

For issues and questions:

Open an issue on GitHub

Check Troubleshooting section

Review MediaPipe documentation

✍️ Authors

Created as a practical implementation of lightweight 3D pose estimation for sports analysis.

🙏 Acknowledgments

Google MediaPipe team for the pose detection framework

OpenCV community for computer vision tools

Sports biomechanics research community

Happy Pose Estimation! 🎯
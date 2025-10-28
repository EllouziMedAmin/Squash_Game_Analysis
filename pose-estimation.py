"""
3D Pose Estimation with MediaPipe
Extracts 3D skeleton and calculates joint angles from video input
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
import argparse


class PoseEstimator3D:
    def __init__(self):
        """Initialize MediaPipe Pose detector"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector with 3D world landmarks
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # 0=Lite, 1=Full, 2=Heavy (most accurate)
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.results_data = []
        
    def calculate_angle_3d(self, a, b, c):
        """
        Calculate 3D angle at point b given three 3D points a, b, c
        Uses dot product method for accurate 3D angle calculation
        
        Args:
            a, b, c: Points with x, y, z coordinates
        Returns:
            Angle in degrees
        """
        # Create vectors
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def calculate_joint_angles(self, landmarks):
        """
        Calculate key joint angles from 3D world landmarks
        
        MediaPipe Pose Landmark indices:
        0: nose, 11: left_shoulder, 12: right_shoulder
        13: left_elbow, 14: right_elbow, 15: left_wrist, 16: right_wrist
        23: left_hip, 24: right_hip, 25: left_knee, 26: right_knee
        27: left_ankle, 28: right_ankle
        """
        angles = {}
        
        try:
            # Right elbow angle (shoulder -> elbow -> wrist)
            if all(landmarks[i] for i in [12, 14, 16]):
                angles['right_elbow'] = self.calculate_angle_3d(
                    landmarks[12], landmarks[14], landmarks[16]
                )
            
            # Left elbow angle
            if all(landmarks[i] for i in [11, 13, 15]):
                angles['left_elbow'] = self.calculate_angle_3d(
                    landmarks[11], landmarks[13], landmarks[15]
                )
            
            # Right shoulder angle (elbow -> shoulder -> hip)
            if all(landmarks[i] for i in [14, 12, 24]):
                angles['right_shoulder'] = self.calculate_angle_3d(
                    landmarks[14], landmarks[12], landmarks[24]
                )
            
            # Left shoulder angle
            if all(landmarks[i] for i in [13, 11, 23]):
                angles['left_shoulder'] = self.calculate_angle_3d(
                    landmarks[13], landmarks[11], landmarks[23]
                )
            
            # Right knee angle (hip -> knee -> ankle)
            if all(landmarks[i] for i in [24, 26, 28]):
                angles['right_knee'] = self.calculate_angle_3d(
                    landmarks[24], landmarks[26], landmarks[28]
                )
            
            # Left knee angle
            if all(landmarks[i] for i in [23, 25, 27]):
                angles['left_knee'] = self.calculate_angle_3d(
                    landmarks[23], landmarks[25], landmarks[27]
                )
            
            # Right hip angle (shoulder -> hip -> knee)
            if all(landmarks[i] for i in [12, 24, 26]):
                angles['right_hip'] = self.calculate_angle_3d(
                    landmarks[12], landmarks[24], landmarks[26]
                )
            
            # Left hip angle
            if all(landmarks[i] for i in [11, 23, 25]):
                angles['left_hip'] = self.calculate_angle_3d(
                    landmarks[11], landmarks[23], landmarks[25]
                )
            
            # Torso angle (shoulder midpoint -> hip midpoint -> vertical)
            if all(landmarks[i] for i in [11, 12, 23, 24]):
                # Calculate midpoints
                shoulder_mid_x = (landmarks[11].x + landmarks[12].x) / 2
                shoulder_mid_y = (landmarks[11].y + landmarks[12].y) / 2
                shoulder_mid_z = (landmarks[11].z + landmarks[12].z) / 2
                
                hip_mid_x = (landmarks[23].x + landmarks[24].x) / 2
                hip_mid_y = (landmarks[23].y + landmarks[24].y) / 2
                hip_mid_z = (landmarks[23].z + landmarks[24].z) / 2
                
                # Create mock points for angle calculation
                class Point:
                    def __init__(self, x, y, z):
                        self.x, self.y, self.z = x, y, z
                
                shoulder_point = Point(shoulder_mid_x, shoulder_mid_y, shoulder_mid_z)
                hip_point = Point(hip_mid_x, hip_mid_y, hip_mid_z)
                vertical_point = Point(hip_mid_x, hip_mid_y + 1, hip_mid_z)
                
                angles['torso_lean'] = self.calculate_angle_3d(
                    shoulder_point, hip_point, vertical_point
                )
        
        except Exception as e:
            print(f"Error calculating angles: {e}")
        
        return angles
    
    def draw_pose_on_frame(self, frame, results):
        """Draw pose landmarks and connections on frame"""
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def process_video(self, video_path, output_path=None, fps_decimation=3):
        """
        Process video and extract 3D pose with joint angles
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            fps_decimation: Process every Nth frame (default 3 = ~10 FPS for 30fps video)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Video Properties:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total Frames: {total_frames}")
        print(f"   Processing every {fps_decimation} frames (~{fps//fps_decimation} FPS)")
        
        # Setup video writer if output path provided
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps//fps_decimation, (width, height))
        
        frame_count = 0
        processed_count = 0
        self.results_data = []
        
        print("\n🔄 Processing video...")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            # Process only every Nth frame (temporal decimation)
            if frame_count % fps_decimation != 0:
                continue
            
            # Convert BGR to RGB for MediaPipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process pose detection
            results = self.pose.process(image)
            
            # Convert back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract data if pose detected
            if results.pose_world_landmarks:
                # Calculate joint angles from 3D world landmarks
                angles = self.calculate_joint_angles(results.pose_world_landmarks.landmark)
                
                # Store frame data
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'joint_angles': angles,
                    'landmarks_3d': []
                }
                
                # Store 3D world coordinates
                for idx, landmark in enumerate(results.pose_world_landmarks.landmark):
                    frame_data['landmarks_3d'].append({
                        'id': idx,
                        'name': self.mp_pose.PoseLandmark(idx).name,
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                self.results_data.append(frame_data)
                
                # Draw pose on frame
                image = self.draw_pose_on_frame(image, results)
                
                # Draw angles on frame
                y_offset = 30
                for joint_name, angle in angles.items():
                    text = f"{joint_name.replace('_', ' ').title()}: {angle:.1f}°"
                    cv2.putText(image, text, (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 128), 2)
                    y_offset += 30
                
                processed_count += 1
            
            # Write frame to output video
            if out:
                out.write(image)
            
            # Display progress
            if processed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({processed_count} frames processed)", end='\r')
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        
        print(f"\n✅ Processing complete!")
        print(f"   Total frames processed: {processed_count}")
        print(f"   Poses detected: {len(self.results_data)}")
        
        return self.results_data
    
    def save_results(self, output_json_path):
        """Save pose data and joint angles to JSON"""
        with open(output_json_path, 'w') as f:
            json.dump(self.results_data, f, indent=2)
        print(f"\n💾 Results saved to: {output_json_path}")
    
    def print_summary(self):
        """Print summary statistics of joint angles"""
        if not self.results_data:
            print("No data to summarize")
            return
        
        print("\n📊 Joint Angle Summary (degrees):")
        print("-" * 60)
        
        # Collect all angles
        all_angles = {}
        for frame in self.results_data:
            for joint, angle in frame['joint_angles'].items():
                if joint not in all_angles:
                    all_angles[joint] = []
                all_angles[joint].append(angle)
        
        # Print statistics
        for joint, angles in sorted(all_angles.items()):
            angles_array = np.array(angles)
            print(f"{joint.replace('_', ' ').title():20s} | "
                  f"Mean: {np.mean(angles_array):6.1f}° | "
                  f"Min: {np.min(angles_array):6.1f}° | "
                  f"Max: {np.max(angles_array):6.1f}° | "
                  f"Std: {np.std(angles_array):6.1f}°")
    
    def cleanup(self):
        """Release resources"""
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(
        description='3D Pose Estimation and Joint Angle Calculator'
    )
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output-video', type=str, help='Path to save output video with pose overlay')
    parser.add_argument('--output-json', type=str, help='Path to save pose data JSON')
    parser.add_argument('--fps-decimation', type=int, default=3, 
                       help='Process every Nth frame (default: 3)')
    
    args = parser.parse_args()
    
    # Validate input
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    # Set default output paths
    output_video = args.output_video or f"output_pose_{video_path.stem}.mp4"
    output_json = args.output_json or f"pose_data_{video_path.stem}.json"
    
    print("=" * 60)
    print("🎯 3D Pose Estimation with MediaPipe")
    print("=" * 60)
    
    # Initialize and process
    estimator = PoseEstimator3D()
    
    try:
        # Process video
        results = estimator.process_video(
            video_path,
            output_path=output_video,
            fps_decimation=args.fps_decimation
        )
        
        # Save results
        estimator.save_results(output_json)
        
        # Print summary
        estimator.print_summary()
        
        print(f"\n🎬 Output video: {output_video}")
        print(f"📄 Output JSON: {output_json}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        estimator.cleanup()


if __name__ == "__main__":
    main()
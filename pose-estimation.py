"""
Enhanced 3D Pose Estimation with MediaPipe
Extracts 3D skeleton and calculates joint angles using rotation matrices
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
import argparse
from scipy.signal import medfilt


class RotationUtils:
    """Utility functions for rotation matrix operations"""
    
    @staticmethod
    def get_R_z(theta):
        """Rotation matrix around Z axis"""
        return np.array([[np.cos(theta), -np.sin(theta), 0],
                        [np.sin(theta), np.cos(theta), 0],
                        [0, 0, 1]])
    
    @staticmethod
    def get_R_x(theta):
        """Rotation matrix around X axis"""
        return np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])
    
    @staticmethod
    def get_R_y(theta):
        """Rotation matrix around Y axis"""
        return np.array([[np.cos(theta), 0, np.sin(theta)],
                        [0, 1, 0],
                        [-np.sin(theta), 0, np.cos(theta)]])
    
    @staticmethod
    def decompose_R_ZXY(R):
        """Decompose rotation matrix into ZXY Euler angles"""
        thetax = np.arcsin(R[2, 1])
        thetaz = np.arctan2(-R[0, 1], R[1, 1])
        thetay = np.arctan2(-R[2, 0], R[2, 2])
        return thetaz, thetay, thetax
    
    @staticmethod
    def get_R2(u, v):
        """Calculate rotation matrix that rotates unit vector u to v"""
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)
        
        w = np.cross(u, v)
        w_norm = np.linalg.norm(w)
        
        if w_norm < 1e-6:
            return np.eye(3)
        
        w = w / w_norm
        theta = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
        
        K = np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]])
        
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R


class PoseEstimator3D:
    def __init__(self):
        """Initialize MediaPipe Pose detector"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.utils = RotationUtils()
        self.results_data = []
        self.skeleton_data = None
        
        # MediaPipe to standard joint mapping
        self.mp_to_standard = {
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
        }
        
        # Define skeleton hierarchy (child: [parent, grandparent, ...])
        self.hierarchy = {
            'hips': [],
            'left_hip': ['hips'],
            'left_knee': ['left_hip', 'hips'],
            'left_ankle': ['left_knee', 'left_hip', 'hips'],
            'right_hip': ['hips'],
            'right_knee': ['right_hip', 'hips'],
            'right_ankle': ['right_knee', 'right_hip', 'hips'],
            'neck': ['hips'],
            'left_shoulder': ['neck', 'hips'],
            'left_elbow': ['left_shoulder', 'neck', 'hips'],
            'left_wrist': ['left_elbow', 'left_shoulder', 'neck', 'hips'],
            'right_shoulder': ['neck', 'hips'],
            'right_elbow': ['right_shoulder', 'neck', 'hips'],
            'right_wrist': ['right_elbow', 'right_shoulder', 'neck', 'hips']
        }
        
        self.joints = list(self.hierarchy.keys())
    
    def extract_keypoints(self, landmarks):
        """Extract keypoints from MediaPipe landmarks"""
        kpts = {}
        
        for joint, idx in self.mp_to_standard.items():
            lm = landmarks[idx]
            kpts[joint] = np.array([lm.x, lm.y, lm.z])
        
        # Add computed joints
        kpts['hips'] = (kpts['left_hip'] + kpts['right_hip']) / 2
        kpts['neck'] = (kpts['left_shoulder'] + kpts['right_shoulder']) / 2
        
        return kpts
    
    def get_bone_lengths(self, all_keypoints):
        """Calculate average bone lengths across all frames"""
        bone_lengths = {joint: [] for joint in self.joints if joint != 'hips'}
        
        for frame_kpts in all_keypoints:
            for joint in self.joints:
                if joint == 'hips':
                    continue
                parent = self.hierarchy[joint][0]
                bone_vec = frame_kpts[joint] - frame_kpts[parent]
                length = np.linalg.norm(bone_vec)
                bone_lengths[joint].append(length)
        
        # Take median length for each bone
        avg_bone_lengths = {}
        for joint, lengths in bone_lengths.items():
            avg_bone_lengths[joint] = np.median(lengths)
        
        return avg_bone_lengths
    
    def get_base_skeleton(self, bone_lengths, normalization_bone='neck'):
        """Define T-pose skeleton with normalized bone lengths"""
        normalization = bone_lengths[normalization_bone]
        
        # Define offset directions for T-pose
        offset_directions = {
            'left_hip': np.array([1, 0, 0]),
            'left_knee': np.array([0, -1, 0]),
            'left_ankle': np.array([0, -1, 0]),
            'right_hip': np.array([-1, 0, 0]),
            'right_knee': np.array([0, -1, 0]),
            'right_ankle': np.array([0, -1, 0]),
            'neck': np.array([0, 1, 0]),
            'left_shoulder': np.array([1, 0, 0]),
            'left_elbow': np.array([1, 0, 0]),
            'left_wrist': np.array([1, 0, 0]),
            'right_shoulder': np.array([-1, 0, 0]),
            'right_elbow': np.array([-1, 0, 0]),
            'right_wrist': np.array([-1, 0, 0])
        }
        
        base_skeleton = {'hips': np.array([0, 0, 0])}
        
        # Average symmetric limbs
        def set_length(joint_type):
            left_len = bone_lengths[f'left_{joint_type}']
            right_len = bone_lengths[f'right_{joint_type}']
            avg_len = (left_len + right_len) / (2 * normalization)
            
            base_skeleton[f'left_{joint_type}'] = offset_directions[f'left_{joint_type}'] * avg_len
            base_skeleton[f'right_{joint_type}'] = offset_directions[f'right_{joint_type}'] * avg_len
        
        set_length('hip')
        set_length('knee')
        set_length('ankle')
        set_length('shoulder')
        set_length('elbow')
        set_length('wrist')
        
        base_skeleton['neck'] = offset_directions['neck'] * (bone_lengths['neck'] / normalization)
        
        return base_skeleton, offset_directions, normalization
    
    def get_hips_position_and_rotation(self, frame_kpts):
        """Calculate root (hips) position and rotation"""
        root_pos = frame_kpts['hips']
        
        # Calculate root coordinate system
        root_u = frame_kpts['left_hip'] - frame_kpts['hips']
        root_u = root_u / np.linalg.norm(root_u)
        
        root_v = frame_kpts['neck'] - frame_kpts['hips']
        root_v = root_v / np.linalg.norm(root_v)
        
        root_w = np.cross(root_u, root_v)
        root_w = root_w / np.linalg.norm(root_w)
        
        # Rotation matrix
        C = np.column_stack([root_u, root_v, root_w])
        tz, ty, tx = self.utils.decompose_R_ZXY(C)
        root_rotation = np.array([tz, tx, ty])
        
        return root_pos, root_rotation
    
    def get_rotation_chain(self, joint, hierarchy, frame_rotations):
        """Compose chain of rotation matrices"""
        hierarchy = hierarchy[::-1]
        R = np.eye(3)
        
        for parent in hierarchy:
            angles = frame_rotations[parent]
            _R = (self.utils.get_R_z(angles[0]) @ 
                  self.utils.get_R_x(angles[1]) @ 
                  self.utils.get_R_y(angles[2]))
            R = R @ _R
        
        return R
    
    def get_joint_rotation(self, joint_name, frame_kpts, frame_rotations, offset_directions):
        """Calculate rotation for a specific joint"""
        hierarchy = self.hierarchy[joint_name]
        
        # Calculate inverse rotation from parent chain
        invR = np.eye(3)
        for i, parent in enumerate(hierarchy):
            if i == 0:
                continue
            angles = frame_rotations[parent]
            R = (self.utils.get_R_z(angles[0]) @ 
                 self.utils.get_R_x(angles[1]) @ 
                 self.utils.get_R_y(angles[2]))
            invR = invR @ R.T
        
        # Calculate bone vector in local space
        parent = hierarchy[0]
        b = invR @ (frame_kpts[joint_name] - frame_kpts[parent])
        
        # Calculate rotation to align offset direction with bone vector
        R = self.utils.get_R2(offset_directions[joint_name], b)
        tz, ty, tx = self.utils.decompose_R_ZXY(R)
        
        return np.array([tz, tx, ty])
    
    def calculate_joint_angles(self, frame_kpts, offset_directions):
        """Calculate all joint angles for a single frame"""
        # Get root rotation
        root_pos, root_rotation = self.get_hips_position_and_rotation(frame_kpts)
        frame_rotations = {'hips': root_rotation}
        
        # Center pose at hips
        centered_kpts = {}
        for joint in self.joints:
            centered_kpts[joint] = frame_kpts[joint] - root_pos
        
        # Calculate joint rotations by hierarchy depth
        max_depth = max(len(self.hierarchy[j]) for j in self.joints)
        
        for depth in range(2, max_depth + 1):
            for joint in self.joints:
                if len(self.hierarchy[joint]) == depth:
                    joint_rotation = self.get_joint_rotation(
                        joint, centered_kpts, frame_rotations, offset_directions
                    )
                    parent = self.hierarchy[joint][0]
                    frame_rotations[parent] = joint_rotation
        
        # Add zero rotation for endpoints
        for joint in self.joints:
            if joint not in frame_rotations:
                frame_rotations[joint] = np.array([0., 0., 0.])
        
        return frame_rotations, root_pos
    
    def angles_to_degrees(self, angles_dict):
        """Convert angles from radians to degrees"""
        return {k: np.degrees(v) for k, v in angles_dict.items()}
    
    def draw_pose_on_frame(self, frame, results):
        """Draw pose landmarks on frame"""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame
    
    def process_video(self, video_path, output_path=None, fps_decimation=3):
        """Process video and extract 3D pose with joint angles"""
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
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps//fps_decimation, (width, height))
        
        frame_count = 0
        processed_count = 0
        all_keypoints = []
        
        print("\n🔄 Phase 1: Extracting keypoints...")
        
        # First pass: extract all keypoints
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            if frame_count % fps_decimation != 0:
                continue
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            
            if results.pose_world_landmarks:
                kpts = self.extract_keypoints(results.pose_world_landmarks.landmark)
                all_keypoints.append(kpts)
                processed_count += 1
            
            if processed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({processed_count} frames)", end='\r')
        
        print(f"\n   Extracted {len(all_keypoints)} frames with keypoints")
        
        if len(all_keypoints) == 0:
            print("❌ No pose detected in video")
            cap.release()
            return []
        
        # Calculate skeleton parameters
        print("\n🔄 Phase 2: Calculating skeleton structure...")
        bone_lengths = self.get_bone_lengths(all_keypoints)
        base_skeleton, offset_directions, normalization = self.get_base_skeleton(bone_lengths)
        
        self.skeleton_data = {
            'bone_lengths': bone_lengths,
            'base_skeleton': base_skeleton,
            'offset_directions': offset_directions,
            'normalization': normalization,
            'hierarchy': self.hierarchy
        }
        
        # Second pass: calculate angles and create output video
        print("\n🔄 Phase 3: Calculating joint angles and creating output...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        processed_count = 0
        self.results_data = []
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            
            if frame_count % fps_decimation != 0:
                continue
            
            if processed_count >= len(all_keypoints):
                break
            
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_world_landmarks:
                kpts = all_keypoints[processed_count]
                
                # Calculate joint angles
                joint_rotations, root_pos = self.calculate_joint_angles(kpts, offset_directions)
                joint_angles_deg = self.angles_to_degrees(joint_rotations)
                
                # Store frame data
                frame_data = {
                    'frame_number': frame_count,
                    'timestamp': frame_count / fps,
                    'root_position': root_pos.tolist(),
                    'joint_angles_rad': {k: v.tolist() for k, v in joint_rotations.items()},
                    'joint_angles_deg': {k: v.tolist() for k, v in joint_angles_deg.items()},
                    'keypoints_3d': {k: v.tolist() for k, v in kpts.items()}
                }
                
                self.results_data.append(frame_data)
                
                # Draw pose
                image = self.draw_pose_on_frame(image, results)
                
                # Draw key angles
                y_offset = 30
                key_joints = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
                for joint in key_joints:
                    if joint in joint_angles_deg:
                        angles = joint_angles_deg[joint]
                        text = f"{joint.replace('_', ' ').title()}: [{angles[0]:.1f}°, {angles[1]:.1f}°, {angles[2]:.1f}°]"
                        cv2.putText(image, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
                        y_offset += 25
                
                processed_count += 1
            
            if out:
                out.write(image)
            
            if processed_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({processed_count} frames)", end='\r')
        
        cap.release()
        if out:
            out.release()
        
        print(f"\n✅ Processing complete!")
        print(f"   Poses with angles: {len(self.results_data)}")
        
        return self.results_data
    
    def save_results(self, output_json_path):
        """Save pose data and joint angles to JSON"""
        output_data = {
            'skeleton_parameters': {
                'bone_lengths': {k: float(v) for k, v in self.skeleton_data['bone_lengths'].items()},
                'normalization': float(self.skeleton_data['normalization']),
                'hierarchy': self.skeleton_data['hierarchy']
            },
            'frames': self.results_data
        }
        
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {output_json_path}")
    
    def print_summary(self):
        """Print summary statistics of joint angles"""
        if not self.results_data:
            print("No data to summarize")
            return
        
        print("\n📊 Joint Angle Summary (ZXY Euler angles in degrees):")
        print("-" * 80)
        
        # Collect all angles
        all_angles = {joint: {'z': [], 'x': [], 'y': []} for joint in self.joints}
        
        for frame in self.results_data:
            for joint, angles in frame['joint_angles_deg'].items():
                all_angles[joint]['z'].append(angles[0])
                all_angles[joint]['x'].append(angles[1])
                all_angles[joint]['y'].append(angles[2])
        
        # Print statistics
        for joint in self.joints:
            if len(all_angles[joint]['z']) == 0:
                continue
            print(f"\n{joint.replace('_', ' ').title()}:")
            for i, axis in enumerate(['Z', 'X', 'Y']):
                vals = np.array(all_angles[joint][axis.lower()])
                print(f"  {axis}-axis: Mean={np.mean(vals):7.1f}° | "
                      f"Min={np.min(vals):7.1f}° | "
                      f"Max={np.max(vals):7.1f}° | "
                      f"Std={np.std(vals):6.1f}°")
    
    def cleanup(self):
        """Release resources"""
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced 3D Pose Estimation with Biomechanical Joint Angles'
    )
    parser.add_argument('video_path', type=str, help='Path to input video file')
    parser.add_argument('--output-video', type=str, help='Path to save output video')
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
    output_json = args.output_json or f"pose_angles_{video_path.stem}.json"
    
    print("=" * 80)
    print("🎯 Enhanced 3D Pose Estimation with Biomechanical Joint Angles")
    print("=" * 80)
    
    # Initialize and process
    estimator = PoseEstimator3D()
    
    try:
        # Process video
        results = estimator.process_video(
            video_path,
            output_path=output_video,
            fps_decimation=args.fps_decimation
        )
        
        if results:
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
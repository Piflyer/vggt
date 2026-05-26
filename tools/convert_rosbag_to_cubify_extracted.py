#!/usr/bin/env python3
import argparse
import json
import os
from collections import defaultdict, deque

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

# Motion Alignment Matrix: Align Camera Forward (Z) with Robot Motion Axis
# Based on Sparkal robot's odometry behavior
R_MOTION_ALIGNED = np.array([
    [ 1,  0,  0, 0],
    [ 0,  0, -1, 0],
    [ 0,  1,  0, 0],
    [ 0,  0,  0, 1]
], dtype=np.float32)

def quaternion_to_matrix(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < np.finfo(float).eps:
        return np.eye(4, dtype=np.float32)
    s = 2.0 / n
    xx = x * x * s
    yy = y * y * s
    zz = z * z * s
    xy = x * y * s
    xz = x * z * s
    yz = y * z * s
    wx = w * x * s
    wy = w * y * s
    wz = w * z * s
    mat = np.eye(4, dtype=np.float32)
    mat[0, 0] = 1.0 - (yy + zz)
    mat[0, 1] = xy - wz
    mat[0, 2] = xz + wy
    mat[1, 0] = xy + wz
    mat[1, 1] = 1.0 - (xx + zz)
    mat[1, 2] = yz - wx
    mat[2, 0] = xz - wy
    mat[2, 1] = yz + wx
    mat[2, 2] = 1.0 - (xx + yy)
    return mat


def pose_to_matrix(position, orientation):
    mat = quaternion_to_matrix((orientation.x, orientation.y, orientation.z, orientation.w))
    mat[0:3, 3] = [position.x, position.y, position.z]
    return mat


def build_static_transform_graph(tf_static_msgs):
    graph = {}
    for msg in tf_static_msgs:
        for tr in msg.transforms:
            parent = tr.header.frame_id
            child = tr.child_frame_id
            mat = pose_to_matrix(tr.transform.translation, tr.transform.rotation)
            graph[(parent, child)] = mat
    return graph


def invert_transform(mat):
    inv = np.eye(4, dtype=np.float32)
    R = mat[0:3, 0:3]
    t = mat[0:3, 3]
    inv[0:3, 0:3] = R.T
    inv[0:3, 3] = -R.T @ t
    return inv


def find_transform(graph, source, target):
    if source == target:
        return np.eye(4, dtype=np.float32)

    queue = deque([(source, np.eye(4, dtype=np.float32))])
    visited = {source}

    while queue:
        frame, mat = queue.popleft()
        if frame == target:
            return mat

        for (parent, child), edge_mat in graph.items():
            if parent == frame and child not in visited:
                visited.add(child)
                queue.append((child, mat @ edge_mat))
            elif child == frame and parent not in visited:
                visited.add(parent)
                queue.append((parent, mat @ invert_transform(edge_mat)))

    raise ValueError(f"Unable to find transform from {source} to {target}")


def parse_camera_info(msg):
    K = np.array(msg.K, dtype=np.float32).reshape(3, 3)
    return K, (msg.width, msg.height)


def save_json_matrix(path, mat):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mat.reshape(-1).tolist(), f)


def save_size_file(path, size_tuple):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"[{size_tuple[0]}, {size_tuple[1]}]\n")


def save_image_file(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(path, img):
        raise RuntimeError(f"Failed to write image to {path}")


def save_depth_file(path, depth):
    if depth.dtype != np.uint16 and depth.dtype != np.float32 and depth.dtype != np.float64:
        depth = depth.astype(np.uint16)
    if not cv2.imwrite(path, depth):
        raise RuntimeError(f"Failed to write depth image to {path}")


def find_nearest_index(timestamps, query_ts):
    idx = np.searchsorted(timestamps, query_ts)
    if idx == 0:
        return 0
    if idx >= len(timestamps):
        return len(timestamps) - 1
    left = timestamps[idx - 1]
    right = timestamps[idx]
    if abs(query_ts - left) <= abs(right - query_ts):
        return idx - 1
    return idx


def main():
    parser = argparse.ArgumentParser(description="Convert a ROS bag to CubifyAnything extracted dataset format.")
    parser.add_argument("--bag", type=str, default="kimera_test_2022-06-24-15-48-36.bag", help="Path to the ROS bag file.")
    parser.add_argument("--output-root", type=str, default="ml-cubifyanything/data/extracted/rosbag", help="Root folder to write extracted dataset.")
    parser.add_argument("--video-id", type=int, default=0, help="Numeric video_id folder name to use.")
    parser.add_argument("--downsample", type=int, default=1, help="Save every Nth color frame.")
    parser.add_argument("--min_disparity", type=float, default=10.0, help="Minimum disparity to generate a new keyframe (set to 0 to disable keyframe selection).")
    parser.add_argument("--start_frame", type=int, default=0, help="Start processing from this frame index (0-based).")
    parser.add_argument("--end_frame", type=int, default=-1, help="End processing at this frame index (0-based, -1 for all).")
    parser.add_argument("--allow-no-depth", action="store_true", help="Allow saving frames even when depth data is missing or unavailable.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to save (0 = all).")
    parser.add_argument("--depth-topic", type=str, default="/sparkal1/forward/depth/image_rect_raw", help="ROS topic for depth frames.")
    parser.add_argument("--depth-info-topic", type=str, default="/sparkal1/forward/depth/camera_info", help="ROS topic for depth camera info.")
    parser.add_argument("--color-topic", type=str, default="/sparkal1/forward/color/image_raw/compressed", help="ROS topic for color image frames.")
    parser.add_argument("--color-info-topic", type=str, default="/sparkal1/forward/color/camera_info", help="ROS topic for color camera info.")
    parser.add_argument("--pose-topic", type=str, default="/sparkal1/kimera_vio_ros/odometry", help="ROS topic for odometry poses.")
    parser.add_argument("--pose-frame", type=str, default="sparkal1/odom", help="Parent frame of pose messages.")
    parser.add_argument("--camera-frame", type=str, default="sparkal1/forward_color_optical_frame", help="Target camera optical frame for color images.")
    parser.add_argument("--depth-frame", type=str, default="sparkal1/forward_depth_optical_frame", help="Target camera optical frame for depth images.")
    parser.add_argument("--base-frame", type=str, default="sparkal1/realsense_base", help="Base frame described by the odometry pose child frame.")
    parser.add_argument("--rescale-mm-to-m", action="store_true", help="Rescale translations from mm to meters.")
    parser.add_argument("--no-lidar", action="store_true", help="Ignore LiDAR/depth data and save zero depth maps instead.")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    video_root = os.path.join(args.output_root, str(args.video_id))
    os.makedirs(video_root, exist_ok=True)

    bridge = CvBridge()
    print(f"Opening bag: {args.bag}")

    bag = rosbag.Bag(args.bag, "r")

    # Load intrinsics.
    color_info = None
    depth_info = None
    for _, msg, _ in bag.read_messages(topics=[args.color_info_topic]):
        color_info = msg
        break
    for _, msg, _ in bag.read_messages(topics=[args.depth_info_topic]):
        depth_info = msg
        break
    if color_info is None:
        raise RuntimeError("Could not read color camera info messages from bag.")
    if depth_info is None:
        if args.allow_no_depth:
            print("Warning: No depth camera info found; saving frames without depth.")
            depth_info = color_info
        else:
            raise RuntimeError("Could not read depth camera info messages from bag.")

    color_K, color_size = parse_camera_info(color_info)
    depth_K, depth_size = parse_camera_info(depth_info)

    print(f"Color size {color_size}, depth size {depth_size}")

    # Load depth images into memory for timestamp matching.
    depth_messages = []
    depth_ts_list = []
    if not args.no_lidar:
        print("Reading depth messages...")
        for _, msg, t in bag.read_messages(topics=[args.depth_topic]):
            timestamp_ns = msg.header.stamp.to_nsec()
            depth_cv = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_messages.append((timestamp_ns, depth_cv))
            depth_ts_list.append(timestamp_ns)
    else:
        print("Ignoring LiDAR/depth data as requested.")
        
    depth_timestamps = np.array(depth_ts_list, dtype=np.int64)
    if not args.no_lidar and depth_timestamps.size == 0 and not args.allow_no_depth:
        raise RuntimeError("No depth messages found on the depth topic; use --allow-no-depth or --no-lidar to save frames anyway.")

    # Load pose messages into memory.
    pose_timestamps = []
    pose_matrices = []
    print("Reading pose messages...")
    for _, msg, _ in bag.read_messages(topics=[args.pose_topic]):
        if msg.header.frame_id != args.pose_frame or msg.child_frame_id != args.base_frame:
            continue
        pose_timestamps.append(msg.header.stamp.to_nsec())
        pose_matrices.append(pose_to_matrix(msg.pose.pose.position, msg.pose.pose.orientation))
    pose_timestamps = np.array(pose_timestamps, dtype=np.int64)
    if len(pose_timestamps) == 0:
        raise RuntimeError(
            f"No pose messages found on topic {args.pose_topic} with header.frame_id={args.pose_frame} "
            f"and child_frame_id={args.base_frame}"
        )

    def find_pose_matrix(timestamp_ns):
        idx = find_nearest_index(pose_timestamps, timestamp_ns)
        return pose_matrices[idx]

    print("Writing extracted dataset frames...")
    saved = 0
    color_index = 0
    keyframe_image = None
    for _, msg, _ in bag.read_messages(topics=[args.color_topic]):
        if color_index < args.start_frame or (args.end_frame >= 0 and color_index > args.end_frame):
            color_index += 1
            continue

        # Decode color image
        color_np = None
        if getattr(msg, '_type', '') == 'sensor_msgs/CompressedImage':
            arr = np.frombuffer(msg.data, np.uint8)
            color_np = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if color_np is None:
                raise RuntimeError(f"Failed to decode compressed color image at index {color_index}")
        elif getattr(msg, '_type', '') == 'sensor_msgs/Image':
            color_np = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        else:
            raise RuntimeError(f"Unsupported color message type: {getattr(msg, '_type', type(msg))}")

        # Keyframe selection based on disparity
        if keyframe_image is not None and args.min_disparity > 0:
            prev_gray = cv2.cvtColor(keyframe_image, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(color_np, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            disparity = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
            if disparity < args.min_disparity:
                color_index += 1
                continue

        # Downsample fallback if disparity disabled
        if args.min_disparity == 0 and color_index % args.downsample != 0:
            color_index += 1
            continue

        if args.max_frames and saved >= args.max_frames:
            break

        timestamp_ns = msg.header.stamp.to_nsec()
        depth_cv = None
        if depth_timestamps.size > 0:
            depth_idx = find_nearest_index(depth_timestamps, timestamp_ns)
            depth_ts, depth_cv = depth_messages[depth_idx]
            dt_ms = abs(depth_ts - timestamp_ns) / 1e6
            if dt_ms > 50:
                if args.allow_no_depth or args.no_lidar:
                    depth_cv = None
                else:
                    print(f"Skipping frame {color_index}: depth timestamp mismatch {dt_ms:.1f}ms")
                    color_index += 1
                    continue

        if (depth_cv is None or args.no_lidar) and (args.allow_no_depth or args.no_lidar):
            depth_cv = np.zeros((depth_size[1], depth_size[0]), dtype=np.uint16)

        pose_mat = find_pose_matrix(timestamp_ns)
        
        if args.rescale_mm_to_m:
            pose_mat[:3, 3] /= 1000.0
            
        # Motion Alignment: Force Camera Forward (Z) to point along Robot Motion Axis
        world_from_color = pose_mat @ R_MOTION_ALIGNED
        camera_from_world = invert_transform(world_from_color)

        frame_name = str(color_index)
        wide_dir = os.path.join(video_root, f"{frame_name}.wide")
        gt_dir = os.path.join(video_root, f"{frame_name}.gt")
        wide_meta_dir = os.path.join(video_root, f"{frame_name}._wide")
        gt_meta_dir = os.path.join(video_root, f"{frame_name}._gt")
        os.makedirs(wide_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(os.path.join(wide_meta_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(wide_meta_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(gt_meta_dir, "depth"), exist_ok=True)

        # Save color image
        save_image_file(os.path.join(wide_dir, "image.png"), color_np)

        # Save wide depth image
        save_depth_file(os.path.join(wide_dir, "depth.png"), depth_cv)
        save_depth_file(os.path.join(gt_dir, "depth.png"), depth_cv)

        # Save intrinsics
        save_json_matrix(os.path.join(wide_dir, "image", "k"), color_K)
        save_json_matrix(os.path.join(wide_dir, "depth", "k"), depth_K)
        save_json_matrix(os.path.join(gt_dir, "depth", "k"), depth_K)
        save_json_matrix(os.path.join(gt_dir, "rt"), camera_from_world)

        # Save size metadata
        save_size_file(os.path.join(wide_meta_dir, "image", "size"), color_size)
        save_size_file(os.path.join(wide_meta_dir, "depth", "size"), depth_size)
        save_size_file(os.path.join(gt_meta_dir, "depth", "size"), depth_size)

        # Save gravity transform as identity for wide frames
        save_json_matrix(os.path.join(wide_dir, "t_gravity"), np.eye(3, dtype=np.float32))

        saved += 1
        keyframe_image = color_np
        color_index += 1
        if saved % 50 == 0:
            print(f"Saved {saved} frames...")

    bag.close()
    print(f"Finished writing {saved} frames to {video_root}")


if __name__ == "__main__":
    main()

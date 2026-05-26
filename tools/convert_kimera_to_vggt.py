#!/usr/bin/env python3
"""
Convert a Kimera ROS bag to the VGGT CubifyAnythingExtracted training format.

Output structure per frame (matches base_dataset.py + cubifyanything_extracted.py):
  <output>/<video_id>/
    <ts>.wide/
      image.png               - RGB image
      depth.png               - LIDAR-projected depth, uint16 mm  (ARKit-style copy)
      image/k.json            - 3×3 camera intrinsics
      depth/k.json            - 3×3 camera intrinsics (same)
      t_gravity.json          - 3×3 gravity alignment (identity)
    <ts>.gt/
      depth.png               - LIDAR-projected depth, uint16 mm
      rt.json                 - 4×4 camera-from-world (OpenCV convention)
      depth/k.json            - 3×3 camera intrinsics
      lidar.ply               - raw LIDAR cloud in world coordinates
    <ts>._wide/
      image/size              - "[W, H]"
      depth/size              - "[W, H]"
    <ts>._gt/
      depth/size              - "[W, H]"

Metadata dirs (._wide, ._gt) are written ONLY after ALL content files are
confirmed on disk — so partial frames are never left behind.
"""

import argparse
import json
import os
import shutil
import struct
from collections import deque

import cv2
import numpy as np
import rosbag
from tqdm import tqdm

MM_TO_M = 1000.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quaternion_to_rot3(q):
    x, y, z, w = q
    n = x*x + y*y + z*z + w*w
    if n < np.finfo(float).eps:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xs, ys, zs = x*s, y*s, z*s
    xx, yy, zz = x*xs, y*ys, z*zs
    xy, xz, yz = x*ys, x*zs, y*zs
    wx, wy, wz = w*xs, w*ys, w*zs
    return np.array([
        [1.0-(yy+zz),  xy-wz,       xz+wy],
        [xy+wz,        1.0-(xx+zz), yz-wx],
        [xz-wy,        yz+wx,       1.0-(xx+yy)],
    ], dtype=np.float32)


def pose_to_matrix(position, orientation) -> np.ndarray:
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = quaternion_to_rot3((
        orientation.x, orientation.y, orientation.z, orientation.w))
    mat[:3, 3] = [position.x, position.y, position.z]
    return mat


def invert_se3(mat: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float32)
    R = mat[:3, :3]
    t = mat[:3, 3]
    inv[:3, :3] = R.T
    inv[:3, 3] = -(R.T @ t)
    return inv


# ---------------------------------------------------------------------------
# TF graph helpers
# ---------------------------------------------------------------------------

def build_tf_graph(tf_static_msgs):
    graph = {}
    for msg in tf_static_msgs:
        if not hasattr(msg, 'transforms'):
            continue
        for tr in msg.transforms:
            parent = tr.header.frame_id.lstrip('/')
            child  = tr.child_frame_id.lstrip('/')
            graph[(parent, child)] = pose_to_matrix(
                tr.transform.translation, tr.transform.rotation)
    return graph


def find_tf(graph, source, target):
    source = source.lstrip('/')
    target = target.lstrip('/')
    if source == target:
        return np.eye(4, dtype=np.float32)
    queue   = deque([(source, np.eye(4, dtype=np.float32))])
    visited = {source}
    while queue:
        frame, mat = queue.popleft()
        if frame == target:
            return mat
        for (p, c), m in graph.items():
            if p == frame and c not in visited:
                visited.add(c)
                queue.append((c, mat @ m))
            elif c == frame and p not in visited:
                visited.add(p)
                queue.append((p, mat @ invert_se3(m)))
    raise ValueError(f"No TF path: {source} → {target}")


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def read_pointcloud2_xyz(msg) -> np.ndarray:
    """Decode a sensor_msgs/PointCloud2 to an (N,3) float32 array via numpy."""
    field_offsets = {f.name: f.offset for f in msg.fields}
    offsets = [field_offsets['x'], field_offsets['y'], field_offsets['z']]
    step   = msg.point_step
    n_pts  = msg.width * msg.height
    raw    = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    pts = np.zeros((n_pts, 3), dtype=np.float32)
    for i, off in enumerate(offsets):
        col_bytes = raw.reshape(n_pts, step)[:, off:off+4]
        pts[:, i] = np.frombuffer(col_bytes.tobytes(), dtype=np.float32)
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def project_lidar_to_depth(points_lidar: np.ndarray,
                            K: np.ndarray,
                            cam_from_lidar_4x4: np.ndarray,
                            width: int, height: int) -> np.ndarray:
    """Project the LIDAR cloud into the camera plane → float32 depth (metres)."""
    if points_lidar.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)
    hom = np.hstack([points_lidar,
                     np.ones((points_lidar.shape[0], 1), dtype=np.float32)])
    pts_cam = (cam_from_lidar_4x4[:3, :] @ hom.T).T
    front = pts_cam[:, 2] > 0.05
    pts_cam = pts_cam[front]
    if pts_cam.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)
    uvz = (K @ pts_cam.T).T
    u = (uvz[:, 0] / uvz[:, 2]).astype(np.int32)
    v = (uvz[:, 1] / uvz[:, 2]).astype(np.int32)
    z = uvz[:, 2]
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z  = u[in_frame], v[in_frame], z[in_frame]
    depth = np.zeros((height, width), dtype=np.float32)
    order = np.argsort(z)[::-1]          # far → near; nearest wins
    depth[v[order], u[order]] = z[order]
    return depth


def transform_points_world(points_lidar: np.ndarray,
                            world_from_lidar_4x4: np.ndarray) -> np.ndarray:
    hom = np.hstack([points_lidar,
                     np.ones((points_lidar.shape[0], 1), dtype=np.float32)])
    return (world_from_lidar_4x4[:3, :] @ hom.T).T


def write_ply(path: str, points_xyz: np.ndarray):
    """Write a minimal ASCII PLY file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points_xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    with open(path, 'w') as f:
        f.write(header)
        np.savetxt(f, points_xyz, fmt='%.6f')


# ---------------------------------------------------------------------------
# Misc I/O
# ---------------------------------------------------------------------------

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f)


def save_size(path, w, h):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"[{w}, {h}]\n")


def file_ok(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Kimera bag → VGGT extracted format")
    p.add_argument("--bag",          required=True)
    p.add_argument("--output",       default="extracted_kimera")
    p.add_argument("--save-endpoint", default=None,
                   help="Direct output directory (overrides --output and --video-id)")
    p.add_argument("--video-id",     type=int, default=0)
    p.add_argument("--downsample",   type=int, default=10)
    p.add_argument("--max-frames",   type=int, default=0,
                   help="Stop after this many saved frames (0 = unlimited)")
    p.add_argument("--img-topic",    default="/sparkal1/forward/color/image_raw/compressed")
    p.add_argument("--info-topic",   default="/sparkal1/forward/color/camera_info")
    p.add_argument("--odom-topic",   default="/sparkal1/kimera_vio_ros/odometry")
    p.add_argument("--lidar-topic",  default="/sparkal1/lidar_points")
    p.add_argument("--camera-frame", default="sparkal1/forward_color_optical_frame")
    p.add_argument("--base-frame",   default="sparkal1/base_link")
    p.add_argument("--lidar-frame",  default="sparkal1/velodyne")
    p.add_argument("--no-ply",       action="store_true",
                   help="Skip writing per-frame PLY files (faster, less disk)")
    args = p.parse_args()

    # Use save-endpoint if provided, otherwise use output/video-id structure
    if args.save_endpoint:
        video_root = args.save_endpoint
    else:
        video_root = os.path.join(args.output, str(args.video_id))
    os.makedirs(video_root, exist_ok=True)

    print(f"Opening bag: {args.bag}")
    bag = rosbag.Bag(args.bag, "r")

    tf_graph     = {}
    K            = None
    size         = None          # (width, height)
    latest_odom  = None
    latest_lidar = None

    # Cached transforms (resolved once TF graph is known)
    cam_from_base  = None
    cam_from_lidar = None

    img_count   = 0
    saved_count = 0
    skip_counts = {
        "no_calibration": 0,
        "no_odom":        0,
        "no_lidar":       0,
        "stale_odom":     0,
        "stale_lidar":    0,
        "no_tf":          0,
        "bad_image":      0,
        "empty_depth":    0,
        "write_failed":   0,
    }

    topics = [args.img_topic, args.info_topic,
              args.odom_topic, args.lidar_topic, '/tf_static']

    print("Starting single-pass extraction...")
    pbar = tqdm(desc="msgs", unit="msg")

    for topic, msg, _t in bag.read_messages(topics=topics):
        pbar.update(1)

        # ── Bookkeeping ───────────────────────────────────────────────────
        if topic == '/tf_static':
            tf_graph.update(build_tf_graph([msg]))
            cam_from_base  = None   # invalidate cache (new transforms arrived)
            cam_from_lidar = None
            continue

        if topic == args.info_topic:
            K    = np.array(msg.K, dtype=np.float32).reshape(3, 3)
            size = (msg.width, msg.height)
            continue

        if topic == args.odom_topic:
            latest_odom = msg
            continue

        if topic == args.lidar_topic:
            latest_lidar = msg
            continue

        if topic != args.img_topic:
            continue

        # ── Downsample ────────────────────────────────────────────────────
        img_count += 1
        if img_count % args.downsample != 0:
            continue

        img_ts_ns = msg.header.stamp.to_nsec()

        # ── Guards: skip frame if any required data is missing/stale ──────
        if K is None or size is None:
            skip_counts["no_calibration"] += 1
            continue

        if latest_odom is None:
            skip_counts["no_odom"] += 1
            continue
        if abs(latest_odom.header.stamp.to_nsec() - img_ts_ns) > 5e9:
            skip_counts["stale_odom"] += 1
            continue

        if latest_lidar is None:
            skip_counts["no_lidar"] += 1
            continue
        if abs(latest_lidar.header.stamp.to_nsec() - img_ts_ns) > 5e8:  # 500 ms
            skip_counts["stale_lidar"] += 1
            continue

        # ── Static transforms (cached) ────────────────────────────────────
        if cam_from_base is None or cam_from_lidar is None:
            try:
                cam_from_base  = find_tf(tf_graph, args.base_frame,  args.camera_frame)
                cam_from_lidar = find_tf(tf_graph, args.lidar_frame, args.camera_frame)
            except Exception:
                skip_counts["no_tf"] += 1
                continue

        # ── Decode image ──────────────────────────────────────────────────
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skip_counts["bad_image"] += 1
            continue

        # ── Project LIDAR → depth ─────────────────────────────────────────
        pts_lidar = read_pointcloud2_xyz(latest_lidar)
        depth_m   = project_lidar_to_depth(
            pts_lidar, K, cam_from_lidar, size[0], size[1])

        if depth_m.max() == 0:
            skip_counts["empty_depth"] += 1
            continue

        # ── Compute poses ─────────────────────────────────────────────────
        world_from_base   = pose_to_matrix(latest_odom.pose.pose.position,
                                           latest_odom.pose.pose.orientation)
        world_from_camera = world_from_base @ invert_se3(cam_from_base)
        camera_from_world = invert_se3(world_from_camera)

        base_from_lidar  = invert_se3(cam_from_base) @ cam_from_lidar
        world_from_lidar = world_from_base @ base_from_lidar
        pts_world        = transform_points_world(pts_lidar, world_from_lidar)

        # ── Build paths ───────────────────────────────────────────────────
        ts   = str(img_ts_ns)
        W, H = size

        wide_dir  = os.path.join(video_root, f"{ts}.wide")
        gt_dir    = os.path.join(video_root, f"{ts}.gt")
        _wide_dir = os.path.join(video_root, f"{ts}._wide")
        _gt_dir   = os.path.join(video_root, f"{ts}._gt")

        # Create CONTENT directories (not metadata yet)
        for d in [wide_dir,
                  os.path.join(wide_dir, "image"),
                  os.path.join(wide_dir, "depth"),
                  gt_dir,
                  os.path.join(gt_dir, "depth")]:
            os.makedirs(d, exist_ok=True)

        depth_mm = (depth_m * MM_TO_M).astype(np.uint16)
        K_list   = K.tolist()

        # ── Write all content files, check every write succeeds ───────────
        ok = True
        try:
            ok = ok and cv2.imwrite(os.path.join(wide_dir, "image.png"),   img)
            ok = ok and cv2.imwrite(os.path.join(wide_dir, "depth.png"),   depth_mm)
            ok = ok and cv2.imwrite(os.path.join(gt_dir,   "depth.png"),   depth_mm)
            save_json(os.path.join(wide_dir, "image", "k.json"),           K_list)
            save_json(os.path.join(wide_dir, "depth", "k.json"),           K_list)
            save_json(os.path.join(gt_dir,   "depth", "k.json"),           K_list)
            save_json(os.path.join(gt_dir,   "rt.json"),                   camera_from_world.tolist())
            save_json(os.path.join(wide_dir, "t_gravity.json"),            np.eye(3).tolist())
            if not args.no_ply:
                write_ply(os.path.join(gt_dir, "lidar.ply"),               pts_world)
        except Exception:
            ok = False

        # Verify required files are non-empty on disk
        if ok:
            for fpath in [os.path.join(wide_dir, "image.png"),
                          os.path.join(wide_dir, "depth.png"),
                          os.path.join(gt_dir,   "depth.png"),
                          os.path.join(gt_dir,   "rt.json")]:
                if not file_ok(fpath):
                    ok = False
                    break

        # If anything failed → remove all dirs for this timestamp
        if not ok:
            for d in [wide_dir, gt_dir, _wide_dir, _gt_dir]:
                shutil.rmtree(d, ignore_errors=True)
            skip_counts["write_failed"] += 1
            continue

        # ── Write metadata ONLY after content verified ────────────────────
        for d in [os.path.join(_wide_dir, "image"),
                  os.path.join(_wide_dir, "depth"),
                  os.path.join(_gt_dir,   "depth")]:
            os.makedirs(d, exist_ok=True)

        save_size(os.path.join(_wide_dir, "image", "size"), W, H)
        save_size(os.path.join(_wide_dir, "depth", "size"), W, H)
        save_size(os.path.join(_gt_dir,   "depth", "size"), W, H)

        saved_count += 1
        pbar.set_postfix(saved=saved_count)

        if args.max_frames > 0 and saved_count >= args.max_frames:
            print(f"\nReached max-frames limit ({args.max_frames}).")
            break

    bag.close()
    pbar.close()

    print(f"\n✓ Extraction complete. {saved_count} frames → {video_root}")
    total_skipped = sum(skip_counts.values())
    if total_skipped:
        print(f"  Skipped {total_skipped} candidate frames:")
        for reason, count in skip_counts.items():
            if count:
                print(f"    {reason:20s}: {count}")


if __name__ == "__main__":
    main()

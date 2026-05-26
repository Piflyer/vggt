#!/bin/bash
# Clean snapshot packaging utility

echo "=========================================================="
echo "📦 Packaging VGGT Rosbag & Fine-Tuning Snapshot..."
echo "=========================================================="

# Resolve parent directory
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ARCHIVE="${PARENT_DIR}/../vggt-rosbag-snapshot.tar.gz"

# Run compression, ignoring git files, caches, weights, and bags
tar -czvf "${TARGET_ARCHIVE}" \
    --exclude-vcs \
    --exclude='*.bag' \
    --exclude='*.pth' \
    --exclude='*.pt' \
    --exclude='*.tar' \
    --exclude='*.tar.gz' \
    --exclude='__pycache__' \
    --exclude='.gradio' \
    -C "${PARENT_DIR}" .

echo "=========================================================="
echo "✅ Snapshot package created successfully!"
echo "Location: $(realpath "${TARGET_ARCHIVE}")"
echo "=========================================================="

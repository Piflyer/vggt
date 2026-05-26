import os
import tarfile
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial

def extract_tar(tar_path, dest_root):
    try:
        with tarfile.open(tar_path, 'r') as tar:
            # We want to extract to dest_root
            # The tar files contain paths like "video_id/timestamp.sensor/measurement"
            # So extracting them directly to dest_root should work and merge them correctly.
            
            # Safety check: ensure members don't have absolute paths or ..
            members = []
            for member in tar.getmembers():
                if member.name.startswith('/') or '..' in member.name:
                    print(f"Skipping unsafe member {member.name} in {tar_path}")
                    continue
                members.append(member)
            
            tar.extractall(path=dest_root, members=members)
            return True
    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Extract CubifyAnything tar files to a directory.")
    parser.add_argument("--src_dir", type=str, required=True, help="Directory containing .tar files")
    parser.add_argument("--dest_dir", type=str, required=True, help="Destination directory for extracted data")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--pattern", type=str, default="*.tar", help="Glob pattern for tar files")
    
    args = parser.parse_args()
    
    dest_path = Path(args.dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Handle src_dir being a glob pattern or a directory
    if '*' in args.src_dir or '?' in args.src_dir:
        # It's a glob pattern
        tar_files = sorted([Path(p) for p in glob.glob(args.src_dir)])
        print(f"Found {len(tar_files)} files matching pattern {args.src_dir}")
    else:
        src_path = Path(args.src_dir)
        if src_path.is_dir():
            tar_files = sorted(list(src_path.glob(args.pattern)))
            print(f"Found {len(tar_files)} tar files in {src_path} with pattern {args.pattern}")
        elif src_path.is_file():
             tar_files = [src_path]
             print(f"Processing single file: {src_path}")
        else:
            print(f"Error: {src_path} is not a directory or file, and not a valid glob pattern.")
            return
    
    if len(tar_files) == 0:
        print("No tar files found. Exiting.")
        return

    # Use multiprocessing to extract in parallel
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        func = partial(extract_tar, dest_root=str(dest_path))
        results = list(tqdm(pool.imap_unordered(func, tar_files), total=len(tar_files), desc="Extracting tars"))
    
    success_count = sum(results)
    print(f"Successfully extracted {success_count}/{len(tar_files)} files.")

if __name__ == "__main__":
    main()

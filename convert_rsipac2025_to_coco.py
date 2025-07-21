#!/usr/bin/env python3
"""
RSIPAC2025 to COCO format conversion script
Converts RSIPAC2025 dataset to COCO format for DSFNet training

Usage:
    python convert_rsipac2025_to_coco.py --input_dir datasets/RSIPAC2025/Preliminary --output_dir data/RsCarData

CSV format: frame_id, object_id, x, y, width, height, class_id(-1), -1, -1, -1
COCO format: Standard COCO JSON with images, annotations, and categories
"""

import os
import json
import csv
import cv2
import argparse
from datetime import datetime
from pathlib import Path
import glob


class RSIPAC2025ToCOCO:
    def __init__(self, input_dir, output_dir, val_ratio=0.2, val_split_strategy='video'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.val_ratio = val_ratio
        self.val_split_strategy = val_split_strategy  # 'video' or 'temporal'
        
        # COCO format structure
        self.coco_train = {
            "info": {
                "description": "RSIPAC2025 Dataset converted to COCO format",
                "version": "1.0",
                "year": 2025,
                "contributor": "RSIPAC2025",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "car", "supercategory": "vehicle"}]
        }
        
        self.coco_val = {
            "info": self.coco_train["info"].copy(),
            "licenses": self.coco_train["licenses"].copy(),
            "images": [],
            "annotations": [],
            "categories": self.coco_train["categories"].copy()
        }
        
        self.image_id = 1
        self.annotation_id = 1
        
    def extract_frames_from_video(self, video_path, output_folder):
        """Extract frames from video file"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return []
        
        frame_paths = []
        frame_id = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_filename = f"{output_folder.name}/frame_{frame_id:06d}.jpg"
            frame_path = output_folder / f"frame_{frame_id:06d}.jpg"
            
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append((frame_filename, frame.shape[1], frame.shape[0]))  # (filename, width, height)
            frame_id += 1
        
        cap.release()
        return frame_paths
    
    def parse_csv_annotations(self, csv_path):
        """Parse CSV annotation file"""
        annotations = {}
        
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 6:
                    frame_id = int(row[0])
                    obj_id = int(row[1])
                    x = float(row[2])
                    y = float(row[3])
                    width = float(row[4])
                    height = float(row[5])
                    
                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    
                    annotations[frame_id].append({
                        'obj_id': obj_id,
                        'bbox': [x, y, width, height],
                        'category_id': 1  # car
                    })
        
        return annotations
    
    def split_videos_for_validation(self, video_files):
        """Split videos into train and validation sets"""
        import random
        
        # Sort for reproducible results
        video_files = sorted(video_files)
        
        if self.val_split_strategy == 'video':
            # Strategy 1: Split by complete videos (recommended for sequence data)
            random.seed(42)  # For reproducible results
            random.shuffle(video_files)
            
            val_count = max(1, int(len(video_files) * self.val_ratio))
            val_videos = video_files[:val_count]
            train_videos = video_files[val_count:]
            
            print(f"Video-based split: {len(train_videos)} train videos, {len(val_videos)} val videos")
            print(f"Validation videos: {[Path(v).stem for v in val_videos]}")
            
            return train_videos, val_videos
        
        else:  # temporal split
            # Strategy 2: Split by temporal segments within each video
            # This is more complex but provides more validation data
            train_videos = []
            val_videos = []
            
            for video_path in video_files:
                # For temporal split, we'll process each video and split its frames
                # This requires different processing logic
                train_videos.append(video_path)
            
            print(f"Temporal split: {len(train_videos)} videos (will split frames within each)")
            return train_videos, val_videos
    
    def process_video_data(self, video_files, output_dir, coco_data, split_name):
        """Process video data for either train or validation"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for video_path in video_files:
            video_path = Path(video_path)
            video_name = video_path.stem
            csv_path = video_path.parent / f"{video_name}-gt.csv"
            
            if not csv_path.exists():
                print(f"Warning: CSV file not found for {video_name}")
                continue
                
            print(f"Processing {split_name} video {video_name}...")
            
            # Create output folder for this video
            video_output_dir = output_dir / video_name
            video_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract frames
            frame_paths = self.extract_frames_from_video(video_path, video_output_dir)
            
            # Parse annotations
            annotations = self.parse_csv_annotations(csv_path)
            
            if self.val_split_strategy == 'temporal' and split_name == 'train':
                # For temporal split, only use middle 60% of frames for training
                # Leave first 20% and last 20% for validation
                total_frames = len(frame_paths)
                start_idx = int(total_frames * 0.2)
                end_idx = int(total_frames * 0.8)
                frame_paths = frame_paths[start_idx:end_idx]
            elif self.val_split_strategy == 'temporal' and split_name == 'val':
                # For temporal split validation, use first 20% and last 20%
                total_frames = len(frame_paths)
                start_idx = int(total_frames * 0.2)
                end_idx = int(total_frames * 0.8)
                frame_paths = frame_paths[:start_idx] + frame_paths[end_idx:]
            
            # Convert to COCO format
            for idx, (frame_filename, width, height) in enumerate(frame_paths):
                if self.val_split_strategy == 'temporal' and split_name == 'val':
                    # Adjust frame_id for temporal validation split
                    if idx < int(len(frame_paths) * 0.5):
                        frame_id = idx  # First part
                    else:
                        # Second part (from end of video)
                        frame_id = int(len(frame_paths) / 0.4 * 0.8) + (idx - int(len(frame_paths) * 0.5))
                else:
                    frame_id = idx if self.val_split_strategy != 'temporal' else int(len(frame_paths) / 0.6 * 0.2) + idx
                
                # Add image info
                image_info = {
                    "id": self.image_id,
                    "width": width,
                    "height": height,
                    "file_name": frame_filename,
                    "license": 1,
                    "date_captured": ""
                }
                coco_data["images"].append(image_info)
                
                # Add annotations for this frame
                if frame_id in annotations:
                    for ann in annotations[frame_id]:
                        bbox = ann['bbox']
                        annotation = {
                            "id": self.annotation_id,
                            "image_id": self.image_id,
                            "category_id": ann['category_id'],
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                            "segmentation": []
                        }
                        coco_data["annotations"].append(annotation)
                        self.annotation_id += 1
                
                self.image_id += 1
    
    def process_train_data(self):
        """Process training data with train/val split"""
        train_dir = self.input_dir / "train"
        train_images_dir = self.output_dir / "train"
        val_images_dir = self.output_dir / "val"
        
        # Get all video files
        video_files = glob.glob(str(train_dir / "*.avi"))
        
        # Split videos into train and validation
        train_videos, val_videos = self.split_videos_for_validation(video_files)
        
        # Process training videos
        self.process_video_data(train_videos, train_images_dir, self.coco_train, "train")
        
        # Process validation videos
        if self.val_split_strategy == 'video':
            self.process_video_data(val_videos, val_images_dir, self.coco_val, "val")
        else:  # temporal split
            # For temporal split, process all videos but split frames within each
            self.process_video_data(video_files, val_images_dir, self.coco_val, "val")
    
    def process_test_data(self):
        """Process original test data (without annotations) for final evaluation"""
        test_dir = self.input_dir / "val"
        test_images_dir = self.output_dir / "test"
        test_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test COCO structure
        coco_test = {
            "info": self.coco_train["info"].copy(),
            "licenses": self.coco_train["licenses"].copy(),
            "images": [],
            "annotations": [],
            "categories": self.coco_train["categories"].copy()
        }
        
        # Check if frames are already extracted
        extracted_dirs = [d for d in test_dir.iterdir() if d.is_dir()]
        if extracted_dirs:
            print("Processing pre-extracted test frames...")
            
            # Copy extracted frames with video name prefix
            import shutil
            for video_dir in extracted_dirs:
                video_name = video_dir.name
                video_output_dir = test_images_dir / video_name
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                for frame_file in video_dir.glob("frame_*.jpg"):
                    dest_path = video_output_dir / frame_file.name
                    shutil.copy2(frame_file, dest_path)
                    
                    # Get image dimensions
                    img = cv2.imread(str(frame_file))
                    height, width = img.shape[:2]
                    
                    # Add image info (no annotations for test)
                    image_info = {
                        "id": self.image_id,
                        "width": width,
                        "height": height,
                        "file_name": f"{video_name}/{frame_file.name}",
                        "license": 1,
                        "date_captured": ""
                    }
                    coco_test["images"].append(image_info)
                    self.image_id += 1
        else:
            # Extract frames from video files
            video_files = glob.glob(str(test_dir / "*.avi"))
            
            for video_path in video_files:
                video_path = Path(video_path)
                video_name = video_path.stem
                
                print(f"Processing test video {video_name}...")
                
                # Create output folder for this video
                video_output_dir = test_images_dir / video_name
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract frames
                frame_paths = self.extract_frames_from_video(video_path, video_output_dir)
                
                # Add image info (no annotations for test)
                for frame_filename, width, height in frame_paths:
                    image_info = {
                        "id": self.image_id,
                        "width": width,
                        "height": height,
                        "file_name": frame_filename,
                        "license": 1,
                        "date_captured": ""
                    }
                    coco_test["images"].append(image_info)
                    self.image_id += 1
        
        # Save test annotations
        annotations_dir = self.output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        test_json_path = annotations_dir / "rsipac2025_test.json"
        with open(test_json_path, 'w') as f:
            json.dump(coco_test, f, indent=2)
    
    def save_coco_files(self):
        """Save COCO JSON files"""
        # Create annotations directory
        annotations_dir = self.output_dir / "annotations"
        annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training annotations
        train_json_path = annotations_dir / "rsipac2025_train.json"
        with open(train_json_path, 'w') as f:
            json.dump(self.coco_train, f, indent=2)
        
        # Save validation annotations
        val_json_path = annotations_dir / "rsipac2025_val.json"
        with open(val_json_path, 'w') as f:
            json.dump(self.coco_val, f, indent=2)
        
        print(f"COCO annotations saved to {annotations_dir}")
        print(f"Training images: {len(self.coco_train['images'])}")
        print(f"Training annotations: {len(self.coco_train['annotations'])}")
        print(f"Validation images: {len(self.coco_val['images'])}")
    
    def convert(self):
        """Main conversion function"""
        print("Starting RSIPAC2025 to COCO conversion...")
        print(f"Split strategy: {self.val_split_strategy}")
        print(f"Validation ratio: {self.val_ratio}")
        
        # Process training data (includes train/val split)
        self.process_train_data()
        
        # Process test data (original val directory without annotations)
        self.process_test_data()
        
        # Save COCO files
        self.save_coco_files()
        
        print("Conversion completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Convert RSIPAC2025 dataset to COCO format")
    parser.add_argument('--input_dir', type=str, default='/data/hlj/Competition/RSIPAC2025/Preliminary',
                        help='Input directory containing RSIPAC2025 dataset')
    parser.add_argument('--output_dir', type=str, default='data/RSIPAC2025',
                        help='Output directory for COCO format dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio (0.0-1.0)')
    parser.add_argument('--val_split_strategy', type=str, default='video', 
                        choices=['video', 'temporal'],
                        help='Validation split strategy: video (split by videos) or temporal (split by time within videos)')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist!")
        return
    
    # Create converter and run conversion
    converter = RSIPAC2025ToCOCO(args.input_dir, args.output_dir, args.val_ratio, args.val_split_strategy)
    converter.convert()


if __name__ == "__main__":
    main()
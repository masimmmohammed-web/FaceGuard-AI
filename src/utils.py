"""
Utility functions for the deepfake-resistant facial biometrics system.
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_image(image_path: str) -> Optional[np.ndarray]:
    """Load an image from file path."""
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image to target dimensions while maintaining aspect ratio."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create target canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Center the resized image
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for deep learning models."""
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to target size
    resized = resize_image(image_rgb, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def extract_frames(video_path: str, max_frames: int = 30, extract_rate: int = 3) -> List[np.ndarray]:
    """Extract frames from video for temporal analysis.
    
    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract
        extract_rate: Extract every nth frame (default: 3)
    
    Returns:
        List of extracted frames as numpy arrays
    """
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every nth frame to get temporal diversity
            if frame_count % extract_rate == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Extracted {len(frames)} frames from video")
        
    except Exception as e:
        logger.error(f"Error extracting frames: {str(e)}")
    
    return frames

def calculate_optical_flow(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Calculate optical flow between two consecutive frames."""
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    return flow

def detect_blinking(eye_landmarks: List[Tuple[int, int]]) -> bool:
    """Detect if eyes are blinking based on landmark positions."""
    if len(eye_landmarks) < 6:
        return False
    
    # Calculate eye aspect ratio (EAR)
    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    try:
        # Extract eye points (simplified - in practice use MediaPipe landmarks)
        p1, p2, p3, p4, p5, p6 = eye_landmarks[:6]
        
        # Calculate distances
        A = np.linalg.norm(np.array(p2) - np.array(p6))
        B = np.linalg.norm(np.array(p3) - np.array(p5))
        C = np.linalg.norm(np.array(p1) - np.array(p4))
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        # Threshold for blinking detection
        return ear < 0.2
        
    except Exception as e:
        logger.error(f"Error in blink detection: {str(e)}")
        return False

def calculate_face_quality(image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> float:
    """Calculate face image quality score."""
    x, y, w, h = face_bbox
    face_roi = image[y:y+h, x:x+w]
    
    if face_roi.size == 0:
        return 0.0
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian variance (sharpness)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Calculate contrast
    contrast = np.std(gray)
    
    # Normalize and combine metrics
    sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize to [0, 1]
    brightness_score = 1.0 - abs(brightness - 128) / 128.0  # Optimal around 128
    contrast_score = min(contrast / 50.0, 1.0)  # Normalize to [0, 1]
    
    # Weighted combination
    quality_score = 0.4 * sharpness_score + 0.3 * brightness_score + 0.3 * contrast_score
    
    return max(0.0, min(1.0, quality_score))

def save_results(results: dict, output_path: str):
    """Save analysis results to file."""
    try:
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def create_output_directory(directory: str):
    """Create output directory if it doesn't exist."""
    try:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Output directory created: {directory}")
    except Exception as e:
        logger.error(f"Error creating directory: {str(e)}")

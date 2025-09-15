#!/usr/bin/env python3
"""
Test script to verify strict validation prevents false positives
"""

import cv2
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(os.path.dirname(os.path.dirname(__file__))))

from src.face_detector import FaceDetector
from live_analyzer import _validate_simple_challenge

def test_strict_validation():
    """Skipped: hand sign challenges removed."""
    print("Hand sign challenges have been removed. Skipping strict validation test.")
    return
    
    # Initialize face detector
    face_detector = FaceDetector()
    
    # Load a test image (you can replace this with any image)
    test_image_path = "../images/sample_face.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please place a test image in the images folder")
        return
    
    # Load image
    frame = cv2.imread(test_image_path)
    if frame is None:
        print("Failed to load test image")
        return
    
    print(f"Testing strict validation with image: {test_image_path}")
    print(f"Image size: {frame.shape}")
    
    # Detect faces
    faces = face_detector.detect_faces(frame)
    
    if not faces:
        print("No faces detected in test image")
        return
    
    print(f"Found {len(faces)} face(s)")
    
    # Test with first face
    face = faces[0]
    bbox = face['bbox']
    print(f"Face bbox: {bbox}")
    
    # Unreachable in skip mode
    
    # Save debug image
    debug_frame = frame.copy()
    x, y, w, h = bbox
    cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    debug_path = "strict_validation_test_debug.jpg"
    cv2.imwrite(debug_path, debug_frame)
    print(f"\nDebug image saved: {debug_path}")
    
    print("\n=== Expected Behavior ===")
    print("1. Main validation should fail (no hand present)")
    print("2. Fallback validation should fail (no hand present)")
    print("3. Temporal consistency should prevent false positives")
    print("4. Challenge should NOT be completed without actual hand signs")

if __name__ == "__main__":
    test_strict_validation()

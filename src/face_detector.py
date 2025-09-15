"""
Face Detector for FaceGuard AI using dlib HOG-based detection
More reliable than Haar cascade and works better with the shape predictor
"""

import cv2
import numpy as np
import logging
from typing import List, Dict

try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    dlib = None
    _DLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

class FaceDetector:
    """Face detection using dlib HOG-based detector (more reliable than Haar cascade)."""
    
    def __init__(self):
        """Initialize face detector with dlib HOG detector."""
        if not _DLIB_AVAILABLE:
            logger.error("dlib not available - falling back to OpenCV Haar cascade")
            self._init_opencv_fallback()
            return
        
        try:
            # Initialize dlib HOG-based face detector
            self.detector = dlib.get_frontal_face_detector()
            self.use_dlib = True
            logger.info("Face detector initialized with dlib HOG detector")
        except Exception as e:
            logger.error(f"Failed to initialize dlib detector: {e}")
            logger.info("Falling back to OpenCV Haar cascade")
            self._init_opencv_fallback()
    
    def _init_opencv_fallback(self):
        """Initialize OpenCV Haar cascade as fallback."""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            raise RuntimeError("No face detection method available")
        
        self.use_dlib = False
        logger.info("Face detector initialized with OpenCV Haar cascade (fallback)")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using dlib HOG detector (or OpenCV fallback).
        Returns list of face dictionaries with bbox and confidence.
        """
        try:
            # Validate input image
            if image is None or image.size == 0:
                logger.warning("Empty or None image provided")
                return []
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Validate image dimensions
            height, width = gray.shape[:2]
            if height < 50 or width < 50:
                logger.warning(f"Image too small for face detection: {width}x{height}")
                return []
            
            if self.use_dlib:
                return self._detect_faces_dlib(gray, image.shape)
            else:
                return self._detect_faces_opencv(gray, image.shape)
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return []
    
    def _detect_faces_dlib(self, gray: np.ndarray, image_shape: tuple) -> List[Dict]:
        """Detect faces using dlib HOG detector."""
        try:
            # dlib expects RGB format, but we have grayscale
            # Convert grayscale to RGB for dlib
            rgb_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            
            # Detect faces using dlib
            detected_faces = self.detector(rgb_image, 1)  # 1 = upsampling factor
            
            faces = []
            for face in detected_faces:
                # Convert dlib rectangle to (x, y, w, h) format
                x = face.left()
                y = face.top()
                w = face.width()
                h = face.height()
                
                # Calculate confidence based on face size and position
                confidence = self._calculate_dlib_confidence(x, y, w, h, image_shape)
                
                face_info = {
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'confidence': confidence
                }
                faces.append(face_info)
            
            # Sort by confidence (highest first)
            faces.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.debug(f"Detected {len(faces)} faces using dlib")
            return faces
            
        except Exception as e:
            logger.error(f"Error in dlib face detection: {str(e)}")
            return []
    
    def _detect_faces_opencv(self, gray: np.ndarray, image_shape: tuple) -> List[Dict]:
        """Detect faces using OpenCV Haar cascade (fallback)."""
        try:
            # Calculate adaptive parameters based on image size
            height, width = gray.shape[:2]
            min_size = max(30, min(height, width) // 10)  # Adaptive minimum size
            max_size = min(400, max(height, width) // 2)  # Adaptive maximum size
            
            # Ensure min_size < max_size
            if min_size >= max_size:
                min_size = max_size - 10
                if min_size < 30:
                    logger.warning("Image too small for reliable face detection")
                    return []
            
            # Use more conservative parameters to avoid scaleIdx errors
            try:
                detected = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,     # More conservative scale factor
                    minNeighbors=5,       # Higher threshold for stability
                    minSize=(min_size, min_size),
                    maxSize=(max_size, max_size),
                    flags=cv2.CASCADE_SCALE_IMAGE  # Explicit flag for better compatibility
                )
            except cv2.error as cv_error:
                # Fallback with even more conservative parameters
                logger.warning(f"Cascade detection failed with error: {cv_error}")
                logger.info("Attempting fallback with minimal parameters")
                try:
                    detected = self.face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                except cv2.error as fallback_error:
                    logger.error(f"Fallback detection also failed: {fallback_error}")
                    return []
            
            faces = []
            for (x, y, w, h) in detected:
                # Calculate simple confidence based on face size and position
                confidence = self._calculate_simple_confidence(x, y, w, h, image_shape)
                
                face_info = {
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'confidence': confidence
                }
                faces.append(face_info)
            
            # Sort by confidence (highest first)
            faces.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.debug(f"Detected {len(faces)} faces using OpenCV")
            return faces
            
        except Exception as e:
            logger.error(f"Error in OpenCV face detection: {str(e)}")
            return []
    
    def _calculate_dlib_confidence(self, x: int, y: int, w: int, h: int, image_shape: tuple) -> float:
        """Calculate confidence score for dlib-detected face."""
        try:
            # dlib HOG detector is generally more reliable, so higher base confidence
            confidence = 0.9
            
            # Adjust based on face size (prefer medium-sized faces)
            face_area = w * h
            image_area = image_shape[0] * image_shape[1]
            face_ratio = face_area / image_area
            
            if 0.01 <= face_ratio <= 0.3:  # Good face size
                confidence += 0.05
            elif face_ratio < 0.01:  # Too small
                confidence -= 0.1
            elif face_ratio > 0.3:  # Too large
                confidence -= 0.05
            
            # Adjust based on position (prefer center faces)
            center_x = x + w/2
            center_y = y + h/2
            image_center_x = image_shape[1] / 2
            image_center_y = image_shape[0] / 2
            
            distance_from_center = ((center_x - image_center_x)**2 + (center_y - image_center_y)**2)**0.5
            max_distance = ((image_shape[1]/2)**2 + (image_shape[0]/2)**2)**0.5
            
            if distance_from_center < max_distance * 0.3:  # Close to center
                confidence += 0.05
            elif distance_from_center > max_distance * 0.7:  # Far from center
                confidence -= 0.05
            
            # Ensure confidence is between 0.1 and 1.0
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating dlib confidence: {e}")
            return 0.9  # Default confidence for dlib
    
    def _calculate_simple_confidence(self, x: int, y: int, w: int, h: int, image_shape: tuple) -> float:
        """Calculate simple confidence score for OpenCV-detected face."""
        try:
            # Base confidence
            confidence = 0.8
            
            # Adjust based on face size (prefer medium-sized faces)
            face_area = w * h
            image_area = image_shape[0] * image_shape[1]
            face_ratio = face_area / image_area
            
            if 0.01 <= face_ratio <= 0.3:  # Good face size
                confidence += 0.1
            elif face_ratio < 0.01:  # Too small
                confidence -= 0.1
            elif face_ratio > 0.3:  # Too large
                confidence -= 0.05
            
            # Adjust based on position (prefer center faces)
            center_x = x + w/2
            center_y = y + h/2
            image_center_x = image_shape[1] / 2
            image_center_y = image_shape[0] / 2
            
            distance_from_center = ((center_x - image_center_x)**2 + (center_y - image_center_y)**2)**0.5
            max_distance = ((image_shape[1]/2)**2 + (image_shape[0]/2)**2)**0.5
            
            if distance_from_center < max_distance * 0.3:  # Close to center
                confidence += 0.05
            elif distance_from_center > max_distance * 0.7:  # Far from center
                confidence -= 0.05
            
            # Ensure confidence is between 0.1 and 1.0
            return max(0.1, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.8  # Default confidence
    
    def cleanup(self):
        """Clean up resources."""
        # dlib detector doesn't need explicit cleanup
        # OpenCV cascade doesn't need explicit cleanup either
        pass

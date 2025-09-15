"""
Facial Movement Detection Module for Advanced Liveness Detection.
Detects specific facial movements and validates direction for enhanced security.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from .face_detection import FaceDetector
from .utils import calculate_optical_flow

logger = logging.getLogger(__name__)

class FacialMovementDetector:
    """
    Advanced facial movement detector for liveness verification.
    Detects specific movements like blinking, smiling, head turns, and eyebrow raises.
    """
    
    def __init__(self):
        self.face_detector = FaceDetector()
        self.movement_history = []
        self.max_history = 30  # 1 second at 30fps
        self.challenge_types = {
            'blink': {
                'name': 'Blink Challenge',
                'description': 'Blink your eyes twice within 3 seconds',
                'icon': '👁️',
                'detection_method': 'eye_closure'
            },
            'smile': {
                'name': 'Smile Challenge',
                'description': 'Smile naturally and hold for 2 seconds',
                'icon': '😊',
                'detection_method': 'mouth_expression'
            },
            'head_left': {
                'name': 'Head Left Challenge',
                'description': 'Turn your head to the left',
                'icon': '⬅️',
                'detection_method': 'head_rotation',
                'direction': 'left'
            },
            'head_right': {
                'name': 'Head Right Challenge',
                'description': 'Turn your head to the right',
                'icon': '➡️',
                'detection_method': 'head_rotation',
                'direction': 'right'
            },
            'head_up': {
                'name': 'Head Up Challenge',
                'description': 'Tilt your head upward',
                'icon': '⬆️',
                'detection_method': 'head_tilt',
                'direction': 'up'
            },
            'head_down': {
                'name': 'Head Down Challenge',
                'description': 'Tilt your head downward',
                'icon': '⬇️',
                'detection_method': 'head_tilt',
                'direction': 'down'
            },
            'eyebrow_raise': {
                'name': 'Eyebrow Raise Challenge',
                'description': 'Raise your eyebrows and hold for 2 seconds',
                'icon': '🤨',
                'detection_method': 'eyebrow_movement'
            }
        }
        # Track recent landmark availability to decide which challenges to offer
        self._recent_landmarks_window = 5
        
    def add_frame(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                  landmarks: Optional[List] = None) -> None:
        """Add frame to movement history for analysis."""
        try:
            if len(self.movement_history) >= self.max_history:
                self.movement_history.pop(0)
            
            # Extract face region
            x, y, w, h = face_bbox
            face_region = frame[y:y+h, x:x+w]
            
            # Get landmarks if not provided
            if landmarks is None:
                landmarks = self.face_detector.extract_landmarks(frame, face_bbox)
            
            self.movement_history.append({
                'frame': face_region.copy(),
                'bbox': face_bbox,
                'landmarks': landmarks,
                'timestamp': len(self.movement_history)
            })
            
        except Exception as e:
            logger.error(f"Error adding frame to movement history: {str(e)}")
    
    def get_random_challenge(self) -> Dict:
        """Get a random movement challenge."""
        import random
        # Prefer head movement challenges when facial landmarks are not available recently
        available = list(self.challenge_types.keys())
        if not self._landmarks_available_recently():
            available = [c for c in available if c.startswith('head_')]
            # Fallback to head movements only
            if not available:
                available = ['head_left', 'head_right', 'head_up', 'head_down']
        challenge_type = random.choice(available)
        return {
            'type': challenge_type,
            **self.challenge_types[challenge_type]
        }

    def _landmarks_available_recently(self) -> bool:
        """Check if valid landmarks were seen in the recent history window."""
        try:
            if not self.movement_history:
                return False
            recent = self.movement_history[-self._recent_landmarks_window:]
            for entry in recent:
                lm = entry.get('landmarks')
                if lm is not None and len(lm) >= 68:
                    return True
            return False
        except Exception:
            return False
    
    def validate_movement(self, challenge_type: str, frame: np.ndarray, 
                         face_bbox: Tuple[int, int, int, int], 
                         landmarks: Optional[List] = None) -> Dict:
        """
        Validate if the user performed the requested movement correctly.
        
        Args:
            challenge_type: Type of challenge to validate
            frame: Current video frame
            face_bbox: Face bounding box
            landmarks: Facial landmarks
            
        Returns:
            Dict with validation results
        """
        try:
            if challenge_type not in self.challenge_types:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Unknown challenge type',
                    'details': {}
                }
            
            # Get landmarks if not provided
            if landmarks is None:
                landmarks = self.face_detector.extract_landmarks(frame, face_bbox)
            
            if landmarks is None or len(landmarks) < 68:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Insufficient facial landmarks detected',
                    'details': {}
                }
            
            # Validate based on challenge type
            if challenge_type == 'blink':
                return self._validate_blink(frame, face_bbox, landmarks)
            elif challenge_type == 'smile':
                return self._validate_smile(frame, face_bbox, landmarks)
            elif challenge_type in ['head_left', 'head_right']:
                return self._validate_head_rotation(challenge_type, frame, face_bbox, landmarks)
            elif challenge_type in ['head_up', 'head_down']:
                return self._validate_head_tilt(challenge_type, frame, face_bbox, landmarks)
            elif challenge_type == 'eyebrow_raise':
                return self._validate_eyebrow_raise(frame, face_bbox, landmarks)
            else:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Challenge type not implemented',
                    'details': {}
                }
                
        except Exception as e:
            logger.error(f"Error validating movement {challenge_type}: {str(e)}")
            return {
                'valid': False,
                'confidence': 0.0,
                'message': f'Error: {str(e)}',
                'details': {}
            }
    
    def _validate_blink(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                        landmarks: List) -> Dict:
        """Validate blinking movement."""
        try:
            if len(self.movement_history) < 10:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Need more frames to detect blinking',
                    'details': {}
                }
            
            # Analyze eye closure over time
            eye_closure_scores = []
            
            for i in range(max(0, len(self.movement_history) - 10), len(self.movement_history)):
                hist_frame = self.movement_history[i]['frame']
                hist_landmarks = self.movement_history[i]['landmarks']
                
                if hist_landmarks and len(hist_landmarks) >= 68:
                    eye_score = self._calculate_eye_closure(hist_frame, hist_landmarks)
                    eye_closure_scores.append(eye_score)
            
            if len(eye_closure_scores) < 5:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Insufficient eye closure data',
                    'details': {}
                }
            
            # Detect blinking pattern
            blink_detected = self._detect_blink_pattern(eye_closure_scores)
            
            if blink_detected:
                return {
                    'valid': True,
                    'confidence': 0.85,
                    'message': 'Blink challenge completed successfully!',
                    'details': {
                        'blink_count': 2,
                        'eye_closure_pattern': eye_closure_scores[-5:],
                        'movement_type': 'blink'
                    }
                }
            else:
                return {
                    'valid': False,
                    'confidence': 0.3,
                    'message': 'Please blink your eyes twice naturally',
                    'details': {
                        'eye_closure_pattern': eye_closure_scores[-5:],
                        'movement_type': 'blink'
                    }
                }
                
        except Exception as e:
            logger.error(f"Error validating blink: {str(e)}")
            return {
                'valid': False,
                'confidence': 0.0,
                'message': f'Blink validation error: {str(e)}',
                'details': {}
            }
    
    def _validate_smile(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                        landmarks: List) -> Dict:
        """Validate smiling movement."""
        try:
            if len(self.movement_history) < 5:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Need more frames to detect smiling',
                    'details': {}
                }
            
            # Analyze mouth expression over time
            smile_scores = []
            
            for i in range(max(0, len(self.movement_history) - 5), len(self.movement_history)):
                hist_frame = self.movement_history[i]['frame']
                hist_landmarks = self.movement_history[i]['landmarks']
                
                if hist_landmarks and len(hist_landmarks) >= 68:
                    smile_score = self._calculate_smile_score(hist_frame, hist_landmarks)
                    smile_scores.append(smile_score)
            
            if len(smile_scores) < 3:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Insufficient smile data',
                    'details': {}
                }
            
            # Check if smile is sustained
            recent_scores = smile_scores[-3:]
            avg_smile = np.mean(recent_scores)
            
            if avg_smile > 0.6:  # Threshold for smile detection
                return {
                    'valid': True,
                    'confidence': 0.8,
                    'message': 'Smile challenge completed successfully!',
                    'details': {
                        'smile_score': avg_smile,
                        'smile_pattern': smile_scores[-5:],
                        'movement_type': 'smile'
                    }
                }
            else:
                return {
                    'valid': False,
                    'confidence': 0.4,
                    'message': 'Please smile naturally and hold for 2 seconds',
                    'details': {
                        'smile_score': avg_smile,
                        'smile_pattern': smile_scores[-5:],
                        'movement_type': 'smile'
                    }
                }
                
        except Exception as e:
            logger.error(f"Error validating smile: {str(e)}")
            return {
                'valid': False,
                'confidence': 0.0,
                'message': f'Smile validation error: {str(e)}',
                'details': {}
            }
    
    def _validate_head_rotation(self, challenge_type: str, frame: np.ndarray, 
                                face_bbox: Tuple[int, int, int, int], 
                                landmarks: List) -> Dict:
        """Validate head rotation (left/right)."""
        try:
            if len(self.movement_history) < 8:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Need more frames to detect head rotation',
                    'details': {}
                }
            
            # Analyze head rotation over time
            rotation_scores = []
            
            # Build rotation scores using landmarks when available, otherwise bbox motion
            history_range_start = max(1, len(self.movement_history) - 8)
            for i in range(history_range_start, len(self.movement_history)):
                prev_entry = self.movement_history[i-1]
                curr_entry = self.movement_history[i]
                hist_landmarks = curr_entry.get('landmarks')
                
                if hist_landmarks and len(hist_landmarks) >= 68:
                    rotation_score = self._calculate_head_rotation(curr_entry['frame'], hist_landmarks)
                else:
                    rotation_score = self._calculate_head_rotation_from_bbox(prev_entry['bbox'], curr_entry['bbox'])
                rotation_scores.append(rotation_score)
            
            if len(rotation_scores) < 5:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Insufficient head rotation data',
                    'details': {}
                }
            
            # Check rotation direction and magnitude
            expected_direction = self.challenge_types[challenge_type]['direction']
            recent_rotations = rotation_scores[-5:]
            
            # Calculate rotation trend
            rotation_trend = np.mean(recent_rotations[-3:]) - np.mean(recent_rotations[:2])
            
            # Validate direction
            if expected_direction == 'left' and rotation_trend < -0.1:
                direction_correct = True
                message = 'Head left challenge completed successfully!'
            elif expected_direction == 'right' and rotation_trend > 0.1:
                direction_correct = True
                message = 'Head right challenge completed successfully!'
            else:
                direction_correct = False
                if expected_direction == 'left':
                    message = 'Please turn your head to the LEFT (not right)'
                else:
                    message = 'Please turn your head to the RIGHT (not left)'
            
            # Check if movement is significant enough
            movement_magnitude = abs(rotation_trend)
            if movement_magnitude < 0.05:
                message = f'Please turn your head more {expected_direction}'
                direction_correct = False
            
            return {
                'valid': direction_correct and movement_magnitude >= 0.05,
                'confidence': 0.75 if direction_correct else 0.3,
                'message': message,
                'details': {
                    'expected_direction': expected_direction,
                    'actual_rotation': rotation_trend,
                    'movement_magnitude': movement_magnitude,
                    'rotation_pattern': rotation_scores[-5:],
                    'movement_type': 'head_rotation'
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating head rotation: {str(e)}")
            return {
                'valid': False,
                'confidence': 0.0,
                'message': f'Head rotation validation error: {str(e)}',
                'details': {}
            }
    
    def _validate_head_tilt(self, challenge_type: str, frame: np.ndarray, 
                            face_bbox: Tuple[int, int, int, int], 
                            landmarks: List) -> Dict:
        """Validate head tilt (up/down)."""
        try:
            if len(self.movement_history) < 8:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Need more frames to detect head tilt',
                    'details': {}
                }
            
            # Analyze head tilt over time
            tilt_scores = []
            
            # Build tilt scores using landmarks when available, otherwise bbox motion
            history_range_start = max(1, len(self.movement_history) - 8)
            for i in range(history_range_start, len(self.movement_history)):
                prev_entry = self.movement_history[i-1]
                curr_entry = self.movement_history[i]
                hist_landmarks = curr_entry.get('landmarks')
                
                if hist_landmarks and len(hist_landmarks) >= 68:
                    tilt_score = self._calculate_head_tilt(curr_entry['frame'], hist_landmarks)
                else:
                    tilt_score = self._calculate_head_tilt_from_bbox(prev_entry['bbox'], curr_entry['bbox'])
                tilt_scores.append(tilt_score)
            
            if len(tilt_scores) < 5:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Insufficient head tilt data',
                    'details': {}
                }
            
            # Check tilt direction and magnitude
            expected_direction = self.challenge_types[challenge_type]['direction']
            recent_tilts = tilt_scores[-5:]
            
            # Calculate tilt trend
            tilt_trend = np.mean(recent_tilts[-3:]) - np.mean(recent_tilts[:2])
            
            # Validate direction
            if expected_direction == 'up' and tilt_trend < -0.1:
                direction_correct = True
                message = 'Head up challenge completed successfully!'
            elif expected_direction == 'down' and tilt_trend > 0.1:
                direction_correct = True
                message = 'Head down challenge completed successfully!'
            else:
                direction_correct = False
                if expected_direction == 'up':
                    message = 'Please tilt your head UPWARD (not down)'
                else:
                    message = 'Please tilt your head DOWNWARD (not up)'
            
            # Check if movement is significant enough
            movement_magnitude = abs(tilt_trend)
            if movement_magnitude < 0.05:
                message = f'Please tilt your head more {expected_direction}'
                direction_correct = False
            
            return {
                'valid': direction_correct and movement_magnitude >= 0.05,
                'confidence': 0.75 if direction_correct else 0.3,
                'message': message,
                'details': {
                    'expected_direction': expected_direction,
                    'actual_tilt': tilt_trend,
                    'movement_magnitude': movement_magnitude,
                    'tilt_pattern': tilt_scores[-5:],
                    'movement_type': 'head_tilt'
                }
            }
            
        except Exception as e:
            logger.error(f"Error validating head tilt: {str(e)}")
            return {
                'valid': False,
                'confidence': 0.0,
                'message': f'Head tilt validation error: {str(e)}',
                'details': {}
            }
    
    def _validate_eyebrow_raise(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                                landmarks: List) -> Dict:
        """Validate eyebrow raise movement."""
        try:
            if len(self.movement_history) < 5:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Need more frames to detect eyebrow movement',
                    'details': {}
                }
            
            # Analyze eyebrow position over time
            eyebrow_scores = []
            
            for i in range(max(0, len(self.movement_history) - 5), len(self.movement_history)):
                hist_frame = self.movement_history[i]['frame']
                hist_landmarks = self.movement_history[i]['landmarks']
                
                if hist_landmarks and len(hist_landmarks) >= 68:
                    eyebrow_score = self._calculate_eyebrow_position(hist_frame, hist_landmarks)
                    eyebrow_scores.append(eyebrow_score)
            
            if len(eyebrow_scores) < 3:
                return {
                    'valid': False,
                    'confidence': 0.0,
                    'message': 'Insufficient eyebrow movement data',
                    'details': {}
                }
            
            # Check if eyebrows are raised
            recent_scores = eyebrow_scores[-3:]
            avg_position = np.mean(recent_scores)
            
            if avg_position > 0.6:  # Threshold for raised eyebrows
                return {
                    'valid': True,
                    'confidence': 0.8,
                    'message': 'Eyebrow raise challenge completed successfully!',
                    'details': {
                        'eyebrow_position': avg_position,
                        'eyebrow_pattern': eyebrow_scores[-5:],
                        'movement_type': 'eyebrow_raise'
                    }
                }
            else:
                return {
                    'valid': False,
                    'confidence': 0.4,
                    'message': 'Please raise your eyebrows and hold for 2 seconds',
                    'details': {
                        'eyebrow_position': avg_position,
                        'eyebrow_pattern': eyebrow_scores[-5:],
                        'movement_type': 'eyebrow_raise'
                    }
                }
                
        except Exception as e:
            logger.error(f"Error validating eyebrow raise: {str(e)}")
            return {
                'valid': False,
                'confidence': 0.0,
                'message': f'Eyebrow raise validation error: {str(e)}',
                'details': {}
            }
    
    def _calculate_eye_closure(self, frame: np.ndarray, landmarks: List) -> float:
        """Calculate eye closure score (0 = open, 1 = closed)."""
        try:
            # Extract eye landmarks (simplified - in real implementation use proper eye region)
            # This is a simplified calculation - real implementation would use more sophisticated methods
            
            # Get eye region dimensions
            eye_width = abs(landmarks[36][0] - landmarks[39][0])  # Left eye width
            eye_height = abs(landmarks[37][1] - landmarks[41][1])  # Left eye height
            
            # Calculate eye aspect ratio (EAR)
            if eye_width > 0:
                ear = eye_height / eye_width
                # Normalize to 0-1 range (0 = open, 1 = closed)
                closure_score = max(0, min(1, (0.3 - ear) / 0.3))
                return closure_score
            else:
                return 0.5  # Default value if landmarks are invalid
                
        except Exception as e:
            logger.error(f"Error calculating eye closure: {str(e)}")
            return 0.5
    
    def _calculate_smile_score(self, frame: np.ndarray, landmarks: List) -> float:
        """Calculate smile score (0 = neutral, 1 = big smile)."""
        try:
            # Extract mouth landmarks
            mouth_width = abs(landmarks[48][0] - landmarks[54][0])  # Mouth width
            mouth_height = abs(landmarks[51][1] - landmarks[57][1])  # Mouth height
            
            # Calculate smile ratio
            if mouth_height > 0:
                smile_ratio = mouth_width / mouth_height
                # Normalize to 0-1 range
                smile_score = max(0, min(1, (smile_ratio - 1.5) / 1.5))
                return smile_score
            else:
                return 0.5
                
        except Exception as e:
            logger.error(f"Error calculating smile score: {str(e)}")
            return 0.5
    
    def _calculate_head_rotation(self, frame: np.ndarray, landmarks: List) -> float:
        """Calculate head rotation score (-1 = left, 0 = center, 1 = right)."""
        try:
            # Use nose and eye positions to estimate head rotation
            nose_x = landmarks[30][0]
            left_eye_x = landmarks[36][0]
            right_eye_x = landmarks[45][0]
            
            # Calculate relative positions
            eye_center_x = (left_eye_x + right_eye_x) / 2
            rotation_offset = (nose_x - eye_center_x) / 50.0  # Normalize
            
            # Clamp to -1 to 1 range
            return max(-1, min(1, rotation_offset))
            
        except Exception as e:
            logger.error(f"Error calculating head rotation: {str(e)}")
            return 0.0
    
    def _calculate_head_tilt(self, frame: np.ndarray, landmarks: List) -> float:
        """Calculate head tilt score (-1 = up, 0 = level, 1 = down)."""
        try:
            # Use eye positions to estimate head tilt
            left_eye_y = landmarks[37][1]
            right_eye_y = landmarks[44][1]
            
            # Calculate tilt angle
            tilt_offset = (left_eye_y - right_eye_y) / 30.0  # Normalize
            
            # Clamp to -1 to 1 range
            return max(-1, min(1, tilt_offset))
            
        except Exception as e:
            logger.error(f"Error calculating head tilt: {str(e)}")
            return 0.0

    def _calculate_head_rotation_from_bbox(self, prev_bbox: Tuple[int, int, int, int], current_bbox: Tuple[int, int, int, int]) -> float:
        """Approximate head rotation using bbox center horizontal motion across frames.
        Negative = left, Positive = right."""
        try:
            px, py, pw, ph = prev_bbox
            cx, cy, cw, ch = current_bbox
            prev_center_x = px + pw / 2.0
            curr_center_x = cx + cw / 2.0
            # Normalize by average width to prevent scale issues
            norm = max(1.0, (pw + cw) / 2.0)
            delta = (curr_center_x - prev_center_x) / norm
            # Clamp to -1..1 range
            return max(-1.0, min(1.0, float(delta)))
        except Exception as e:
            logger.error(f"Error calculating bbox-based head rotation: {str(e)}")
            return 0.0

    def _calculate_head_tilt_from_bbox(self, prev_bbox: Tuple[int, int, int, int], current_bbox: Tuple[int, int, int, int]) -> float:
        """Approximate head tilt using bbox center vertical motion across frames.
        Negative = up, Positive = down."""
        try:
            px, py, pw, ph = prev_bbox
            cx, cy, cw, ch = current_bbox
            prev_center_y = py + ph / 2.0
            curr_center_y = cy + ch / 2.0
            # Normalize by average height
            norm = max(1.0, (ph + ch) / 2.0)
            delta = (curr_center_y - prev_center_y) / norm
            # Clamp to -1..1 range
            return max(-1.0, min(1.0, float(delta)))
        except Exception as e:
            logger.error(f"Error calculating bbox-based head tilt: {str(e)}")
            return 0.0
    
    def _calculate_eyebrow_position(self, frame: np.ndarray, landmarks: List) -> float:
        """Calculate eyebrow position score (0 = normal, 1 = raised)."""
        try:
            # Use eyebrow landmarks to estimate position
            eyebrow_y = (landmarks[19][1] + landmarks[24][1]) / 2  # Average eyebrow Y position
            eye_y = (landmarks[37][1] + landmarks[44][1]) / 2  # Average eye Y position
            
            # Calculate relative position
            eyebrow_offset = (eye_y - eyebrow_y) / 20.0  # Normalize
            
            # Clamp to 0-1 range
            return max(0, min(1, eyebrow_offset))
            
        except Exception as e:
            logger.error(f"Error calculating eyebrow position: {str(e)}")
            return 0.5
    
    def _detect_blink_pattern(self, eye_closure_scores: List[float]) -> bool:
        """Detect blinking pattern from eye closure scores."""
        try:
            if len(eye_closure_scores) < 5:
                return False
            
            # Look for pattern: open -> closed -> open -> closed -> open
            blink_count = 0
            threshold = 0.6  # Threshold for eye closure
            
            for i in range(1, len(eye_closure_scores)):
                prev_score = eye_closure_scores[i-1]
                curr_score = eye_closure_scores[i]
                
                # Detect transition from open to closed
                if prev_score < threshold and curr_score >= threshold:
                    blink_count += 1
            
            # Require at least 2 blinks
            return blink_count >= 2
            
        except Exception as e:
            logger.error(f"Error detecting blink pattern: {str(e)}")
            return False
    
    def get_movement_summary(self) -> Dict:
        """Get a summary of detected movements."""
        try:
            if not self.movement_history:
                return {
                    'movements_detected': [],
                    'total_frames': 0,
                    'analysis_ready': False,
                    'message': 'No movement data available'
                }
            
            # Analyze recent movements
            recent_movements = []
            
            if len(self.movement_history) >= 5:
                # Check for various movement types
                for challenge_type in self.challenge_types:
                    validation = self.validate_movement(
                        challenge_type,
                        self.movement_history[-1]['frame'],
                        self.movement_history[-1]['bbox'],
                        self.movement_history[-1]['landmarks']
                    )
                    
                    if validation['valid']:
                        recent_movements.append({
                            'type': challenge_type,
                            'confidence': validation['confidence'],
                            'message': validation['message']
                        })
            
            return {
                'movements_detected': recent_movements,
                'total_frames': len(self.movement_history),
                'analysis_ready': len(self.movement_history) >= 10,
                'message': f'Detected {len(recent_movements)} movements'
            }
            
        except Exception as e:
            logger.error(f"Error getting movement summary: {str(e)}")
            return {
                'movements_detected': [],
                'total_frames': 0,
                'analysis_ready': False,
                'message': f'Error: {str(e)}'
            }
    
    def clear_history(self) -> None:
        """Clear movement history."""
        self.movement_history.clear()

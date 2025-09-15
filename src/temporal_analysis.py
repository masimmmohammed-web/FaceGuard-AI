"""
Temporal analysis module for micro-expressions and temporal consistency analysis.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from .utils import calculate_optical_flow

logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """Analyzes temporal patterns and micro-expressions for liveness detection."""
    
    def __init__(self):
        self.frame_buffer = []
        self.max_buffer_size = 60  # 2 seconds at 30fps
        self.min_frames_for_analysis = 10
        
    def add_frame(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int]):
        """Add frame and face bounding box to buffer."""
        if len(self.frame_buffer) >= self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        self.frame_buffer.append({
            'frame': frame.copy(),
            'bbox': face_bbox,
            'timestamp': len(self.frame_buffer)
        })
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns for natural vs artificial behavior."""
        try:
            if len(self.frame_buffer) < self.min_frames_for_analysis:
                return {
                    'is_natural': False,
                    'confidence': 0.0,
                    'reason': 'Insufficient frames for analysis'
                }
            
            # Analyze various temporal patterns
            movement_analysis = self._analyze_movement_patterns()
            expression_analysis = self._analyze_expression_patterns()
            consistency_analysis = self._analyze_temporal_consistency()
            
            # Combine results
            natural_score = 0.0
            total_weight = 0.0
            
            if movement_analysis['is_natural']:
                natural_score += movement_analysis['confidence'] * 0.4
                total_weight += 0.4
            
            if expression_analysis['is_natural']:
                natural_score += expression_analysis['confidence'] * 0.3
                total_weight += 0.3
            
            if consistency_analysis['is_natural']:
                natural_score += consistency_analysis['confidence'] * 0.3
                total_weight += 0.3
            
            if total_weight > 0:
                final_score = natural_score / total_weight
                is_natural = final_score > 0.6
            else:
                final_score = 0.0
                is_natural = False
            
            return {
                'is_natural': is_natural,
                'confidence': final_score,
                'movement_analysis': movement_analysis,
                'expression_analysis': expression_analysis,
                'consistency_analysis': consistency_analysis,
                'overall_assessment': 'Natural' if is_natural else 'Artificial'
            }
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {str(e)}")
            return {'is_natural': False, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _analyze_movement_patterns(self) -> Dict:
        """Analyze natural movement patterns."""
        try:
            if len(self.frame_buffer) < 5:
                return {'is_natural': False, 'confidence': 0.0, 'reason': 'Insufficient frames'}
            
            movements = []
            for i in range(1, len(self.frame_buffer)):
                prev_frame = self.frame_buffer[i-1]
                curr_frame = self.frame_buffer[i]
                
                # Extract face regions
                prev_bbox = prev_frame['bbox']
                curr_bbox = curr_frame['bbox']
                
                # Calculate face movement
                prev_center = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
                curr_center = (curr_bbox[0] + curr_bbox[2]//2, curr_bbox[1] + curr_bbox[3]//2)
                
                movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                movements.append(movement)
            
            if len(movements) < 2:
                return {'is_natural': False, 'confidence': 0.0, 'reason': 'Insufficient movement data'}
            
            # Analyze movement characteristics
            mean_movement = np.mean(movements)
            movement_variance = np.var(movements)
            movement_range = max(movements) - min(movements)
            
            # Natural movement criteria
            is_natural = True
            confidence = 0.0
            reasons = []
            
            # Check for reasonable movement range
            if mean_movement < 0.5:
                is_natural = False
                reasons.append('Too static')
            elif mean_movement > 50:
                is_natural = False
                reasons.append('Too much movement')
            else:
                confidence += 0.3
            
            # Check for natural variation
            if 0.1 < movement_variance < 100:
                confidence += 0.4
            else:
                is_natural = False
                reasons.append('Unnatural movement variation')
            
            # Check for smooth transitions
            if movement_range < 100:
                confidence += 0.3
            else:
                is_natural = False
                reasons.append('Erratic movement')
            
            return {
                'is_natural': is_natural,
                'confidence': confidence,
                'mean_movement': mean_movement,
                'movement_variance': movement_variance,
                'movement_range': movement_range,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Error in movement pattern analysis: {str(e)}")
            return {'is_natural': False, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _analyze_expression_patterns(self) -> Dict:
        """Analyze natural expression patterns and micro-expressions."""
        try:
            if len(self.frame_buffer) < 10:
                return {'is_natural': False, 'confidence': 0.0, 'reason': 'Insufficient frames'}
            
            # Analyze facial expressions over time
            expression_changes = []
            
            for i in range(1, len(self.frame_buffer)):
                prev_frame = self.frame_buffer[i-1]
                curr_frame = self.frame_buffer[i]
                
                # Calculate expression change using optical flow
                prev_face = self._extract_face_region(prev_frame)
                curr_face = self._extract_face_region(curr_frame)
                
                if prev_face is not None and curr_face is not None:
                    flow = calculate_optical_flow(prev_face, curr_face)
                    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    expression_change = np.mean(magnitude)
                    expression_changes.append(expression_change)
            
            if len(expression_changes) < 3:
                return {'is_natural': False, 'confidence': 0.0, 'reason': 'Insufficient expression data'}
            
            # Analyze expression patterns
            mean_change = np.mean(expression_changes)
            change_variance = np.var(expression_changes)
            
            # Natural expression criteria
            is_natural = True
            confidence = 0.0
            reasons = []
            
            # Check for reasonable expression changes
            if 0.1 < mean_change < 5.0:
                confidence += 0.4
            else:
                is_natural = False
                reasons.append('Unnatural expression intensity')
            
            # Check for natural variation
            if 0.01 < change_variance < 10.0:
                confidence += 0.3
            else:
                is_natural = False
                reasons.append('Unnatural expression variation')
            
            # Check for micro-expression patterns
            micro_expressions = self._detect_micro_expressions(expression_changes)
            if micro_expressions:
                confidence += 0.3
            else:
                reasons.append('No micro-expressions detected')
            
            return {
                'is_natural': is_natural,
                'confidence': confidence,
                'mean_expression_change': mean_change,
                'expression_variance': change_variance,
                'micro_expressions_detected': micro_expressions,
                'reasons': reasons
            }
            
        except Exception as e:
            logger.error(f"Error in expression pattern analysis: {str(e)}")
            return {'is_natural': False, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _analyze_temporal_consistency(self) -> Dict:
        """Analyze temporal consistency for detecting artificial patterns."""
        try:
            if len(self.frame_buffer) < 15:
                return {'is_natural': False, 'confidence': 0.0, 'reason': 'Insufficient frames'}
            
            # Analyze consistency across different time scales
            short_term_consistency = self._analyze_short_term_consistency()
            long_term_consistency = self._analyze_long_term_consistency()
            
            # Combine consistency measures
            is_natural = short_term_consistency['is_natural'] and long_term_consistency['is_natural']
            confidence = (short_term_consistency['confidence'] + long_term_consistency['confidence']) / 2
            
            return {
                'is_natural': is_natural,
                'confidence': confidence,
                'short_term': short_term_consistency,
                'long_term': long_term_consistency
            }
            
        except Exception as e:
            logger.error(f"Error in temporal consistency analysis: {str(e)}")
            return {'is_natural': False, 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _extract_face_region(self, frame_data: Dict) -> Optional[np.ndarray]:
        """Extract face region from frame data."""
        try:
            frame = frame_data['frame']
            bbox = frame_data['bbox']
            
            x, y, w, h = bbox
            # Ensure coordinates are within bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame.shape[1] - x)
            h = min(h, frame.shape[0] - y)
            
            if w <= 0 or h <= 0:
                return None
            
            return frame[y:y+h, x:x+w]
            
        except Exception as e:
            logger.error(f"Error extracting face region: {str(e)}")
            return None
    
    def _detect_micro_expressions(self, expression_changes: List[float]) -> bool:
        """Detect subtle micro-expressions."""
        try:
            if len(expression_changes) < 5:
                return False
            
            # Look for subtle, rapid changes that indicate micro-expressions
            micro_expression_count = 0
            
            for i in range(1, len(expression_changes)):
                change = abs(expression_changes[i] - expression_changes[i-1])
                
                # Micro-expressions are small but noticeable changes
                if 0.1 < change < 1.0:
                    micro_expression_count += 1
            
            # Require at least 2 micro-expressions for natural behavior
            return micro_expression_count >= 2
            
        except Exception as e:
            logger.error(f"Error detecting micro-expressions: {str(e)}")
            return False
    
    def _analyze_short_term_consistency(self) -> Dict:
        """Analyze consistency over short time periods."""
        try:
            if len(self.frame_buffer) < 5:
                return {'is_natural': False, 'confidence': 0.0}
            
            # Analyze consistency over 5-frame windows
            consistencies = []
            
            for i in range(0, len(self.frame_buffer) - 4, 2):
                window = self.frame_buffer[i:i+5]
                
                # Calculate consistency within window
                if len(window) >= 3:
                    movements = []
                    for j in range(1, len(window)):
                        prev_bbox = window[j-1]['bbox']
                        curr_bbox = window[j]['bbox']
                        
                        prev_center = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
                        curr_center = (curr_bbox[0] + curr_bbox[2]//2, curr_bbox[1] + curr_bbox[3]//2)
                        
                        movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                        movements.append(movement)
                    
                    if movements:
                        consistency = 1.0 - np.std(movements) / (np.mean(movements) + 1e-10)
                        consistencies.append(consistency)
            
            if not consistencies:
                return {'is_natural': False, 'confidence': 0.0}
            
            # Natural behavior should have moderate consistency
            mean_consistency = np.mean(consistencies)
            is_natural = 0.3 < mean_consistency < 0.8
            
            return {
                'is_natural': is_natural,
                'confidence': mean_consistency,
                'consistency_score': mean_consistency
            }
            
        except Exception as e:
            logger.error(f"Error in short-term consistency analysis: {str(e)}")
            return {'is_natural': False, 'confidence': 0.0}
    
    def _analyze_long_term_consistency(self) -> Dict:
        """Analyze consistency over longer time periods."""
        try:
            if len(self.frame_buffer) < 15:
                return {'is_natural': False, 'confidence': 0.0}
            
            # Analyze consistency over the entire sequence
            movements = []
            for i in range(1, len(self.frame_buffer)):
                prev_bbox = self.frame_buffer[i-1]['bbox']
                curr_bbox = self.frame_buffer[i]['bbox']
                
                prev_center = (prev_bbox[0] + prev_bbox[2]//2, prev_bbox[1] + prev_bbox[3]//2)
                curr_center = (curr_bbox[0] + curr_bbox[2]//2, curr_bbox[1] + curr_bbox[3]//2)
                
                movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
                movements.append(movement)
            
            if not movements:
                return {'is_natural': False, 'confidence': 0.0}
            
            # Calculate long-term patterns
            movement_variance = np.var(movements)
            movement_trend = np.polyfit(range(len(movements)), movements, 1)[0]
            
            # Natural behavior should have moderate variance and no strong trends
            is_natural = (0.1 < movement_variance < 50.0 and abs(movement_trend) < 0.1)
            confidence = 1.0 - min(movement_variance / 50.0, 1.0)
            
            return {
                'is_natural': is_natural,
                'confidence': confidence,
                'movement_variance': movement_variance,
                'movement_trend': movement_trend
            }
            
        except Exception as e:
            logger.error(f"Error in long-term consistency analysis: {str(e)}")
            return {'is_natural': False, 'confidence': 0.0}
    
    def get_analysis_summary(self) -> Dict:
        """Get a summary of all temporal analysis results."""
        try:
            temporal_patterns = self.analyze_temporal_patterns()
            
            return {
                'temporal_analysis': temporal_patterns,
                'frame_count': len(self.frame_buffer),
                'analysis_ready': len(self.frame_buffer) >= self.min_frames_for_analysis,
                'recommendations': self._generate_recommendations(temporal_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {str(e)}")
            return {'error': str(e)}
    
    def _generate_recommendations(self, analysis_result: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if not analysis_result['is_natural']:
            if 'movement_analysis' in analysis_result:
                movement = analysis_result['movement_analysis']
                if 'Too static' in movement.get('reasons', []):
                    recommendations.append("Try moving your head slightly to demonstrate natural movement")
                elif 'Too much movement' in movement.get('reasons', []):
                    recommendations.append("Please reduce head movement for better analysis")
            
            if 'expression_analysis' in analysis_result:
                expression = analysis_result['expression_analysis']
                if 'No micro-expressions detected' in expression.get('reasons', []):
                    recommendations.append("Try blinking naturally to demonstrate liveness")
        
        if len(self.frame_buffer) < self.min_frames_for_analysis:
            recommendations.append("Please wait for more frames to complete analysis")
        
        if not recommendations:
            recommendations.append("Analysis complete - face appears natural")
        
        return recommendations

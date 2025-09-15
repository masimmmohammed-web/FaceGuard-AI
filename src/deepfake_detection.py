"""
Deepfake detection module using pre-trained models and CNN-based analysis.
Supports PyTorch (.pth), pickle (.pkl), and TensorFlow models.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
import os
from pathlib import Path
from .utils import preprocess_image, resize_image

logger = logging.getLogger(__name__)

class DeepfakeDetector:
    """Detects AI-generated deepfake faces using pre-trained models and multiple detection methods."""
    
    def __init__(self, model_path: str = None):
        self.texture_threshold = 0.7
        self.artifact_threshold = 0.6
        
        # Model loading
        self.model = None
        self.model_type = None
        self.model_path = None
        
        # Try to load pre-trained model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a pre-trained deepfake detection model."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            self.model_path = str(model_path)
            
            # Determine model type and load accordingly
            if model_path.suffix == '.pth':
                return self._load_pytorch_model(model_path)
            elif model_path.suffix == '.pkl':
                return self._load_pickle_model(model_path)
            elif model_path.suffix in ['.h5', '.pb']:
                return self._load_tensorflow_model(model_path)
            else:
                logger.warning(f"Unsupported model format: {model_path.suffix}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _load_pytorch_model(self, model_path: Path) -> bool:
        """Load PyTorch model (.pth file)."""
        try:
            import torch
            import torch.nn as nn
            
            # Load the model
            self.model = torch.load(model_path, map_location='cpu')
            self.model.eval()  # Set to evaluation mode
            self.model_type = 'pytorch'
            
            logger.info(f"PyTorch model loaded successfully: {model_path}")
            return True
            
        except ImportError:
            logger.error("PyTorch not available. Install with: pip install torch")
            return False
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {str(e)}")
            return False
    
    def _load_pickle_model(self, model_path: Path) -> bool:
        """Load pickle model (.pkl file)."""
        try:
            import pickle
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'pickle'
            
            logger.info(f"Pickle model loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading pickle model: {str(e)}")
            return False
    
    def _load_tensorflow_model(self, model_path: Path) -> bool:
        """Load TensorFlow model."""
        try:
            import tensorflow as tf
            
            if model_path.suffix == '.h5':
                self.model = tf.keras.models.load_model(model_path)
            else:
                self.model = tf.keras.models.load_model(str(model_path))
            
            self.model_type = 'tensorflow'
            logger.info(f"TensorFlow model loaded successfully: {model_path}")
            return True
            
        except ImportError:
            logger.error("TensorFlow not available. Install with: pip install tensorflow")
            return False
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {str(e)}")
            return False
    
    def detect_deepfake(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int]) -> Dict:
        """Comprehensive deepfake detection analysis using pre-trained model if available."""
        try:
            # Extract face region
            x, y, w, h = face_bbox
            face_roi = image[y:y+h, x:x+w]
            
            # Try pre-trained model first
            if self.model is not None:
                model_result = self._predict_with_model(face_roi)
                if model_result:
                    return model_result
            
            # Fallback to traditional analysis methods
            return self._traditional_analysis(face_roi)
            
        except Exception as e:
            logger.error(f"Error in deepfake detection: {str(e)}")
            return {'is_deepfake': False, 'confidence': 0.0, 'overall_risk': 0.0}
    
    def _predict_with_model(self, face_roi: np.ndarray) -> Optional[Dict]:
        """Make prediction using the loaded pre-trained model."""
        try:
            if self.model is None:
                return None
            
            # Preprocess image for model input
            processed_image = self._preprocess_for_model(face_roi)
            if processed_image is None:
                return None
            
            # Make prediction based on model type
            if self.model_type == 'pytorch':
                return self._pytorch_predict(processed_image)
            elif self.model_type == 'pickle':
                return self._pickle_predict(processed_image)
            elif self.model_type == 'tensorflow':
                return self._tensorflow_predict(processed_image)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            return None
    
    def _preprocess_for_model(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess face ROI for model input."""
        try:
            # Resize to standard size (224x224 for most models)
            resized = resize_image(face_roi, (224, 224))
            
            # Convert to RGB if needed
            if len(resized.shape) == 2:  # grayscale
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 1:  # single channel
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 3:  # already RGB/BGR
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def _pytorch_predict(self, processed_image: np.ndarray) -> Dict:
        """Make prediction using PyTorch model."""
        try:
            import torch
            
            # Convert to tensor
            input_tensor = torch.from_numpy(processed_image).unsqueeze(0)
            input_tensor = input_tensor.permute(0, 3, 1, 2)  # NHWC to NCHW
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
                
            # Handle different output formats
            if isinstance(output, torch.Tensor):
                if output.dim() == 2:
                    # Binary classification
                    score = torch.sigmoid(output[0, 0]).item()
                else:
                    # Single value
                    score = torch.sigmoid(output[0]).item()
            else:
                # Handle other output types
                score = float(output) if hasattr(output, 'item') else 0.5
            
            return {
                'is_deepfake': score > 0.5,
                'confidence': score,
                'overall_risk': score,
                'method': 'pytorch_model',
                'model_path': self.model_path
            }
            
        except Exception as e:
            logger.error(f"Error in PyTorch prediction: {str(e)}")
            return None
    
    def _pickle_predict(self, processed_image: np.ndarray) -> Dict:
        """Make prediction using pickle model."""
        try:
            # Try to use the model's predict method
            if hasattr(self.model, 'predict'):
                # Reshape for sklearn models
                input_data = processed_image.reshape(1, -1)
                prediction = self.model.predict(input_data)
                score = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
                
                # Convert to probability if needed
                if score in [0, 1]:  # Binary prediction
                    score = 0.9 if score == 1 else 0.1
                
            elif hasattr(self.model, 'predict_proba'):
                # Use predict_proba for probability
                input_data = processed_image.reshape(1, -1)
                proba = self.model.predict_proba(input_data)
                score = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
                
            else:
                # Fallback
                score = 0.5
            
            return {
                'is_deepfake': score > 0.5,
                'confidence': score,
                'overall_risk': score,
                'method': 'pickle_model',
                'model_path': self.model_path
            }
            
        except Exception as e:
            logger.error(f"Error in pickle model prediction: {str(e)}")
            return None
    
    def _tensorflow_predict(self, processed_image: np.ndarray) -> Dict:
        """Make prediction using TensorFlow model."""
        try:
            # Add batch dimension
            input_batch = np.expand_dims(processed_image, axis=0)
            
            # Make prediction
            prediction = self.model.predict(input_batch, verbose=0)
            
            # Extract score
            if isinstance(prediction, np.ndarray):
                score = float(prediction[0, 0]) if prediction.ndim > 1 else float(prediction[0])
            else:
                score = float(prediction)
            
            return {
                'is_deepfake': score > 0.5,
                'confidence': score,
                'overall_risk': score,
                'method': 'tensorflow_model',
                'model_path': self.model_path
            }
            
        except Exception as e:
            logger.error(f"Error in TensorFlow prediction: {str(e)}")
            return None
    
    def _traditional_analysis(self, face_roi: np.ndarray) -> Dict:
        """Traditional analysis methods as fallback."""
        # Run detection methods
        texture_result = self._analyze_texture_artifacts(face_roi)
        compression_result = self._analyze_compression_artifacts(face_roi)
        frequency_result = self._analyze_frequency_domain(face_roi)
        
        # Combine results
        is_deepfake = False
        confidence = 0.0
        
        # Weighted combination of detection methods
        total_score = 0.0
        total_weight = 0.0
        
        if texture_result['is_deepfake']:
            total_score += texture_result['confidence'] * 0.4
            total_weight += 0.4
        
        if compression_result['is_deepfake']:
            total_score += compression_result['confidence'] * 0.3
            total_weight += 0.3
        
        if frequency_result['is_deepfake']:
            total_score += frequency_result['confidence'] * 0.3
            total_weight += 0.3
        
        if total_weight > 0:
            confidence = total_score / total_weight
            is_deepfake = confidence > 0.5
        
        return {
            'is_deepfake': is_deepfake,
            'confidence': confidence,
            'texture_analysis': texture_result,
            'compression_analysis': compression_result,
            'frequency_analysis': frequency_result,
            'overall_risk': confidence,
            'method': 'traditional_analysis'
        }
    
    def _analyze_texture_artifacts(self, face_roi: np.ndarray) -> Dict:
        """Analyze texture artifacts that are common in deepfakes."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Pattern (LBP)
            lbp = self._calculate_lbp(gray)
            
            # Analyze LBP histogram
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            hist = hist.astype(np.float32) / hist.sum()
            
            # Calculate texture uniformity
            uniformity = np.sum(hist ** 2)
            
            # Deepfakes often have more uniform textures
            is_deepfake = uniformity > self.texture_threshold
            confidence = min(uniformity, 1.0)
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': confidence,
                'uniformity_score': uniformity,
                'method': 'texture_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {str(e)}")
            return {'is_deepfake': False, 'confidence': 0.0, 'method': 'texture_analysis'}
    
    def _analyze_compression_artifacts(self, face_roi: np.ndarray) -> Dict:
        """Analyze compression artifacts that may indicate deepfakes."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply DCT (Discrete Cosine Transform)
            dct = cv2.dct(np.float32(gray))
            
            # Analyze DCT coefficients
            dct_abs = np.abs(dct)
            
            # Calculate energy distribution
            total_energy = np.sum(dct_abs)
            low_freq_energy = np.sum(dct_abs[:8, :8])  # Low frequency components
            
            # Energy ratio (deepfakes often have different energy distributions)
            energy_ratio = low_freq_energy / total_energy if total_energy > 0 else 0
            
            # Calculate artifact score
            artifact_score = 1.0 - energy_ratio
            
            is_deepfake = artifact_score > self.artifact_threshold
            confidence = artifact_score
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': confidence,
                'artifact_score': artifact_score,
                'energy_ratio': energy_ratio,
                'method': 'compression_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in compression analysis: {str(e)}")
            return {'is_deepfake': False, 'confidence': 0.0, 'method': 'compression_analysis'}
    
    def _analyze_frequency_domain(self, face_roi: np.ndarray) -> Dict:
        """Analyze frequency domain characteristics."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply FFT
            fft = np.fft.fft2(gray)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.log(np.abs(fft_shift) + 1)
            
            # Calculate frequency distribution
            height, width = magnitude.shape
            center_y, center_x = height // 2, width // 2
            
            # Create frequency mask
            y, x = np.ogrid[:height, :width]
            mask = (x - center_x)**2 + (y - center_y)**2 <= min(height, width)**2 // 4
            
            # Calculate high vs low frequency energy
            low_freq_energy = np.sum(magnitude[mask])
            high_freq_energy = np.sum(magnitude[~mask])
            
            # Frequency ratio
            freq_ratio = high_freq_energy / (low_freq_energy + 1e-10)
            
            # Deepfakes often have different frequency characteristics
            is_deepfake = freq_ratio > 2.0 or freq_ratio < 0.3
            confidence = min(abs(freq_ratio - 1.0), 1.0)
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': confidence,
                'freq_ratio': freq_ratio,
                'method': 'frequency_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in frequency analysis: {str(e)}")
            return {'is_deepfake': False, 'confidence': 0.0, 'method': 'frequency_analysis'}
    
    def _calculate_lbp(self, gray_image: np.ndarray) -> np.ndarray:
        """Calculate Local Binary Pattern."""
        try:
            height, width = gray_image.shape
            lbp = np.zeros((height, width), dtype=np.uint8)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = gray_image[i, j]
                    code = 0
                    
                    # 8-neighbor LBP
                    code |= (gray_image[i-1, j-1] > center) << 7
                    code |= (gray_image[i-1, j] > center) << 6
                    code |= (gray_image[i-1, j+1] > center) << 5
                    code |= (gray_image[i, j+1] > center) << 4
                    code |= (gray_image[i+1, j+1] > center) << 3
                    code |= (gray_image[i+1, j] > center) << 2
                    code |= (gray_image[i+1, j-1] > center) << 1
                    code |= (gray_image[i, j-1] > center) << 0
                    
                    lbp[i, j] = code
            
            return lbp
            
        except Exception as e:
            logger.error(f"Error calculating LBP: {str(e)}")
            return np.zeros_like(gray_image)
    
    def analyze_multiple_frames(self, frames: list, face_bboxes: list) -> Dict:
        """Analyze multiple frames for temporal consistency."""
        try:
            if len(frames) != len(face_bboxes) or len(frames) < 2:
                return {'is_deepfake': False, 'confidence': 0.0, 'method': 'temporal_analysis'}
            
            # Analyze each frame
            results = []
            for frame, bbox in zip(frames, face_bboxes):
                result = self.detect_deepfake(frame, bbox)
                results.append(result)
            
            # Check for temporal consistency
            deepfake_scores = [r['confidence'] for r in results]
            temporal_consistency = np.std(deepfake_scores)
            
            # High consistency in deepfake scores may indicate real deepfake
            is_deepfake = temporal_consistency < 0.2 and np.mean(deepfake_scores) > 0.6
            confidence = np.mean(deepfake_scores)
            
            return {
                'is_deepfake': is_deepfake,
                'confidence': confidence,
                'temporal_consistency': temporal_consistency,
                'frame_results': results,
                'method': 'temporal_analysis'
            }
            
        except Exception as e:
            logger.error(f"Error in temporal analysis: {str(e)}")
            return {'is_deepfake': False, 'confidence': 0.0, 'method': 'temporal_analysis'}

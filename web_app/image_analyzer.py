"""
Dedicated Image Analyzer for FaceGuard AI
ONLY handles image uploads and analysis - nothing else!
"""

from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector
from src.anti_spoofing import AntiSpoofingDetector

# Create Blueprint for image analysis
image_analyzer = Blueprint('image_analyzer', __name__)

# Initialize detection modules
face_detector = FaceDetector()

# Always use the Hugging Face engine as the deepfake detector
from web_app.face_hf_engine import HFDeepfakeEngine
hf_engine = HFDeepfakeEngine()

anti_spoofing = AntiSpoofingDetector()

def convert_numpy_types(obj):
    """Convert NumPy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

@image_analyzer.route('/analyze_image', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for deepfake detection."""
    try:
        # Accept either multipart file upload ("image") OR JSON base64 ("frame_data")
        image = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'})
            image_bytes = file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            try:
                data_json = request.get_json(silent=True) or {}
                frame_data = data_json.get('frame_data')
                if frame_data and isinstance(frame_data, str) and ',' in frame_data:
                    base64_str = frame_data.split(',', 1)[1]
                    import base64 as _b64
                    frame_bytes = _b64.b64decode(base64_str)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    if 'image' not in request.files:
                        return jsonify({'error': 'No image provided (expected file "image" or JSON "frame_data")'})
            except Exception as _e:
                return jsonify({'error': f'Failed to parse image: {_e}'})
        
        if image is None:
            return jsonify({'error': 'Failed to decode image'})
        
        # Simple face detection
        faces = face_detector.detect_faces(image)
        
        if not faces:
            return jsonify({
                'success': False,
                'error': 'No faces detected in the image. Please upload a clear photo with a visible human face.',
                'face_detected': False,
                'assessment': 'NO_FACE',
                'overall_risk': None,
                'anti_spoofing': {'overall_risk': None, 'assessment': 'N/A'},
                'deepfake_detection': {'overall_risk': None, 'assessment': 'N/A'}
            })
        
        # Analyze first detected face
        face = faces[0]
        bbox = face['bbox']
        
        # Run analysis
        print(f"🔍 Running anti-spoofing analysis...")
        anti_spoofing_result = anti_spoofing.comprehensive_analysis(image, bbox, [])
        print(f"   Anti-spoofing result: {anti_spoofing_result}")
        
        print(f"🔍 Running deepfake detection...")
        deepfake_raw = hf_engine.score_frame(image, bbox)
        deepfake_result = {
            'overall_risk': float(deepfake_raw.get('overall_risk', 0.0)),
            'assessment': 'FAKE' if deepfake_raw.get('overall_risk', 0.0) >= 0.5 else 'REAL',
            'engine': 'hf'
        }
        deepfake_details = deepfake_raw.get('details', {})
        print(f"   Deepfake result: {deepfake_result}")
        
        # Get risk scores with simple defaults
        anti_spoofing_risk = anti_spoofing_result.get('overall_risk', 0.15)
        deepfake_risk = deepfake_result.get('overall_risk', 0.12)
        
        print(f"📊 Risk scores - Anti-spoofing: {anti_spoofing_risk:.4f}, Deepfake: {deepfake_risk:.4f}")
        
        # Simple risk calculation
        overall_risk = (anti_spoofing_risk * 0.6) + (deepfake_risk * 0.4)
        
        # Simple classification
        if overall_risk < 0.3:
            assessment = 'AUTHENTIC'
            is_authentic = True
        elif overall_risk < 0.6:
            assessment = 'SUSPICIOUS'
            is_authentic = False
        else:
            assessment = 'DEEPFAKE'
            is_authentic = False
        
        # Create results
        result = {
            'success': True,
            'face_detected': True,
            'bbox': bbox,
            'confidence': face.get('confidence', 0.95),
            'anti_spoofing': {
                'overall_risk': anti_spoofing_risk,
                'assessment': 'PASS' if anti_spoofing_risk < 0.5 else 'FAIL'
            },
            'deepfake_detection': {
                'overall_risk': deepfake_risk,
                'assessment': 'REAL' if deepfake_risk < 0.5 else 'FAKE'
            },
            'deepfake_details': deepfake_details,
            'overall_risk': overall_risk,
            'is_authentic': is_authentic,
            'assessment': assessment
        }
        
        # Convert NumPy types
        result = convert_numpy_types(result)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        return jsonify({'error': str(e)})

"""
Simple Flask router for FaceGuard AI - Clean and Focused
Each feature has its own dedicated handler file.
"""

from flask import Flask, render_template
import os
from flask_cors import CORS

# Import our dedicated analyzers
from image_analyzer import image_analyzer
from live_analyzer import live_analyzer
from face_registration import face_registration
from authentication import authentication
from restricted_pages import restricted_pages

app = Flask(__name__)
CORS(app)

# Load environment from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Register the blueprints
app.register_blueprint(image_analyzer)
app.register_blueprint(live_analyzer)
app.register_blueprint(face_registration)
app.register_blueprint(authentication)
app.register_blueprint(restricted_pages)

@app.route('/')
def index():
    """Main dashboard - just navigation."""
    return render_template('index.html')

@app.route('/analyze_image')
def image_analysis_page():
    """Image analysis page - handled by image_analyzer.py."""
    return render_template('image_analysis.html')

@app.route('/live_analysis')
def live_analysis_page():
    """Live analysis page - handled by live_analyzer.py."""
    return render_template('live_analysis.html')

@app.route('/batch_analysis')
def batch_analysis_page():
    """Batch analysis page - for future implementation."""
    return render_template('batch_analysis.html')

@app.route('/register')
def registration_page():
    """Face registration page."""
    return render_template('face_registration.html')

if __name__ == '__main__':
    # Configure Flask for better stability and resource management
    app.config.update(
        DEBUG=True,
        THREADED=True,  # Enable threading for better performance
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
        SEND_FILE_MAX_AGE_DEFAULT=0,  # Disable caching for development
        TEMPLATES_AUTO_RELOAD=True
    )
    
    # Run with better configuration for stability
    app.run(
        debug=True, 
        host='0.0.0.0', 
        port=5000,
        threaded=True,  # Enable threading
        use_reloader=False,  # Disable reloader to prevent memory issues
        use_debugger=True
    )

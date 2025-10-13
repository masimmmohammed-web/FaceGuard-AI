"""
Secure Flask router for FaceGuard AI
Each feature has its own dedicated handler file.
Includes security hardening and safe defaults.
"""

from flask import Flask, render_template
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv
import os

# Import our dedicated analyzers
from image_analyzer import image_analyzer
from live_analyzer import live_analyzer
from face_registration import face_registration
from authentication import authentication
from restricted_pages import restricted_pages

# Initialize app
app = Flask(__name__)

# --- Load environment variables securely ---
load_dotenv()

# --- Security Configurations ---
app.config.update(
    DEBUG=False,  # Disable debug in production
    THREADED=True,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB file limit
    SEND_FILE_MAX_AGE_DEFAULT=31536000,  # Cache static files for 1 year
    SESSION_COOKIE_HTTPONLY=True,  # JS cannot access cookies
    SESSION_COOKIE_SECURE=True,    # Only send cookies via HTTPS
    SESSION_COOKIE_SAMESITE='Lax', # Prevent CSRF via third-party sites
    TEMPLATES_AUTO_RELOAD=False,
    PERMANENT_SESSION_LIFETIME=1800,  # 30 min session expiry
)

# --- Secure headers middleware ---
@app.after_request
def apply_security_headers(response):
    """Add strict HTTP security headers."""
    response.headers["X-Frame-Options"] = "DENY"  # Prevent clickjacking
    response.headers["X-Content-Type-Options"] = "nosniff"  # Prevent MIME sniffing
    response.headers["X-XSS-Protection"] = "1; mode=block"  # XSS filter (legacy)
    response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), fullscreen=(self)"
    )
    return response

# --- Trusted proxy fix (if behind reverse proxy like Nginx) ---
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# --- Register Blueprints ---
app.register_blueprint(image_analyzer)
app.register_blueprint(live_analyzer)
app.register_blueprint(face_registration)
app.register_blueprint(authentication)
app.register_blueprint(restricted_pages)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_image')
def image_analysis_page():
    return render_template('image_analysis.html')

@app.route('/live_analysis')
def live_analysis_page():
    return render_template('live_analysis.html')

@app.route('/batch_analysis')
def batch_analysis_page():
    return render_template('batch_analysis.html')

@app.route('/register')
def registration_page():
    return render_template('face_registration.html')


if __name__ == '__main__':
    # Disable Flask’s built-in server in production — use Gunicorn instead
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )

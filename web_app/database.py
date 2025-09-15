"""
Database management for FaceGuard AI authentication system.
"""

import sqlite3
import os
import json
import hashlib
import base64
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Load .env as early as possible so secrets are available before DB init
try:
    from dotenv import load_dotenv  # type: ignore
    import os as _os
    # Load from project root and from this directory; latter takes precedence
    load_dotenv()
    load_dotenv(_os.path.join(_os.path.dirname(__file__), '.env'))
except Exception:
    pass

class UserDatabase:
    """Database manager for user face registration and authentication."""
    
    def __init__(self, db_path: str = "database/users.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = db_path
        self._fernet = self._init_crypto()
        
        # Ensure database directory exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Created database directory: {db_dir}")
        
        self.init_database()
    
    def init_database(self):
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Users table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE,
                        full_name TEXT,
                        face_encoding BLOB,
                        face_features TEXT,
                        registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Authentication sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS auth_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        session_token TEXT UNIQUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                # Authentication attempts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS auth_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        attempt_type TEXT, -- 'face_match', 'blink_detection', 'success', 'failure'
                        success BOOLEAN,
                        confidence_score REAL,
                        attempt_data TEXT, -- JSON data about the attempt
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (user_id) REFERENCES users (id)
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def register_user(self, username: str, email: str, full_name: str, 
                     face_encoding: bytes, face_features: Dict) -> int:
        """Register a new user with face data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if user already exists
                cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", 
                             (username, email))
                if cursor.fetchone():
                    raise ValueError("User already exists")
                
                # Insert new user
                enc_email = self._encrypt_text(email)
                enc_full  = self._encrypt_text(full_name)
                enc_face = self._encrypt_bytes(face_encoding)
                enc_features = self._encrypt_text(json.dumps(face_features))
                cursor.execute('''
                    INSERT INTO users (username, email, full_name, face_encoding, face_features)
                    VALUES (?, ?, ?, ?, ?)
                ''', (username, enc_email, enc_full, enc_face, enc_features))
                
                user_id = cursor.lastrowid
                conn.commit()
                
                logger.info(f"User {username} registered successfully with ID {user_id}")
                return user_id
                
        except Exception as e:
            logger.error(f"User registration failed: {e}")
            raise
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user data by username."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, username, email, full_name, face_encoding, face_features,
                           registration_date, last_login, is_active
                    FROM users WHERE username = ? AND is_active = 1
                ''', (username,))
                
                row = cursor.fetchone()
                if row:
                    email = self._decrypt_text(row[2]) if row[2] else row[2]
                    full  = self._decrypt_text(row[3]) if row[3] else row[3]
                    face_enc = self._decrypt_bytes(row[4]) if row[4] is not None else None
                    face_feat = self._decrypt_text(row[5]) if row[5] is not None else None
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': email,
                        'full_name': full,
                        'face_encoding': face_enc,
                        'face_features': json.loads(face_feat) if face_feat else {},
                        'registration_date': row[6],
                        'last_login': row[7],
                        'is_active': bool(row[8])
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user {username}: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """Get user data by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, username, email, full_name, face_encoding, face_features,
                           registration_date, last_login, is_active
                    FROM users WHERE id = ? AND is_active = 1
                ''', (user_id,))
                
                row = cursor.fetchone()
                if row:
                    email = self._decrypt_text(row[2]) if row[2] else row[2]
                    full  = self._decrypt_text(row[3]) if row[3] else row[3]
                    face_enc = self._decrypt_bytes(row[4]) if row[4] is not None else None
                    face_feat = self._decrypt_text(row[5]) if row[5] is not None else None
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': email,
                        'full_name': full,
                        'face_encoding': face_enc,
                        'face_features': json.loads(face_feat) if face_feat else {},
                        'registration_date': row[6],
                        'last_login': row[7],
                        'is_active': bool(row[8])
                    }
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all active users."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, username, email, full_name, registration_date, last_login
                    FROM users WHERE is_active = 1 ORDER BY username
                ''')
                
                users = []
                for row in cursor.fetchall():
                    users.append({
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[3],
                        'registration_date': row[4],
                        'last_login': row[5]
                    })
                
                return users
                
        except Exception as e:
            logger.error(f"Failed to get all users: {e}")
            return []
    
    def update_last_login(self, user_id: int):
        """Update user's last login timestamp."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                ''', (user_id,))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update last login for user {user_id}: {e}")
    
    def create_session(self, user_id: int, expires_hours: int = 24) -> str:
        """Create a new authentication session."""
        try:
            import secrets
            session_token = secrets.token_urlsafe(32)
            # store a hashed form in the DB to reduce token exposure; keep
            # backward-compat by encoding with a prefix
            token_hash = hashlib.sha256(session_token.encode('utf-8')).hexdigest()
            stored_token = f"sha256:{token_hash}"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Deactivate old sessions for this user
                cursor.execute('''
                    UPDATE auth_sessions SET is_active = 0 WHERE user_id = ?
                ''', (user_id,))
                
                # Create new session (store hash-prefixed token)
                cursor.execute('''
                    INSERT INTO auth_sessions (user_id, session_token, expires_at)
                    VALUES (?, ?, datetime('now', '+{} hours'))
                '''.format(expires_hours), (user_id, stored_token))
                
                conn.commit()
                
                logger.info(f"Session created for user {user_id}")
                return session_token
                
        except Exception as e:
            logger.error(f"Failed to create session for user {user_id}: {e}")
            raise
    
    def validate_session(self, session_token: str) -> Optional[Dict]:
        """Validate session token and return user data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT u.id, u.username, u.email, u.full_name, s.expires_at, s.session_token
                    FROM users u
                    JOIN auth_sessions s ON u.id = s.user_id
                    WHERE s.is_active = 1 AND u.is_active = 1 AND s.expires_at > datetime('now')
                ''')
                rows = cursor.fetchall()
                for row in rows:
                    stored = row[5] or ''
                    if stored.startswith('sha256:'):
                        if hashlib.sha256(session_token.encode('utf-8')).hexdigest() == stored.split(':',1)[1]:
                            return {
                                'id': row[0], 'username': row[1], 'email': row[2], 'full_name': row[3], 'expires_at': row[4]
                            }
                    else:
                        if stored == session_token:
                            return {
                                'id': row[0], 'username': row[1], 'email': row[2], 'full_name': row[3], 'expires_at': row[4]
                            }
                return None
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'full_name': row[3],
                        'expires_at': row[4]
                    }
                return None
                
        except Exception as e:
            logger.error(f"Session validation failed: {e}")
            return None
    
    def logout_session(self, session_token: str):
        """Deactivate a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Deactivate by matching either plaintext or hash-prefixed stored token
                token_hash = hashlib.sha256(session_token.encode('utf-8')).hexdigest()
                cursor.execute('''
                    UPDATE auth_sessions SET is_active = 0 
                    WHERE session_token = ? OR session_token = ?
                ''', (session_token, f"sha256:{token_hash}"))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to logout session: {e}")
    
    def log_auth_attempt(self, user_id: int, attempt_type: str, success: bool, 
                        confidence_score: float = None, attempt_data: Dict = None):
        """Log an authentication attempt."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO auth_attempts (user_id, attempt_type, success, confidence_score, attempt_data)
                    VALUES (?, ?, ?, ?, ?)
                ''', (user_id, attempt_type, success, confidence_score, 
                     json.dumps(attempt_data) if attempt_data else None))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log auth attempt: {e}")
    
    def get_auth_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        """Get authentication history for a user."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT attempt_type, success, confidence_score, attempt_data, timestamp
                    FROM auth_attempts 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (user_id, limit))
                
                history = []
                for row in cursor.fetchall():
                    history.append({
                        'attempt_type': row[0],
                        'success': bool(row[1]),
                        'confidence_score': row[2],
                        'attempt_data': json.loads(row[3]) if row[3] else {},
                        'timestamp': row[4]
                    })
                
                return history
                
        except Exception as e:
            logger.error(f"Failed to get auth history for user {user_id}: {e}")
            return []

    # ----------------------- crypto helpers -----------------------
    def _init_crypto(self):
        """Initialize Fernet from FG_DB_SECRET (32 urlsafe b64 key) or passphrase.
        Returns a fernet-like object with encrypt/decrypt or None if unavailable.
        """
        try:
            from cryptography.fernet import Fernet
        except Exception:
            logger.warning("cryptography not available; DB encryption disabled")
            return None

        key_b64 = os.getenv('FG_DB_SECRET', '').strip()
        if not key_b64:
            passphrase = os.getenv('FG_DB_SECRET_PASSPHRASE', '').encode('utf-8')
            if not passphrase:
                logger.warning("FG_DB_SECRET not set; DB encryption disabled")
                return None
            # Derive a 32-byte key via SHA256 (simple; in production use PBKDF2)
            key = hashlib.sha256(passphrase).digest()
            key_b64 = base64.urlsafe_b64encode(key).decode('ascii')
        try:
            return Fernet(key_b64)
        except Exception as e:
            logger.error(f"Invalid FG_DB_SECRET: {e}")
            return None

    def _encrypt_bytes(self, data: bytes) -> bytes:
        if not data:
            return data
        if self._fernet is None:
            return data
        ct = self._fernet.encrypt(data)
        return b'enc:' + base64.b64encode(ct)

    def _decrypt_bytes(self, data: bytes) -> bytes:
        if data is None:
            return data
        if isinstance(data, str):
            data = data.encode('utf-8')
        if not data.startswith(b'enc:') or self._fernet is None:
            return data
        try:
            blob = base64.b64decode(data[4:])
            return self._fernet.decrypt(blob)
        except Exception:
            return b''

    def _encrypt_text(self, text: str) -> str:
        if text is None:
            return None
        if self._fernet is None:
            return text
        ct = self._fernet.encrypt(text.encode('utf-8'))
        return 'enc:' + base64.b64encode(ct).decode('ascii')

    def _decrypt_text(self, text: str) -> str:
        if text is None:
            return None
        if not isinstance(text, str) or not text.startswith('enc:') or self._fernet is None:
            return text
        try:
            blob = base64.b64decode(text[4:].encode('ascii'))
            pt = self._fernet.decrypt(blob)
            return pt.decode('utf-8')
        except Exception:
            return ''

# Global database instance
db = UserDatabase()

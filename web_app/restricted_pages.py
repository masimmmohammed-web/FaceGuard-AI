"""
Restricted pages system for authenticated users.
"""

from flask import Blueprint, request, jsonify, render_template, session
import json
import logging
from functools import wraps

from database import db
from authentication import require_auth

logger = logging.getLogger(__name__)

# Create Blueprint for restricted pages
restricted_pages = Blueprint('restricted_pages', __name__)

def get_current_user():
    """Get current authenticated user from session token."""
    try:
        session_token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not session_token:
            return None
        
        user_data = db.validate_session(session_token)
        return user_data
    except Exception as e:
        logger.error(f"Failed to get current user: {e}")
        return None

@restricted_pages.route('/dashboard')
def dashboard():
    """Main dashboard for authenticated users."""
    return render_template('dashboard.html')

@restricted_pages.route('/profile')
def profile():
    """User profile page."""
    return render_template('profile.html')

@restricted_pages.route('/settings')
def settings():
    """User settings page."""
    return render_template('settings.html')

@restricted_pages.route('/api/user_info')
@require_auth
def get_user_info():
    """Get current user information."""
    try:
        user_data = request.user_data
        
        # Get additional user data from database
        full_user_data = db.get_user_by_id(user_data['id'])
        if not full_user_data:
            return jsonify({'error': 'User not found'}), 404
        
        # Get authentication history
        auth_history = db.get_auth_history(user_data['id'], limit=10)
        
        return jsonify({
            'success': True,
            'user': {
                'id': full_user_data['id'],
                'username': full_user_data['username'],
                'email': full_user_data['email'],
                'full_name': full_user_data['full_name'],
                'registration_date': full_user_data['registration_date'],
                'last_login': full_user_data['last_login']
            },
            'auth_history': auth_history
        })
        
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        return jsonify({'error': 'Failed to retrieve user information'}), 500

@restricted_pages.route('/api/update_profile', methods=['POST'])
@require_auth
def update_profile():
    """Update user profile information."""
    try:
        user_data = request.user_data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input
        email = data.get('email', '').strip()
        full_name = data.get('full_name', '').strip()
        
        if not email and not full_name:
            return jsonify({'error': 'At least one field must be provided'}), 400
        
        # Update user in database
        with db.db_path as conn:
            cursor = conn.cursor()
            
            update_fields = []
            update_values = []
            
            if email:
                update_fields.append('email = ?')
                update_values.append(email)
            
            if full_name:
                update_fields.append('full_name = ?')
                update_values.append(full_name)
            
            if update_fields:
                update_values.append(user_data['id'])
                query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
                cursor.execute(query, update_values)
                conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Profile updated successfully'
        })
        
    except Exception as e:
        logger.error(f"Failed to update profile: {e}")
        return jsonify({'error': 'Failed to update profile'}), 500

@restricted_pages.route('/api/auth_history')
@require_auth
def get_auth_history():
    """Get user's authentication history."""
    try:
        user_data = request.user_data
        limit = request.args.get('limit', 20, type=int)
        
        auth_history = db.get_auth_history(user_data['id'], limit=limit)
        
        return jsonify({
            'success': True,
            'auth_history': auth_history,
            'count': len(auth_history)
        })
        
    except Exception as e:
        logger.error(f"Failed to get auth history: {e}")
        return jsonify({'error': 'Failed to retrieve authentication history'}), 500

@restricted_pages.route('/api/change_password', methods=['POST'])
@require_auth
def change_password():
    """Change user password (placeholder for future implementation)."""
    try:
        user_data = request.user_data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        current_password = data.get('current_password', '')
        new_password = data.get('new_password', '')
        
        if not current_password or not new_password:
            return jsonify({'error': 'Current and new passwords are required'}), 400
        
        # For now, just return success (password functionality not implemented)
        return jsonify({
            'success': True,
            'message': 'Password change functionality not yet implemented'
        })
        
    except Exception as e:
        logger.error(f"Failed to change password: {e}")
        return jsonify({'error': 'Failed to change password'}), 500

@restricted_pages.route('/api/delete_account', methods=['POST'])
@require_auth
def delete_account():
    """Delete user account."""
    try:
        user_data = request.user_data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        confirm = data.get('confirm', False)
        if not confirm:
            return jsonify({'error': 'Account deletion must be confirmed'}), 400
        
        # Deactivate user account (soft delete)
        with db.db_path as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users SET is_active = 0 WHERE id = ?
            ''', (user_data['id'],))
            
            # Deactivate all sessions
            cursor.execute('''
                UPDATE auth_sessions SET is_active = 0 WHERE user_id = ?
            ''', (user_data['id'],))
            
            conn.commit()
        
        return jsonify({
            'success': True,
            'message': 'Account deleted successfully'
        })
        
    except Exception as e:
        logger.error(f"Failed to delete account: {e}")
        return jsonify({'error': 'Failed to delete account'}), 500

@restricted_pages.route('/api/system_status')
@require_auth
def get_system_status():
    """Get system status information."""
    try:
        user_data = request.user_data
        
        # Get system statistics
        total_users = len(db.get_all_users())
        
        # Get user's recent activity
        auth_history = db.get_auth_history(user_data['id'], limit=5)
        
        return jsonify({
            'success': True,
            'system_info': {
                'total_users': total_users,
                'user_registration_date': user_data.get('registration_date'),
                'last_login': user_data.get('last_login')
            },
            'recent_activity': auth_history
        })
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return jsonify({'error': 'Failed to retrieve system status'}), 500

# Database Directory

This directory contains the SQLite database files for the FaceGuard AI authentication system.

## Files

- **`users.db`** - Main database containing:
  - User registration data
  - Face encodings and features
  - Authentication sessions
  - Authentication attempt history

## Database Schema

### Users Table
- `id` - Primary key
- `username` - Unique username
- `email` - User email address
- `full_name` - User's full name
- `face_encoding` - Binary face data
- `face_features` - JSON face feature data
- `registration_date` - When user registered
- `last_login` - Last login timestamp
- `is_active` - Account status

### Auth Sessions Table
- `id` - Primary key
- `user_id` - Foreign key to users table
- `session_token` - Unique session token
- `created_at` - Session creation time
- `expires_at` - Session expiration time
- `is_active` - Session status

### Auth Attempts Table
- `id` - Primary key
- `user_id` - Foreign key to users table
- `attempt_type` - Type of attempt (face_match, blink_detection, etc.)
- `success` - Whether attempt was successful
- `confidence_score` - Confidence score of attempt
- `attempt_data` - JSON data about the attempt
- `timestamp` - When attempt occurred

## Backup

To backup the database:
```bash
cp database/users.db database/users_backup_$(date +%Y%m%d_%H%M%S).db
```

## Security

- The database contains sensitive user data including face encodings
- Ensure proper file permissions are set
- Regular backups are recommended
- Consider encryption for production environments

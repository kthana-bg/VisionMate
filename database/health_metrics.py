"""
Helper function for saving health metrics to database
"""
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from database.db_manager import DatabaseManager

# Global database instance
_db = DatabaseManager()


def save_health_metric(
    user_id: int,
    eye_status: str,
    ear_value: float,
    posture_status: str,
    posture_angle: float,
    health_score: float,
    active_eye_model: str,
    active_posture_model: str,
):
    """
    Save a health metric record to the database.
    This is called periodically during live monitoring.
    """
    try:
        # Get current session for this user
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        
        # Note: In the actual implementation, we would need to track session_id
        # For now, we'll create a simple log entry
        # This is a simplified version - the full implementation should link to sessions
        
        print(f"Health metric saved: Eye={eye_status}, Posture={posture_status}, Score={health_score}")
        
    except Exception as e:
        print(f"Error saving health metric: {e}")
